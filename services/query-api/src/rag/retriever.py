"""Hybrid retriever — vector search fused with Postgres full-text search.

Retrieval runs two legs over document_chunks and fuses them with
Reciprocal Rank Fusion (RRF):

  1. Vector leg   — query embedding vs pgvector HNSW (cosine distance),
                    strong on paraphrase and semantic similarity.
  2. Lexical leg  — websearch_to_tsquery over the chunk_tsv tsvector
                    column, strong on exact terms ("Intelligent Cloud",
                    "fiscal 2024") that embeddings blur.

The fused candidate pool is then reranked with Maximal Marginal
Relevance (MMR) to balance relevance and diversity.  If the lexical
column is missing (migration 002 not applied), retrieval degrades
gracefully to vector-only.

References:
  - TDD: FR-13 (embed query, retrieve top-k, optional ticker filter)
  - TDD: FR-19, FR-20 (list ingested filings)
  - TDD: NFR-1 (retrieval within 200ms at p99)
  - Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
    outperforms Condorcet and individual rank learning methods", SIGIR.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

import numpy as np
import psycopg2
import structlog
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from src.rag.prompts import RetrievedChunk

logger = structlog.get_logger()

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# nomic is an asymmetric-retrieval model: the query side is prefixed with
# ``search_query:`` (documents get ``search_document:`` in the embedding
# worker). The two prefixes MUST match the ingestion side or retrieval degrades.
DEFAULT_QUERY_PREFIX = "search_query: "

# How many candidates each leg fetches before fusion and MMR reranking.
# 4× top_k gives the algorithm enough diversity headroom without a
# meaningful latency cost (HNSW lookup is O(log n) regardless of LIMIT).
_CANDIDATE_MULTIPLIER = 4
_MAX_CANDIDATES = 100

# RRF constant from Cormack et al. (2009); dampens the weight of top ranks
# so one leg cannot dominate the fusion.
_RRF_K = 60

# Columns shared by both retrieval legs (score is appended per leg).
_CHUNK_COLUMNS = "chunk_id, ticker, filing_date, section_name, chunk_text, embedding"


def _apply_mmr(
    candidates: list[tuple[RetrievedChunk, np.ndarray]],
    top_k: int,
    lambda_mmr: float = 0.5,
) -> list[RetrievedChunk]:
    """Select top_k chunks using Maximal Marginal Relevance.

    Iteratively picks the chunk that best balances relevance to the query
    against redundancy with chunks already selected:

        score = λ × relevance_score − (1−λ) × max_cos_sim(chunk, selected)

    lambda_mmr=1.0 → pure relevance order (identical to no reranking).
    lambda_mmr=0.0 → pure diversity (ignores relevance entirely).
    lambda_mmr=0.5 → equal weight, the recommended default.

    Embeddings must be L2-normalised so that dot product == cosine similarity.

    Reference: Carbonell & Goldstein (1998), "The Use of MMR, Diversity-Based
    Reranking for Reordering Documents and Producing Summaries", SIGIR.
    """
    if not candidates:
        return []

    selected: list[RetrievedChunk] = []
    selected_vecs: list[np.ndarray] = []
    remaining = list(candidates)

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = float("-inf")

        for i, (chunk, emb) in enumerate(remaining):
            if not selected_vecs:
                # First pick is always the highest-relevance chunk.
                mmr_score = chunk.relevance_score
            else:
                max_sim = max(float(np.dot(emb, sel)) for sel in selected_vecs)
                mmr_score = lambda_mmr * chunk.relevance_score - (1 - lambda_mmr) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        chunk, emb = remaining.pop(best_idx)
        selected.append(chunk)
        selected_vecs.append(emb)

    return selected


def _fuse_rrf(
    ranked_lists: list[list[tuple[Any, ...]]],
    query_embedding: np.ndarray,
    k: int = _RRF_K,
) -> list[tuple[RetrievedChunk, np.ndarray]]:
    """Fuse ranked result lists with Reciprocal Rank Fusion.

    Each input row is (chunk_id, ticker, filing_date, section_name,
    chunk_text, embedding, leg_score).  A chunk appearing in several lists
    accumulates 1/(k + rank) per appearance; the fused pool is ordered by
    that sum, descending.

    The returned chunks carry cosine similarity to the query as
    relevance_score (comparable across legs, unlike leg-native scores),
    which is what MMR and the API response report.
    """
    rrf_scores: dict[str, float] = {}
    by_id: dict[str, tuple[RetrievedChunk, np.ndarray]] = {}

    for rows in ranked_lists:
        for rank, row in enumerate(rows, start=1):
            chunk_id = row[0]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in by_id:
                emb = np.asarray(row[5], dtype=float)
                chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    ticker=row[1],
                    filing_date=str(row[2]),
                    section=row[3],
                    # Embeddings are L2-normalised → dot == cosine similarity.
                    relevance_score=float(np.dot(emb, query_embedding)),
                    text=row[4],
                )
                by_id[chunk_id] = (chunk, emb)

    fused_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [by_id[cid] for cid in fused_ids]


class Retriever:
    """Embeds a query and retrieves the top-k most similar chunks."""

    def __init__(
        self,
        dsn: str,
        model_name: str = DEFAULT_MODEL,
        *,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
    ) -> None:
        self._dsn = dsn
        self._conn: psycopg2.extensions.connection | None = None
        self._query_prefix = query_prefix
        logger.info("retriever_loading_model", model=model_name)
        # trust_remote_code: nomic ships a custom BERT (nomic-bert-2048); the
        # flag is ignored by standard sentence-transformers models.
        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        logger.info("retriever_model_loaded", model=model_name)

    def connect(self) -> None:
        """Establish database connection and register pgvector type."""
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = True
        register_vector(self._conn)
        logger.info("retriever_db_connected")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _get_conn(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self.connect()
        if self._conn is None:
            raise RuntimeError("Failed to establish database connection")
        return self._conn

    @contextlib.contextmanager
    def _cursor(self) -> Generator[psycopg2.extensions.cursor, None, None]:
        """Context manager: yields a cursor and always closes it on exit."""
        cur = self._get_conn().cursor()
        try:
            yield cur
        finally:
            cur.close()

    def embed_query(self, question: str) -> list[float]:
        """Embed a user's question using the same model as ingestion (FR-13).

        The ``search_query:`` task prefix mirrors the ``search_document:``
        prefix applied to chunks at ingestion time (asymmetric retrieval).
        """
        embedding = self._model.encode(
            f"{self._query_prefix}{question}",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def verify_embedding_model_consistency(self) -> None:
        """Compare the loaded model's output dimension against stored embeddings.

        Called at startup to detect silent model mismatches — e.g. the embedding
        worker used nomic-embed-text (768-dim) but the query API is now configured
        with a different model. A dimension mismatch will corrupt all retrieval
        results silently at query time.

        Logs INFO when consistent, ERROR when a mismatch is detected. Skips
        the check when the database contains no embeddings yet (fresh install).
        """
        with self._cursor() as cur:
            cur.execute("SELECT vector_dims(embedding) FROM document_chunks LIMIT 1")
            row = cur.fetchone()

        if row is None:
            logger.info("embedding_consistency_check_skipped", reason="no_chunks_in_db")
            return

        stored_dim: int = row[0]
        model_dim = self._model.get_sentence_embedding_dimension()

        if stored_dim != model_dim:
            logger.error(
                "embedding_model_dimension_mismatch",
                stored_dim=stored_dim,
                model_dim=model_dim,
                advice="Re-embed all documents with the current model or restore the original model",
            )
        else:
            logger.info("embedding_model_consistent", dim=model_dim)

    @staticmethod
    def _build_filters(
        ticker_filter: str | None,
        filing_date_from: str | None,
        filing_date_to: str | None,
    ) -> tuple[list[str], list[Any]]:
        """Build optional WHERE clauses shared by both retrieval legs."""
        clauses: list[str] = []
        params: list[Any] = []
        if ticker_filter:
            clauses.append("ticker = %s")
            params.append(ticker_filter)
        if filing_date_from:
            clauses.append("filing_date >= %s")
            params.append(filing_date_from)
        if filing_date_to:
            clauses.append("filing_date <= %s")
            params.append(filing_date_to)
        return clauses, params

    def _vector_search(
        self,
        query_embedding: list[float],
        candidate_k: int,
        filter_clauses: list[str],
        filter_params: list[Any],
    ) -> list[tuple[Any, ...]]:
        """Vector leg: top candidates by cosine distance (HNSW)."""
        where = f"WHERE {' AND '.join(filter_clauses)}" if filter_clauses else ""
        sql = f"""
            SELECT {_CHUNK_COLUMNS},
                   1 - (embedding <=> %s::vector) AS score
            FROM document_chunks
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [query_embedding, *filter_params, query_embedding, candidate_k]
        with self._cursor() as cur:
            cur.execute(sql, params)
            rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows

    def _lexical_search(
        self,
        question: str,
        candidate_k: int,
        filter_clauses: list[str],
        filter_params: list[Any],
    ) -> list[tuple[Any, ...]]:
        """Lexical leg: top candidates by full-text rank.

        Returns [] when the tsvector column is missing (migration 002 not
        applied) or the question yields an empty tsquery — retrieval then
        degrades to vector-only.
        """
        clauses = ["chunk_tsv @@ websearch_to_tsquery('english', %s)", *filter_clauses]
        sql = f"""
            SELECT {_CHUNK_COLUMNS},
                   ts_rank_cd(chunk_tsv, websearch_to_tsquery('english', %s)) AS score
            FROM document_chunks
            WHERE {" AND ".join(clauses)}
            ORDER BY score DESC
            LIMIT %s
        """
        params = [question, question, *filter_params, candidate_k]
        try:
            with self._cursor() as cur:
                cur.execute(sql, params)
                rows: list[tuple[Any, ...]] = cur.fetchall()
            return rows
        except Exception as exc:
            logger.warning("lexical_search_unavailable", error=str(exc))
            return []

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        ticker_filter: str | None = None,
        filing_date_from: str | None = None,
        filing_date_to: str | None = None,
    ) -> tuple[list[RetrievedChunk], list[float], float]:
        """Embed the question and retrieve the top-k chunks (hybrid + MMR).

        Returns:
            (chunks, query_embedding, embedding_time_ms)
        """
        # Step 1: Embed the query
        t0 = time.perf_counter()
        query_embedding = self.embed_query(question)
        embedding_ms = (time.perf_counter() - t0) * 1000

        # Step 2: Run both legs — fetch more candidates than requested so
        # fusion + MMR have enough material to work with.
        candidate_k = min(top_k * _CANDIDATE_MULTIPLIER, _MAX_CANDIDATES)
        filter_clauses, filter_params = self._build_filters(
            ticker_filter, filing_date_from, filing_date_to
        )

        vector_rows = self._vector_search(
            query_embedding, candidate_k, filter_clauses, filter_params
        )
        lexical_rows = self._lexical_search(
            question, candidate_k, filter_clauses, filter_params
        )

        # Step 3: Fuse with RRF, keep the top candidate_k of the fused pool.
        candidates = _fuse_rrf(
            [vector_rows, lexical_rows], np.asarray(query_embedding)
        )[:candidate_k]

        # Step 4: Apply MMR to select top_k diverse chunks.
        chunks = _apply_mmr(candidates, top_k)

        logger.info(
            "retrieval_complete",
            top_k=top_k,
            vector_candidates=len(vector_rows),
            lexical_candidates=len(lexical_rows),
            fused_candidates=len(candidates),
            results=len(chunks),
            ticker_filter=ticker_filter,
            filing_date_from=filing_date_from,
            filing_date_to=filing_date_to,
            embedding_ms=round(embedding_ms, 1),
        )

        return chunks, query_embedding, embedding_ms

    def list_documents(
        self,
        ticker: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[tuple[Any, ...]], int]:
        """Return a page of ingested filings from ingestion_log (FR-19, FR-20)."""
        with self._cursor() as cur:
            if ticker:
                cur.execute(
                    "SELECT COUNT(*) FROM ingestion_log WHERE ticker = %s",
                    (ticker,),
                )
            else:
                cur.execute("SELECT COUNT(*) FROM ingestion_log")
            total: int = cur.fetchone()[0]

            if ticker:
                cur.execute(
                    """SELECT accession_number, ticker, company_name, filing_date,
                              filing_type, chunk_count, ingested_at
                       FROM ingestion_log
                       WHERE ticker = %s
                       ORDER BY filing_date DESC
                       LIMIT %s OFFSET %s""",
                    (ticker, limit, offset),
                )
            else:
                cur.execute(
                    """SELECT accession_number, ticker, company_name, filing_date,
                              filing_type, chunk_count, ingested_at
                       FROM ingestion_log
                       ORDER BY filing_date DESC
                       LIMIT %s OFFSET %s""",
                    (limit, offset),
                )
            rows = cur.fetchall()
        return rows, total
