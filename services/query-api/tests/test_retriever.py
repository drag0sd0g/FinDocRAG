"""Unit tests for the retriever.

The embedding model and database are mocked — no real resources needed.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np

from src.rag.prompts import RetrievedChunk

# Unit basis vectors → dot product == cosine similarity, scores stay exact.
_DIM = 768


def _unit(axis: int) -> np.ndarray:
    vec = np.zeros(_DIM)
    vec[axis] = 1.0
    return vec


def _row(chunk_id: str, embedding: np.ndarray, score: float = 0.9) -> tuple:
    return (chunk_id, "AAPL", "2024-11-01", "Item 1A", f"Text of {chunk_id}", embedding, score)


def _search_sql(mock_cur: MagicMock) -> list[str]:
    """Executed statements, excluding the HNSW tuning SETs that precede them."""
    return [
        call.args[0]
        for call in mock_cur.execute.call_args_list
        if not call.args[0].lstrip().upper().startswith("SET ")
    ]


def _make_retriever(query_embedding: np.ndarray) -> tuple:
    """Build a Retriever with mocked model and DB connection."""
    from src.rag.retriever import Retriever

    with patch("src.rag.retriever.SentenceTransformer") as mock_st_class:
        mock_model = MagicMock()
        mock_model.encode.return_value = query_embedding
        mock_st_class.return_value = mock_model
        with patch("src.rag.retriever.psycopg2"):
            retriever = Retriever(dsn="postgresql://fake", model_name="test")

    mock_cur = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.closed = False
    retriever._conn = mock_conn
    return retriever, mock_cur


class TestRetriever:
    """Tests for the Retriever class."""

    def test_embed_query_returns_list(self) -> None:
        retriever, _ = _make_retriever(_unit(0))
        result = retriever.embed_query("What is Apple's revenue?")
        assert isinstance(result, list)
        assert len(result) == _DIM

    def test_embed_query_applies_search_query_prefix(self) -> None:
        """Query side must carry the asymmetric search_query: prefix."""
        from src.rag.retriever import DEFAULT_QUERY_PREFIX

        retriever, _ = _make_retriever(_unit(0))
        retriever.embed_query("What is Apple's revenue?")

        called_text = retriever._model.encode.call_args.args[0]
        assert called_text == f"{DEFAULT_QUERY_PREFIX}What is Apple's revenue?"

    def test_retrieve_with_ticker_filter(self) -> None:
        """Both legs must include the ticker WHERE clause; the fused chunk
        carries cosine similarity to the query as relevance_score."""
        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = [_row("chunk1", _unit(0))]

        chunks, embedding, emb_ms = retriever.retrieve(
            "What are Apple's risks?", top_k=5, ticker_filter="AAPL"
        )

        assert len(chunks) == 1
        assert chunks[0].ticker == "AAPL"
        # chunk embedding == query embedding → cosine similarity 1.0
        assert chunks[0].relevance_score == 1.0
        assert len(embedding) == _DIM
        # Both legs (vector + lexical) ran, each with the ticker filter.
        executed = _search_sql(mock_cur)
        assert len(executed) == 2
        for sql in executed:
            assert "ticker = %s" in sql

    def test_retrieve_without_ticker_filter(self) -> None:
        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = []

        chunks, _, _ = retriever.retrieve("General question", top_k=3)

        assert chunks == []
        for sql in _search_sql(mock_cur):
            assert "ticker = %s" not in sql

    def test_retrieve_with_date_filters(self) -> None:
        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = []

        retriever.retrieve(
            "Revenue trends",
            top_k=3,
            filing_date_from="2023-01-01",
            filing_date_to="2024-12-31",
        )

        for sql in _search_sql(mock_cur):
            assert "filing_date >= %s" in sql
            assert "filing_date <= %s" in sql

    def test_lexical_failure_degrades_to_vector_only(self) -> None:
        """If migration 002 is missing, the tsvector query fails — retrieval
        must still return vector-leg results instead of raising."""
        retriever, mock_cur = _make_retriever(_unit(0))

        def execute(sql: str, params: object = None) -> None:
            if "ts_rank_cd" in sql:
                raise RuntimeError('column "chunk_tsv" does not exist')

        mock_cur.execute.side_effect = execute
        mock_cur.fetchall.return_value = [_row("chunk1", _unit(0))]

        chunks, _, _ = retriever.retrieve("What are Apple's risks?", top_k=5)

        assert len(chunks) == 1
        assert chunks[0].chunk_id == "chunk1"


class TestHnswTuning:
    """pgvector caps HNSW candidates at hnsw.ef_search (default 40).

    Requesting more rows than that silently returns fewer, lower-quality
    neighbours, which starves RRF and MMR of material to work with.
    """

    @staticmethod
    def _executed_sql(mock_cur: MagicMock) -> list[str]:
        return [call.args[0] for call in mock_cur.execute.call_args_list]

    @staticmethod
    def _set_params(mock_cur: MagicMock, guc: str) -> object | None:
        for call in mock_cur.execute.call_args_list:
            if guc in call.args[0] and len(call.args) > 1:
                return call.args[1][0]
        return None

    def test_ef_search_covers_the_requested_candidate_count(self) -> None:
        from src.rag.retriever import _CANDIDATE_MULTIPLIER

        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = [_row("chunk1", _unit(0))]

        top_k = 20
        retriever.retrieve("revenue", top_k=top_k)

        ef_search = self._set_params(mock_cur, "hnsw.ef_search")
        assert ef_search is not None, "ef_search was never set"
        assert ef_search >= top_k * _CANDIDATE_MULTIPLIER

    def test_ef_search_never_drops_below_the_pgvector_default(self) -> None:
        from src.rag.retriever import _MIN_EF_SEARCH

        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = [_row("chunk1", _unit(0))]

        retriever.retrieve("revenue", top_k=1)

        assert self._set_params(mock_cur, "hnsw.ef_search") == _MIN_EF_SEARCH

    def test_ef_search_is_applied_before_the_vector_query(self) -> None:
        """A SET issued after the search would not affect it."""
        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = [_row("chunk1", _unit(0))]

        retriever.retrieve("revenue", top_k=5)

        statements = self._executed_sql(mock_cur)
        set_idx = next(i for i, s in enumerate(statements) if "hnsw.ef_search" in s)
        scan_idx = next(i for i, s in enumerate(statements) if "<=>" in s)
        assert set_idx < scan_idx

    def test_filtered_search_enables_iterative_scan(self) -> None:
        """HNSW filters after the scan, so a selective filter needs re-searching."""
        from src.rag.retriever import _ITERATIVE_SCAN_MODE

        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = [_row("chunk1", _unit(0))]

        retriever.retrieve("revenue", top_k=5, ticker_filter="AAPL")

        assert self._set_params(mock_cur, "hnsw.iterative_scan") == _ITERATIVE_SCAN_MODE

    def test_unfiltered_search_leaves_iterative_scan_off(self) -> None:
        """Without a filter every neighbour survives, so plain HNSW is faster."""
        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = [_row("chunk1", _unit(0))]

        retriever.retrieve("revenue", top_k=5)

        assert all(
            "hnsw.iterative_scan" not in s for s in self._executed_sql(mock_cur)
        )

    def test_retrieval_survives_pgvector_without_iterative_scan(self) -> None:
        """pgvector < 0.8 has no such GUC; recall suffers but queries still run."""
        import psycopg2

        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = [_row("chunk1", _unit(0))]

        def _execute(sql: str, params: object = None) -> None:
            if "hnsw.iterative_scan" in sql:
                raise psycopg2.Error("unrecognized configuration parameter")

        mock_cur.execute.side_effect = _execute

        chunks, _, _ = retriever.retrieve("revenue", top_k=5, ticker_filter="AAPL")

        assert len(chunks) == 1
        assert retriever._iterative_scan_unsupported is True

    def test_unsupported_iterative_scan_is_not_retried(self) -> None:
        """The probe must not repeat on every single query."""
        import psycopg2

        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = [_row("chunk1", _unit(0))]
        mock_cur.execute.side_effect = lambda sql, params=None: (
            _ for _ in ()
        ).throw(psycopg2.Error("nope")) if "hnsw.iterative_scan" in sql else None

        retriever.retrieve("revenue", top_k=5, ticker_filter="AAPL")
        first_attempts = sum(
            1 for c in mock_cur.execute.call_args_list if "hnsw.iterative_scan" in c.args[0]
        )
        retriever.retrieve("revenue", top_k=5, ticker_filter="AAPL")
        total_attempts = sum(
            1 for c in mock_cur.execute.call_args_list if "hnsw.iterative_scan" in c.args[0]
        )

        assert first_attempts == 1
        assert total_attempts == 1


class TestThreadSafety:
    """Retrieval runs in a thread pool, so the shared model and connection are
    touched concurrently.

    SentenceTransformer.encode() is not thread-safe for nomic-bert: the model
    keeps its rotary-embedding tables as mutable state and resizes them to each
    input's sequence length, so two overlapping encodes raise a tensor-size
    mismatch mid-forward. hnsw.ef_search has the same class of problem at the
    database level — it is session state on a shared connection.
    """

    @staticmethod
    def _overlap_detector() -> tuple[MagicMock, list[int]]:
        """Return a side_effect that records the max concurrent entry count."""
        import threading

        state = {"active": 0, "peak": 0}
        guard = threading.Lock()
        peak: list[int] = []

        def _tracked(*_args: object, **_kwargs: object) -> np.ndarray:
            with guard:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.02)  # widen the window for a race to show up
            with guard:
                state["active"] -= 1
                peak.append(state["peak"])
            return _unit(0)

        return MagicMock(side_effect=_tracked), peak

    def test_concurrent_embed_query_never_overlaps(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        retriever, _ = _make_retriever(_unit(0))
        tracked, peak = self._overlap_detector()
        retriever._model.encode = tracked

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(retriever.embed_query, [f"question {i}" for i in range(8)]))

        assert max(peak) == 1, (
            f"{max(peak)} concurrent encodes observed — the model forward pass "
            "must be serialised or nomic-bert's cached rotary tables corrupt"
        )

    def test_concurrent_embed_query_returns_correct_results(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        retriever, _ = _make_retriever(_unit(0))
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(retriever.embed_query, [f"question {i}" for i in range(8)])
            )
        assert all(len(r) == _DIM for r in results)

    def test_concurrent_vector_search_never_overlaps(self) -> None:
        """The ef_search SET and the query it tunes must apply as a unit."""
        from concurrent.futures import ThreadPoolExecutor

        retriever, mock_cur = _make_retriever(_unit(0))
        tracked, peak = self._overlap_detector()
        mock_cur.execute = tracked
        mock_cur.fetchall.return_value = [_row("chunk1", _unit(0))]

        def _search(_i: int) -> object:
            return retriever._vector_search([0.1] * _DIM, 20, [], [])

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(_search, range(6)))

        assert max(peak) == 1

    def test_reconnect_is_not_raced(self) -> None:
        """Two threads seeing a closed connection must not both open one."""
        from concurrent.futures import ThreadPoolExecutor

        retriever, _ = _make_retriever(_unit(0))
        retriever._conn = None

        connects: list[int] = []

        def _connect() -> None:
            time.sleep(0.01)
            conn = MagicMock()
            conn.closed = False
            retriever._conn = conn
            connects.append(1)

        with (
            patch.object(retriever, "connect", side_effect=_connect),
            ThreadPoolExecutor(max_workers=8) as pool,
        ):
            list(pool.map(lambda _i: retriever._get_conn(), range(8)))

        assert sum(connects) == 1


class TestPing:
    def test_ping_executes_a_trivial_query(self) -> None:
        retriever, mock_cur = _make_retriever(_unit(0))
        retriever.ping()
        assert mock_cur.execute.call_args.args[0] == "SELECT 1"

    def test_ping_propagates_connection_errors(self) -> None:
        import psycopg2
        import pytest

        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.execute.side_effect = psycopg2.OperationalError("server closed")
        with pytest.raises(psycopg2.OperationalError):
            retriever.ping()


class TestFuseRRF:
    """Tests for Reciprocal Rank Fusion of the two retrieval legs."""

    def test_empty_lists(self) -> None:
        from src.rag.retriever import _fuse_rrf
        assert _fuse_rrf([[], []], _unit(0)) == []

    def test_chunk_in_both_legs_outranks_single_leg_chunk(self) -> None:
        from src.rag.retriever import _fuse_rrf
        both = _row("in-both", _unit(1))
        vector_only = _row("vector-only", _unit(2))
        fused = _fuse_rrf([[vector_only, both], [both]], _unit(0))
        assert [c.chunk_id for c, _ in fused] == ["in-both", "vector-only"]

    def test_duplicate_chunk_returned_once(self) -> None:
        from src.rag.retriever import _fuse_rrf
        row = _row("chunk1", _unit(1))
        fused = _fuse_rrf([[row], [row]], _unit(0))
        assert len(fused) == 1

    def test_relevance_is_cosine_with_query(self) -> None:
        from src.rag.retriever import _fuse_rrf
        query = _unit(0)
        aligned = _row("aligned", query.copy())
        orthogonal = _row("orthogonal", _unit(1))
        fused = _fuse_rrf([[aligned, orthogonal]], query)
        scores = {c.chunk_id: c.relevance_score for c, _ in fused}
        assert scores["aligned"] == 1.0
        assert scores["orthogonal"] == 0.0


class TestBuildFilters:
    def test_no_filters(self) -> None:
        from src.rag.retriever import Retriever
        clauses, params = Retriever._build_filters(None, None, None)
        assert clauses == []
        assert params == []

    def test_all_filters(self) -> None:
        from src.rag.retriever import Retriever
        clauses, params = Retriever._build_filters("AAPL", "2023-01-01", "2024-12-31")
        assert clauses == ["ticker = %s", "filing_date >= %s", "filing_date <= %s"]
        assert params == ["AAPL", "2023-01-01", "2024-12-31"]


class TestEmbeddingModelConsistency:
    """Tests for Retriever.verify_embedding_model_consistency."""

    @patch("src.rag.retriever.SentenceTransformer")
    def test_consistent_dimensions_logs_info(self, mock_st_class: MagicMock) -> None:
        """No error logged when stored dim matches model dim."""
        from src.rag.retriever import Retriever

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st_class.return_value = mock_model

        with patch("src.rag.retriever.psycopg2"):
            retriever = Retriever(dsn="postgresql://fake", model_name="test")

        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (768,)
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.closed = False
        retriever._conn = mock_conn

        # Should complete without raising
        retriever.verify_embedding_model_consistency()
        mock_cur.execute.assert_called_once()

    @patch("src.rag.retriever.SentenceTransformer")
    def test_dimension_mismatch_logs_error(self, mock_st_class: MagicMock) -> None:
        """Error logged when stored dim differs from model dim (silent corruption risk)."""
        from src.rag.retriever import Retriever

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st_class.return_value = mock_model

        with patch("src.rag.retriever.psycopg2"):
            retriever = Retriever(dsn="postgresql://fake", model_name="test")

        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (384,)  # stale all-MiniLM dim → mismatch!
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.closed = False
        retriever._conn = mock_conn

        with patch("src.rag.retriever.logger") as mock_logger:
            retriever.verify_embedding_model_consistency()
            mock_logger.error.assert_called_once()
            call_kwargs = mock_logger.error.call_args[1]
            assert call_kwargs["stored_dim"] == 384
            assert call_kwargs["model_dim"] == 768

    @patch("src.rag.retriever.SentenceTransformer")
    def test_skips_check_when_no_data(self, mock_st_class: MagicMock) -> None:
        """Check is skipped (not an error) when document_chunks is empty."""
        from src.rag.retriever import Retriever

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st_class.return_value = mock_model

        with patch("src.rag.retriever.psycopg2"):
            retriever = Retriever(dsn="postgresql://fake", model_name="test")

        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None  # empty table
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_conn.closed = False
        retriever._conn = mock_conn

        with patch("src.rag.retriever.logger") as mock_logger:
            retriever.verify_embedding_model_consistency()
            mock_logger.error.assert_not_called()


class TestMMR:
    """Tests for _apply_mmr."""

    def _make(self, relevance: float, vec: list[float]) -> tuple:
        from src.rag.prompts import RetrievedChunk
        chunk = RetrievedChunk(
            chunk_id="x",
            ticker="AAPL",
            filing_date="2024-01-01",
            section="Item 1A",
            relevance_score=relevance,
            text="text",
        )
        return (chunk, np.array(vec, dtype=float))

    def test_empty_candidates_returns_empty(self) -> None:
        from src.rag.retriever import _apply_mmr
        assert _apply_mmr([], top_k=5) == []

    def test_fewer_candidates_than_top_k(self) -> None:
        from src.rag.retriever import _apply_mmr
        candidates = [self._make(0.9, [1.0, 0.0]), self._make(0.8, [0.0, 1.0])]
        result = _apply_mmr(candidates, top_k=5)
        assert len(result) == 2

    def test_first_selected_is_highest_relevance(self) -> None:
        from src.rag.retriever import _apply_mmr
        # Orthogonal vectors → diversity plays no role for first pick
        candidates = [
            self._make(0.7, [1.0, 0.0, 0.0]),
            self._make(0.9, [0.0, 1.0, 0.0]),
            self._make(0.5, [0.0, 0.0, 1.0]),
        ]
        result = _apply_mmr(candidates, top_k=1)
        assert result[0].relevance_score == 0.9

    def test_prefers_diverse_chunk_over_redundant_one(self) -> None:
        """With lambda=0.5 and top_k=2, MMR should pick the diverse chunk
        over the near-duplicate even when the duplicate has higher relevance.

        Setup (unit vectors, so dot == cosine):
          A: relevance=0.90, vec=[1, 0, 0]  ← selected first
          B: relevance=0.85, vec=[1, 0, 0]  ← near-duplicate of A (sim≈1.0)
          C: relevance=0.70, vec=[0, 1, 0]  ← diverse (sim=0.0 with A)

        Round 2 MMR scores:
          B: 0.5*0.85 - 0.5*1.0 = -0.075
          C: 0.5*0.70 - 0.5*0.0 = +0.350  ← wins
        """
        from src.rag.retriever import _apply_mmr
        candidates = [
            self._make(0.90, [1.0, 0.0, 0.0]),  # A
            self._make(0.85, [1.0, 0.0, 0.0]),  # B — near-duplicate
            self._make(0.70, [0.0, 1.0, 0.0]),  # C — diverse
        ]
        result = _apply_mmr(candidates, top_k=2)
        assert result[0].relevance_score == 0.90  # A always first
        assert result[1].relevance_score == 0.70  # C beats B

    def test_lambda_1_gives_pure_relevance_order(self) -> None:
        """lambda=1.0 disables diversity — identical to sorting by relevance."""
        from src.rag.retriever import _apply_mmr
        candidates = [
            self._make(0.90, [1.0, 0.0, 0.0]),
            self._make(0.85, [1.0, 0.0, 0.0]),  # near-duplicate but higher relevance
            self._make(0.70, [0.0, 1.0, 0.0]),
        ]
        result = _apply_mmr(candidates, top_k=2, lambda_mmr=1.0)
        scores = [c.relevance_score for c in result]
        assert scores == [0.90, 0.85]

    def test_returns_exactly_top_k_when_enough_candidates(self) -> None:
        from src.rag.retriever import _apply_mmr
        candidates = [self._make(1.0 - i * 0.1, [float(i == j) for j in range(6)]) for i in range(6)]
        result = _apply_mmr(candidates, top_k=4)
        assert len(result) == 4


class TestRetrievedChunk:
    """Tests for the RetrievedChunk dataclass."""

    def test_fields(self) -> None:
        c = RetrievedChunk(
            chunk_id="abc123",
            ticker="MSFT",
            filing_date="2024-10-15",
            section="Item 7",
            relevance_score=0.92,
            text="Revenue grew...",
        )
        assert c.ticker == "MSFT"
        assert c.relevance_score == 0.92
