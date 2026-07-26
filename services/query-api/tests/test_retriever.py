"""Unit tests for the retriever.

The embedding model and database are mocked — no real resources needed.
"""

from __future__ import annotations

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
        executed = [call[0][0] for call in mock_cur.execute.call_args_list]
        assert len(executed) == 2
        for sql in executed:
            assert "ticker = %s" in sql

    def test_retrieve_without_ticker_filter(self) -> None:
        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = []

        chunks, _, _ = retriever.retrieve("General question", top_k=3)

        assert chunks == []
        for call in mock_cur.execute.call_args_list:
            assert "ticker = %s" not in call[0][0]

    def test_retrieve_with_date_filters(self) -> None:
        retriever, mock_cur = _make_retriever(_unit(0))
        mock_cur.fetchall.return_value = []

        retriever.retrieve(
            "Revenue trends",
            top_k=3,
            filing_date_from="2023-01-01",
            filing_date_to="2024-12-31",
        )

        for call in mock_cur.execute.call_args_list:
            sql = call[0][0]
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
