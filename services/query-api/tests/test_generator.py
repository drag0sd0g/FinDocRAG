"""Unit tests for the RAG generator, prompt builder, auth, and models.

All external dependencies (retriever, LLM, DB) are mocked.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.auth import verify_api_key
from src.llm.backend import LLMResponse
from src.models import QueryRequest, QueryResponse, SourceChunk
from src.rag.generator import RAGGenerator
from src.rag.prompts import RetrievedChunk, build_prompt

# ── Prompt builder ───────────────────────────────────────────────

class TestBuildPrompt:
    def test_includes_system_instruction(self) -> None:
        prompt = build_prompt("What is revenue?", [])
        assert "financial document analyst" in prompt

    def test_includes_question(self) -> None:
        prompt = build_prompt("What is revenue?", [])
        assert "What is revenue?" in prompt

    def test_includes_numbered_sources(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id="a", ticker="AAPL", filing_date="2024-11-01",
                section="Item 7", relevance_score=0.9, text="Revenue was $391B."
            ),
            RetrievedChunk(
                chunk_id="b", ticker="AAPL", filing_date="2024-11-01",
                section="Item 1A", relevance_score=0.8, text="Risk factors include..."
            ),
        ]
        prompt = build_prompt("Tell me about revenue", chunks)
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt
        assert "Revenue was $391B." in prompt
        assert "(AAPL, 2024-11-01, Item 7, relevance: 0.90)" in prompt


# ── RAGGenerator ─────────────────────────────────────────────────

class TestRAGGenerator:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        """Full pipeline: retrieve → prompt → LLM → response."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = (
            [
                RetrievedChunk(
                    chunk_id="chunk1", ticker="AAPL", filing_date="2024-11-01",
                    section="Item 1A", relevance_score=0.87, text="Supply chain risks...",
                ),
            ],
            [0.1] * 768,
            12.0,  # embedding_ms
        )

        mock_llm = AsyncMock()
        mock_llm.model_name = "test-model"
        mock_llm.generate.return_value = LLMResponse(
            text="Apple faces supply chain risks [Source 1].",
            model="test-model",
            prompt_tokens=100,
            completion_tokens=20,
        )

        gen = RAGGenerator(retriever=mock_retriever, llm=mock_llm)
        result = await gen.answer("What are Apple's risk factors?", top_k=5)

        assert isinstance(result, QueryResponse)
        assert result.answer is not None
        assert "supply chain" in result.answer.lower()
        assert len(result.sources) == 1
        assert result.sources[0].chunk_id == "chunk1"
        assert result.degraded is False
        assert result.timing.embedding_ms == 12.0

    @pytest.mark.asyncio
    async def test_degraded_mode_on_llm_failure(self) -> None:
        """If LLM fails, return sources without answer (TDD Section 7.4)."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = (
            [
                RetrievedChunk(
                    chunk_id="chunk1", ticker="AAPL", filing_date="2024-11-01",
                    section="Item 1A", relevance_score=0.87, text="Some text",
                ),
            ],
            [0.1] * 768,
            10.0,
        )

        mock_llm = AsyncMock()
        mock_llm.model_name = "test-model"
        mock_llm.generate.side_effect = Exception("LLM timeout")

        gen = RAGGenerator(retriever=mock_retriever, llm=mock_llm)
        result = await gen.answer("Question?")

        assert result.answer is None
        assert result.degraded is True
        assert len(result.sources) == 1

    @pytest.mark.asyncio
    async def test_no_chunks_returns_no_answer(self) -> None:
        """If no chunks retrieved, no prompt is sent to LLM."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = ([], [0.1] * 768, 8.0)

        mock_llm = AsyncMock()
        mock_llm.model_name = "test-model"

        gen = RAGGenerator(retriever=mock_retriever, llm=mock_llm)
        result = await gen.answer("Unknown question")

        assert result.answer is None
        assert result.sources == []
        mock_llm.generate.assert_not_called()


# ── Event loop must stay free during retrieval ───────────────────

class TestEventLoopIsNotBlocked:
    """Retrieval is synchronous CPU + IO work (a sentence-transformers forward
    pass and psycopg2 queries).  Awaiting it inline would pin the event loop
    for the whole query, so a single-worker uvicorn would serve exactly one
    request at a time regardless of how many arrive.
    """

    @staticmethod
    def _retriever_recording_thread(record: dict[str, int]) -> MagicMock:
        def _retrieve(**_kwargs: object) -> tuple:
            record["retrieve"] = threading.get_ident()
            return ([], [0.1] * 768, 1.0)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = _retrieve
        return mock_retriever

    @pytest.mark.asyncio
    async def test_retrieval_runs_off_the_event_loop_thread(self) -> None:
        record: dict[str, int] = {}
        mock_llm = AsyncMock()
        mock_llm.model_name = "test-model"

        gen = RAGGenerator(
            retriever=self._retriever_recording_thread(record), llm=mock_llm
        )
        await gen.answer("What are Apple's risks?")

        assert record["retrieve"] != threading.get_ident()

    @pytest.mark.asyncio
    async def test_fact_lookup_runs_off_the_event_loop_thread(self) -> None:
        """The XBRL lookup is another blocking psycopg2 round trip."""
        record: dict[str, int] = {}

        def _facts_for_question(*_args: object) -> list:
            record["facts"] = threading.get_ident()
            return []

        mock_facts = MagicMock()
        mock_facts.facts_for_question.side_effect = _facts_for_question

        mock_llm = AsyncMock()
        mock_llm.model_name = "test-model"

        gen = RAGGenerator(
            retriever=self._retriever_recording_thread({}),
            llm=mock_llm,
            facts=mock_facts,
        )
        await gen.answer("What was revenue in 2024?", ticker_filter="AAPL")

        assert record["facts"] != threading.get_ident()

    @pytest.mark.asyncio
    async def test_concurrent_queries_overlap(self) -> None:
        """Three slow retrievals must run concurrently, not back to back."""
        retrieve_seconds = 0.2
        concurrency = 3

        def _slow_retrieve(**_kwargs: object) -> tuple:
            time.sleep(retrieve_seconds)  # blocking on purpose
            return ([], [0.1] * 768, 1.0)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = _slow_retrieve
        mock_llm = AsyncMock()
        mock_llm.model_name = "test-model"

        gen = RAGGenerator(retriever=mock_retriever, llm=mock_llm)

        started = time.perf_counter()
        await asyncio.gather(*(gen.answer("question") for _ in range(concurrency)))
        elapsed = time.perf_counter() - started

        serial = retrieve_seconds * concurrency
        # Generous margin: the point is "clearly not serial", not a precise time.
        assert elapsed < serial * 0.7, (
            f"{concurrency} queries took {elapsed:.2f}s; "
            f"serial execution would be ~{serial:.2f}s"
        )

    @pytest.mark.asyncio
    async def test_other_coroutines_progress_during_retrieval(self) -> None:
        """A blocked loop would starve every unrelated task on it."""
        ticks = 0

        async def _ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        def _slow_retrieve(**_kwargs: object) -> tuple:
            time.sleep(0.2)
            return ([], [0.1] * 768, 1.0)

        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = _slow_retrieve
        mock_llm = AsyncMock()
        mock_llm.model_name = "test-model"

        gen = RAGGenerator(retriever=mock_retriever, llm=mock_llm)

        background = asyncio.create_task(_ticker())
        try:
            await gen.answer("question")
        finally:
            background.cancel()

        assert ticks > 0, "event loop made no progress while retrieval ran"


# ── Auth ─────────────────────────────────────────────────────────

class TestAuth:
    @pytest.mark.asyncio
    async def test_no_keys_configured_allows_all(self) -> None:
        """When API_KEYS is not set, all requests pass."""
        with patch.dict(os.environ, {"API_KEYS": ""}, clear=False):
            mock_request = MagicMock()
            mock_request.headers = {}
            # Should not raise
            await verify_api_key(mock_request)

    @pytest.mark.asyncio
    async def test_valid_key_passes(self) -> None:
        with patch.dict(os.environ, {"API_KEYS": "key1,key2"}, clear=False):
            mock_request = MagicMock()
            mock_request.headers = {"X-API-Key": "key1"}
            await verify_api_key(mock_request)

    @pytest.mark.asyncio
    async def test_invalid_key_raises_401(self) -> None:
        with patch.dict(os.environ, {"API_KEYS": "key1,key2"}, clear=False):
            mock_request = MagicMock()
            mock_request.headers = {"X-API-Key": "wrong"}
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(mock_request)
            assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_key_raises_401(self) -> None:
        with patch.dict(os.environ, {"API_KEYS": "key1"}, clear=False):
            mock_request = MagicMock()
            mock_request.headers = {}
            with pytest.raises(HTTPException) as exc_info:
                await verify_api_key(mock_request)
            assert exc_info.value.status_code == 401


# ── Models ───────────────────────────────────────────────────────

class TestModels:
    def test_query_request_defaults(self) -> None:
        req = QueryRequest(question="What is revenue?")
        assert req.top_k == 5
        assert req.ticker_filter is None

    def test_query_request_with_filter(self) -> None:
        req = QueryRequest(question="Revenue?", ticker_filter="AAPL", top_k=10)
        assert req.ticker_filter == "AAPL"
        assert req.top_k == 10

    def test_empty_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_whitespace_only_question_raises(self) -> None:
        with pytest.raises(ValidationError):
            QueryRequest(question="   ")

    def test_question_is_stripped(self) -> None:
        req = QueryRequest(question="  What is revenue?  ")
        assert req.question == "What is revenue?"

    def test_source_chunk_text_preview(self) -> None:
        s = SourceChunk(
            chunk_id="abc",
            ticker="MSFT",
            filing_date="2024-10-15",
            section="Item 7",
            relevance_score=0.92,
            text_preview="Revenue grew significantly...",
        )
        assert len(s.text_preview) <= 200
