"""Unit tests for the EDGAR client, config, Kafka producer, and FastAPI app.

All HTTP calls are mocked — no real network traffic.
Covers: ticker→CIK resolution, submissions listing, document fetch,
retry/backoff, skip-on-failure (FR-5), Filing dataclass, config loading,
Kafka producer serialisation, and FastAPI endpoints.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.config import Settings, load_tickers
from src.edgar_client import EdgarClient, Filing
from src.kafka_producer import PublishOutcome

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

# ── Helpers ──────────────────────────────────────────────────────


def _make_async_response(
    *,
    json_data: dict[str, Any] | None = None,
    text_data: str | None = None,
    raise_on_status: Exception | None = None,
) -> tuple[MagicMock, Any]:
    """Create a mock that works as an ``async with session.get(...) as resp:`` target."""

    mock_resp = MagicMock()

    if raise_on_status:
        mock_resp.raise_for_status.side_effect = raise_on_status
    else:
        mock_resp.raise_for_status = MagicMock()

    if json_data is not None:

        async def _json(**kwargs: Any) -> dict[str, Any]:
            return json_data

        mock_resp.json = _json

    if text_data is not None:
        text_bytes = text_data.encode("utf-8")

        async def _iter_chunked(size: int):  # type: ignore[no-untyped-def]
            yield text_bytes

        mock_content = MagicMock()
        mock_content.iter_chunked = _iter_chunked
        mock_resp.content = mock_content
        mock_resp.content_length = len(text_bytes)

    @asynccontextmanager
    async def _ctx_manager(*args: Any, **kwargs: Any) -> AsyncGenerator[MagicMock, None]:
        yield mock_resp

    return mock_resp, _ctx_manager


_TICKER_MAP = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corporation"},
}


def _submissions(entries: list[tuple[str, str, str, str]]) -> dict[str, Any]:
    """Build a submissions JSON from (form, accession, filing_date, primary_doc) tuples."""
    return {
        "name": "Apple Inc.",
        "filings": {
            "recent": {
                "form": [e[0] for e in entries],
                "accessionNumber": [e[1] for e in entries],
                "filingDate": [e[2] for e in entries],
                "primaryDocument": [e[3] for e in entries],
            }
        },
    }


@pytest.fixture
def client() -> EdgarClient:
    return EdgarClient(user_agent="TestAgent test@example.com", rate_limit_rps=100, max_retries=1)


# ── EdgarClient.resolve_cik ──────────────────────────────────────


class TestResolveCik:
    """Tests for ticker → CIK resolution via company_tickers.json."""

    @pytest.mark.asyncio
    async def test_resolves_known_ticker(self, client: EdgarClient) -> None:
        _, ctx = _make_async_response(json_data=_TICKER_MAP)
        mock_session = MagicMock()
        mock_session.get = ctx

        resolved = await client.resolve_cik("AAPL", mock_session)
        assert resolved == (320193, "Apple Inc.")

    @pytest.mark.asyncio
    async def test_resolution_is_case_insensitive(self, client: EdgarClient) -> None:
        _, ctx = _make_async_response(json_data=_TICKER_MAP)
        mock_session = MagicMock()
        mock_session.get = ctx

        resolved = await client.resolve_cik("aapl", mock_session)
        assert resolved == (320193, "Apple Inc.")

    @pytest.mark.asyncio
    async def test_unknown_ticker_returns_none(self, client: EdgarClient) -> None:
        _, ctx = _make_async_response(json_data=_TICKER_MAP)
        mock_session = MagicMock()
        mock_session.get = ctx

        assert await client.resolve_cik("ZZZZ", mock_session) is None

    @pytest.mark.asyncio
    async def test_mapping_is_cached_after_first_call(self, client: EdgarClient) -> None:
        call_count = 0

        @asynccontextmanager
        async def _counting_ctx(*a: Any, **kw: Any) -> AsyncGenerator[MagicMock, None]:
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()

            async def _json(**kwargs: Any) -> dict[str, Any]:
                return _TICKER_MAP

            mock_resp.json = _json
            yield mock_resp

        mock_session = MagicMock()
        mock_session.get = _counting_ctx

        await client.resolve_cik("AAPL", mock_session)
        await client.resolve_cik("MSFT", mock_session)
        assert call_count == 1  # second lookup served from cache

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_none(self, client: EdgarClient) -> None:
        import aiohttp

        _, ctx = _make_async_response(raise_on_status=aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=500, message="Server Error"
        ))
        mock_session = MagicMock()
        mock_session.get = ctx

        assert await client.resolve_cik("AAPL", mock_session) is None


# ── EdgarClient.list_10k_filings ─────────────────────────────────


class TestList10kFilings:
    """Tests for filing listing via the submissions API."""

    @pytest.mark.asyncio
    async def test_returns_only_10k_forms(self, client: EdgarClient) -> None:
        data = _submissions([
            ("10-K", "0001-24-000001", "2024-11-01", "aapl-2024.htm"),
            ("10-Q", "0001-24-000002", "2024-08-01", "aapl-q3.htm"),
            ("8-K", "0001-24-000003", "2024-07-01", "aapl-8k.htm"),
            ("10-K/A", "0001-24-000004", "2024-12-01", "aapl-2024a.htm"),
        ])
        _, ctx = _make_async_response(json_data=data)
        mock_session = MagicMock()
        mock_session.get = ctx

        filings = await client.list_10k_filings(320193, mock_session)
        assert len(filings) == 1
        assert filings[0]["accession_number"] == "0001-24-000001"
        assert filings[0]["primary_document"] == "aapl-2024.htm"

    @pytest.mark.asyncio
    async def test_respects_filings_since_cutoff(self, client: EdgarClient) -> None:
        data = _submissions([
            ("10-K", "0001-24-000001", "2024-11-01", "new.htm"),
            ("10-K", "0001-19-000001", "2019-10-30", "old.htm"),
        ])
        _, ctx = _make_async_response(json_data=data)
        mock_session = MagicMock()
        mock_session.get = ctx

        filings = await client.list_10k_filings(320193, mock_session)
        assert [f["accession_number"] for f in filings] == ["0001-24-000001"]

    @pytest.mark.asyncio
    async def test_caps_at_max_filings_per_ticker(self) -> None:
        client = EdgarClient(
            user_agent="Test", rate_limit_rps=100, max_retries=1, max_filings_per_ticker=2
        )
        data = _submissions([
            ("10-K", f"0001-2{i}-000001", f"202{4 - i}-11-01", f"doc{i}.htm")
            for i in range(4)
        ])
        _, ctx = _make_async_response(json_data=data)
        mock_session = MagicMock()
        mock_session.get = ctx

        filings = await client.list_10k_filings(320193, mock_session)
        assert len(filings) == 2

    @pytest.mark.asyncio
    async def test_skips_entries_missing_primary_document(self, client: EdgarClient) -> None:
        data = _submissions([
            ("10-K", "0001-24-000001", "2024-11-01", ""),
            ("10-K", "0001-23-000001", "2023-11-01", "aapl-2023.htm"),
        ])
        _, ctx = _make_async_response(json_data=data)
        mock_session = MagicMock()
        mock_session.get = ctx

        filings = await client.list_10k_filings(320193, mock_session)
        assert [f["accession_number"] for f in filings] == ["0001-23-000001"]

    @pytest.mark.asyncio
    async def test_fetch_failure_returns_empty(self, client: EdgarClient) -> None:
        import aiohttp

        _, ctx = _make_async_response(raise_on_status=aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=503, message="Unavailable"
        ))
        mock_session = MagicMock()
        mock_session.get = ctx

        assert await client.list_10k_filings(320193, mock_session) == []

    @pytest.mark.asyncio
    async def test_empty_submissions_returns_empty(self, client: EdgarClient) -> None:
        _, ctx = _make_async_response(json_data={})
        mock_session = MagicMock()
        mock_session.get = ctx

        assert await client.list_10k_filings(320193, mock_session) == []


# ── EdgarClient retry / backoff (JSON fetches) ───────────────────


class TestGetJsonRetry:
    """Verify exponential backoff in _get_json (max_retries=3)."""

    @pytest.mark.asyncio
    async def test_retries_on_client_error_then_succeeds(self) -> None:
        import aiohttp

        client = EdgarClient(user_agent="Test", rate_limit_rps=100, max_retries=3)
        error = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=503, message="Unavailable"
        )

        call_count = 0

        @asynccontextmanager
        async def _flaky_ctx(*a: Any, **kw: Any):  # type: ignore[misc]
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            if call_count < 3:
                mock_resp.raise_for_status.side_effect = error
            else:
                mock_resp.raise_for_status = MagicMock()

                async def _json(**kwargs: Any) -> dict[str, Any]:
                    return _TICKER_MAP

                mock_resp.json = _json
            yield mock_resp

        mock_session = MagicMock()
        mock_session.get = _flaky_ctx

        with patch("src.edgar_client.asyncio.sleep") as mock_sleep:
            resolved = await client.resolve_cik("AAPL", mock_session)

        assert resolved == (320193, "Apple Inc.")
        assert call_count == 3
        # Two backoff sleeps (2 s, 4 s) plus one sub-second rate-limit sleep on success.
        backoff_args = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] >= 1]
        assert backoff_args == [2, 4]

    @pytest.mark.asyncio
    async def test_returns_failure_value_after_all_retries_exhausted(self) -> None:
        import aiohttp

        client = EdgarClient(user_agent="Test", rate_limit_rps=100, max_retries=3)
        error = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=503, message="Unavailable"
        )

        @asynccontextmanager
        async def _always_fail(*a: Any, **kw: Any):  # type: ignore[misc]
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = error
            yield mock_resp

        mock_session = MagicMock()
        mock_session.get = _always_fail

        with patch("src.edgar_client.asyncio.sleep") as mock_sleep:
            resolved = await client.resolve_cik("AAPL", mock_session)

        assert resolved is None
        # 3 attempts → 2 sleeps (no sleep after the final failure)
        assert mock_sleep.call_args_list == [call(2), call(4)]

    @pytest.mark.asyncio
    async def test_unexpected_exception_fails_immediately(self) -> None:
        client = EdgarClient(user_agent="Test", rate_limit_rps=100, max_retries=3)

        @asynccontextmanager
        async def _raise_ctx(*a: Any, **kw: Any) -> AsyncGenerator[None, None]:
            raise ValueError("unexpected")
            yield  # noqa: RET504  # type: ignore[misc]

        mock_session = MagicMock()
        mock_session.get = _raise_ctx

        with patch("src.edgar_client.asyncio.sleep") as mock_sleep:
            resolved = await client.resolve_cik("AAPL", mock_session)

        assert resolved is None
        mock_sleep.assert_not_called()


# ── EdgarClient.fetch_filing_document ────────────────────────────


class TestFetchFilingDocument:
    """Tests for EdgarClient.fetch_filing_document."""

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self, client: EdgarClient) -> None:
        """If fetch fails, return None instead of raising (FR-5)."""
        import aiohttp

        _, ctx = _make_async_response(raise_on_status=aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=404, message="Not Found"
        ))
        mock_session = MagicMock()
        mock_session.get = ctx

        result = await client.fetch_filing_document("https://example.com/filing", mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_text_on_success(self, client: EdgarClient) -> None:
        _, ctx = _make_async_response(text_data="Item 1. Business...")
        mock_session = MagicMock()
        mock_session.get = ctx

        result = await client.fetch_filing_document("https://example.com/filing", mock_session)
        assert result == "Item 1. Business..."

    @pytest.mark.asyncio
    async def test_returns_none_on_unexpected_exception(self, client: EdgarClient) -> None:
        @asynccontextmanager
        async def _raise_ctx(*a: Any, **kw: Any) -> AsyncGenerator[None, None]:
            raise RuntimeError("boom")
            yield  # noqa: RET504  # type: ignore[misc]

        mock_session = MagicMock()
        mock_session.get = _raise_ctx

        result = await client.fetch_filing_document("https://example.com/f", mock_session)
        assert result is None

    @pytest.mark.asyncio
    async def test_retries_on_client_error_then_succeeds(self) -> None:
        import aiohttp

        client = EdgarClient(user_agent="Test", rate_limit_rps=100, max_retries=3)
        error = aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=503, message="Unavailable"
        )
        text_bytes = b"Item 1. Business..."

        call_count = 0

        @asynccontextmanager
        async def _flaky_ctx(*a: Any, **kw: Any):  # type: ignore[misc]
            nonlocal call_count
            call_count += 1
            mock_resp = MagicMock()
            if call_count < 3:
                mock_resp.raise_for_status.side_effect = error
            else:
                mock_resp.raise_for_status = MagicMock()
                mock_resp.content_length = len(text_bytes)

                async def _iter_chunked(size: int):  # type: ignore[no-untyped-def]
                    yield text_bytes

                mock_content = MagicMock()
                mock_content.iter_chunked = _iter_chunked
                mock_resp.content = mock_content
            yield mock_resp

        mock_session = MagicMock()
        mock_session.get = _flaky_ctx

        with patch("src.edgar_client.asyncio.sleep") as mock_sleep:
            result = await client.fetch_filing_document("https://example.com/f", mock_session)

        assert result == "Item 1. Business..."
        assert call_count == 3
        backoff_args = [c.args[0] for c in mock_sleep.call_args_list if c.args[0] >= 1]
        assert backoff_args == [2, 4]


# ── EdgarClient.fetch_company_facts ──────────────────────────────


class TestFetchCompanyFacts:
    @pytest.mark.asyncio
    async def test_returns_parsed_json(self, client: EdgarClient) -> None:
        doc = {"cik": 320193, "facts": {"us-gaap": {}}}
        _, ctx = _make_async_response(json_data=doc)
        mock_session = MagicMock()
        mock_session.get = ctx

        assert await client.fetch_company_facts(320193, mock_session) == doc

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self, client: EdgarClient) -> None:
        import aiohttp

        _, ctx = _make_async_response(raise_on_status=aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=404, message="Not Found"
        ))
        mock_session = MagicMock()
        mock_session.get = ctx

        assert await client.fetch_company_facts(320193, mock_session) is None


# ── EdgarClient.get_filings_for_ticker ───────────────────────────


class TestGetFilingsForTicker:
    """Tests for the end-to-end per-ticker method."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_ticker_unknown(self, client: EdgarClient) -> None:
        with patch.object(client, "resolve_cik", return_value=None):
            result = await client.get_filings_for_ticker("ZZZZ", "Unknown", MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_filings(self, client: EdgarClient) -> None:
        with patch.object(client, "resolve_cik", return_value=(320193, "Apple Inc.")), \
             patch.object(client, "list_10k_filings", return_value=[]):
            result = await client.get_filings_for_ticker("AAPL", "Apple Inc.", MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_when_fetch_returns_none(self, client: EdgarClient) -> None:
        with patch.object(client, "resolve_cik", return_value=(320193, "Apple Inc.")), \
             patch.object(client, "list_10k_filings", return_value=[
                 {"accession_number": "0001-24-000001", "filing_date": "2024-11-01",
                  "primary_document": "aapl-2024.htm"},
             ]), \
             patch.object(client, "fetch_filing_document", return_value=None):
            result = await client.get_filings_for_ticker("AAPL", "Apple Inc.", MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_when_document_parses_to_empty(self, client: EdgarClient) -> None:
        with patch.object(client, "resolve_cik", return_value=(320193, "Apple Inc.")), \
             patch.object(client, "list_10k_filings", return_value=[
                 {"accession_number": "0001-24-000001", "filing_date": "2024-11-01",
                  "primary_document": "aapl-2024.htm"},
             ]), \
             patch.object(client, "fetch_filing_document",
                          return_value="<html><body><script>x</script></body></html>"):
            result = await client.get_filings_for_ticker("AAPL", "Apple Inc.", MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_parsed_filing_on_success(self, client: EdgarClient) -> None:
        """Happy path: HTML document is fetched and converted to clean text."""
        html = "<html><body><div>Item 1. Business</div><p>We sell devices.</p></body></html>"
        with patch.object(client, "resolve_cik", return_value=(320193, "Apple Inc.")), \
             patch.object(client, "list_10k_filings", return_value=[
                 {"accession_number": "0001-24-000001", "filing_date": "2024-11-01",
                  "primary_document": "aapl-2024.htm"},
             ]), \
             patch.object(client, "fetch_filing_document", return_value=html):
            result = await client.get_filings_for_ticker("AAPL", "Apple Inc.", MagicMock())

        assert len(result) == 1
        filing = result[0]
        assert filing.ticker == "AAPL"
        assert filing.accession_number == "0001-24-000001"
        assert filing.company_name == "Apple Inc."
        assert filing.filing_type == "10-K"
        assert "Item 1. Business" in filing.raw_text
        assert "We sell devices." in filing.raw_text
        assert "<" not in filing.raw_text  # no HTML survives

    @pytest.mark.asyncio
    async def test_constructs_primary_document_url(self, client: EdgarClient) -> None:
        with patch.object(client, "resolve_cik", return_value=(320193, "Apple Inc.")), \
             patch.object(client, "list_10k_filings", return_value=[
                 {"accession_number": "0001-24-000001", "filing_date": "2024-11-01",
                  "primary_document": "aapl-2024.htm"},
             ]), \
             patch.object(client, "fetch_filing_document", return_value="Item 1. text") as mock_fetch:
            await client.get_filings_for_ticker("AAPL", "Apple Inc.", MagicMock())

        url = mock_fetch.call_args[0][0]
        assert url == (
            "https://www.sec.gov/Archives/edgar/data/320193"
            "/000124000001/aapl-2024.htm"
        )

    @pytest.mark.asyncio
    async def test_multiple_filings(self, client: EdgarClient) -> None:
        with patch.object(client, "resolve_cik", return_value=(320193, "Apple Inc.")), \
             patch.object(client, "list_10k_filings", return_value=[
                 {"accession_number": "0001-24-000001", "filing_date": "2024-11-01",
                  "primary_document": "a.htm"},
                 {"accession_number": "0001-23-000001", "filing_date": "2023-11-03",
                  "primary_document": "b.htm"},
             ]), \
             patch.object(client, "fetch_filing_document", return_value="Item 1. text"):
            result = await client.get_filings_for_ticker("AAPL", "Apple Inc.", MagicMock())
        assert [f.accession_number for f in result] == ["0001-24-000001", "0001-23-000001"]


# ── Filing dataclass ─────────────────────────────────────────────


class TestFiling:
    """Tests for the Filing dataclass."""

    def test_stores_all_fields(self) -> None:
        f = Filing(
            accession_number="0001-24-000001",
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_date="2024-11-01",
            filing_type="10-K",
            source_url="https://sec.gov/...",
            raw_text="Item 1. Business...",
        )
        assert f.ticker == "AAPL"
        assert f.filing_type == "10-K"
        assert f.accession_number == "0001-24-000001"
        assert len(f.raw_text) > 0


# ── Config ───────────────────────────────────────────────────────


class TestConfig:
    """Tests for config.py."""

    def test_settings_defaults(self) -> None:
        s = Settings()
        assert s.postgres_db == "findocdrag"
        assert s.kafka_bootstrap_servers == "kafka:9092"
        assert s.edgar_rate_limit_rps == 10

    def test_postgres_dsn(self) -> None:
        s = Settings()
        assert s.postgres_dsn.startswith("postgresql://")
        assert "findocdrag" in s.postgres_dsn

    def test_settings_all_defaults(self) -> None:
        s = Settings()
        assert s.postgres_host == "postgres"
        assert s.postgres_port == 5432
        assert s.postgres_user == "findocdrag"
        assert s.postgres_password == "changeme"
        assert s.edgar_user_agent == "FinDocDRAG findocdrag@example.com"
        assert s.log_level == "INFO"
        assert s.tickers_config_path == "config/tickers.yml"

    def test_load_tickers_missing_file(self) -> None:
        result = load_tickers("/nonexistent/path.yml")
        assert result == []

    def test_load_tickers_valid_file(self, tmp_path: Any) -> None:
        f = tmp_path / "tickers.yml"
        f.write_text('tickers:\n  - symbol: "AAPL"\n    name: "Apple Inc."\n')
        result = load_tickers(str(f))
        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"

    def test_load_tickers_empty_file(self, tmp_path: Any) -> None:
        f = tmp_path / "tickers.yml"
        f.write_text("{}")
        result = load_tickers(str(f))
        assert result == []


# ── Kafka Producer ───────────────────────────────────────────────


def _delivery_test_filing() -> Filing:
    return Filing(
        accession_number="0001-24-000001",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_date="2024-11-01",
        filing_type="10-K",
        source_url="https://sec.gov/...",
        raw_text="Item 1. Business...",
    )


def _make_acking_producer_mock(delivery_error: str | None = None) -> MagicMock:
    """Mock Producer whose flush() fires the pending delivery callback.

    Mirrors librdkafka: produce() only enqueues, and the callback registered
    with it runs later — during flush() — carrying the delivery result.
    """
    instance = MagicMock()
    pending: list[Any] = []

    def _produce(**kwargs: Any) -> None:
        pending.append(kwargs["callback"])

    def _flush(_timeout: float = 0.0) -> int:
        msg = MagicMock()
        msg.topic.return_value = "filings.raw"
        msg.partition.return_value = 0
        msg.offset.return_value = 1
        err = None
        if delivery_error is not None:
            err = MagicMock()
            err.__str__ = lambda _self: delivery_error  # type: ignore[assignment]
        for callback in pending:
            callback(err, msg)
        pending.clear()
        return 0  # nothing left queued

    instance.produce.side_effect = _produce
    instance.flush.side_effect = _flush
    return instance


class TestKafkaProducer:
    """Tests for kafka_producer.py (serialisation logic)."""

    def test_publish_filing_calls_produce(self) -> None:
        from src.kafka_producer import FilingProducer

        with patch("src.kafka_producer.Producer") as mock_producer:
            mock_instance = MagicMock()
            mock_producer.return_value = mock_instance

            producer = FilingProducer(bootstrap_servers="localhost:9092")

            filing = Filing(
                accession_number="0001-24-000001",
                ticker="AAPL",
                company_name="Apple Inc.",
                filing_date="2024-11-01",
                filing_type="10-K",
                source_url="https://sec.gov/...",
                raw_text="Item 1. Business...",
            )
            producer.publish_filing(filing)

            mock_instance.produce.assert_called_once()
            call_kwargs = mock_instance.produce.call_args
            assert call_kwargs.kwargs["topic"] == "filings.raw"
            assert call_kwargs.kwargs["key"] == "0001-24-000001"

            # Verify the serialised message has all required fields
            msg = json.loads(call_kwargs.kwargs["value"])
            assert msg["ticker"] == "AAPL"
            assert msg["filing_type"] == "10-K"
            assert "published_at" in msg
            assert msg["company_name"] == "Apple Inc."
            assert msg["raw_text"] == "Item 1. Business..."
            assert msg["source_url"] == "https://sec.gov/..."
            assert msg["accession_number"] == "0001-24-000001"

    def test_publish_filing_returns_delivered_once_broker_acks(self) -> None:
        """A clean flush with no callback error means the broker took the message."""
        from src.kafka_producer import FilingProducer

        with patch("src.kafka_producer.Producer") as mock_producer:
            mock_instance = _make_acking_producer_mock()
            mock_producer.return_value = mock_instance

            producer = FilingProducer(bootstrap_servers="localhost:9092")
            outcome = producer.publish_filing(_delivery_test_filing())

            assert outcome is PublishOutcome.DELIVERED
            # The ack must be waited for, not left in the background buffer.
            mock_instance.flush.assert_called_once()

    def test_publish_filing_returns_failed_when_broker_rejects(self) -> None:
        """A delivery callback carrying an error must not be reported as success."""
        from src.kafka_producer import FilingProducer

        with patch("src.kafka_producer.Producer") as mock_producer:
            mock_instance = _make_acking_producer_mock(delivery_error="Broker unavailable")
            mock_producer.return_value = mock_instance

            producer = FilingProducer(bootstrap_servers="localhost:9092")
            outcome = producer.publish_filing(_delivery_test_filing())

            assert outcome is PublishOutcome.FAILED

    def test_publish_filing_returns_failed_when_flush_times_out(self) -> None:
        """Messages still queued after flush() are undelivered, not delivered."""
        from src.kafka_producer import FilingProducer

        with patch("src.kafka_producer.Producer") as mock_producer:
            mock_instance = MagicMock()
            mock_instance.flush.return_value = 1  # one message still queued
            mock_producer.return_value = mock_instance

            producer = FilingProducer(bootstrap_servers="localhost:9092")
            outcome = producer.publish_filing(_delivery_test_filing(), timeout=0.1)

            assert outcome is PublishOutcome.FAILED

    def test_publish_filing_returns_failed_when_produce_raises(self) -> None:
        """A full local queue surfaces as a failure rather than an exception."""
        from src.kafka_producer import FilingProducer

        with patch("src.kafka_producer.Producer") as mock_producer:
            mock_instance = MagicMock()
            mock_instance.produce.side_effect = BufferError("Local queue full")
            mock_producer.return_value = mock_instance

            producer = FilingProducer(bootstrap_servers="localhost:9092")
            outcome = producer.publish_filing(_delivery_test_filing())

            assert outcome is PublishOutcome.FAILED

    def test_publish_filing_returns_too_large_without_producing(self) -> None:
        """Oversize filings are rejected before hitting the broker."""
        from src.kafka_producer import FilingProducer

        with patch("src.kafka_producer.Producer") as mock_producer:
            mock_instance = _make_acking_producer_mock()
            mock_producer.return_value = mock_instance

            producer = FilingProducer(bootstrap_servers="localhost:9092")
            producer.MAX_RAW_BYTES = 10  # smaller than any serialised filing

            outcome = producer.publish_filing(_delivery_test_filing())

            assert outcome is PublishOutcome.TOO_LARGE
            mock_instance.produce.assert_not_called()

    def test_flush_delegates_to_producer(self) -> None:
        from src.kafka_producer import FilingProducer

        with patch("src.kafka_producer.Producer") as mock_producer:
            mock_instance = MagicMock()
            mock_instance.flush.return_value = 0
            mock_producer.return_value = mock_instance

            producer = FilingProducer(bootstrap_servers="localhost:9092")
            remaining = producer.flush()

            assert remaining == 0
            mock_instance.flush.assert_called_once_with(10.0)

    def test_delivery_callback_on_error(self) -> None:
        """Test _delivery_callback logs error on failure."""
        from src.kafka_producer import _delivery_callback

        mock_msg = MagicMock()
        mock_msg.topic.return_value = "filings.raw"
        mock_err = MagicMock()
        mock_err.__str__ = lambda self: "Broker unavailable"

        # Should not raise
        _delivery_callback(mock_err, mock_msg)

    def test_delivery_callback_on_success(self) -> None:
        """Test _delivery_callback logs success on delivery."""
        from src.kafka_producer import _delivery_callback

        mock_msg = MagicMock()
        mock_msg.topic.return_value = "filings.raw"
        mock_msg.partition.return_value = 0
        mock_msg.offset.return_value = 42

        # Should not raise
        _delivery_callback(None, mock_msg)


# ── FastAPI App (main.py) ────────────────────────────────────────


def _make_edgar_mock(*, filings: list[Filing] | None = None) -> MagicMock:
    """Edgar client mock whose facts flow no-ops (resolve_cik → None)."""
    mock_edgar = MagicMock()
    mock_edgar.get_filings_for_ticker = AsyncMock(return_value=filings or [])
    mock_edgar.resolve_cik = AsyncMock(return_value=None)
    mock_edgar.fetch_company_facts = AsyncMock(return_value=None)
    return mock_edgar


class TestFastAPIApp:
    """Tests for the FastAPI endpoints in main.py.

    Uses FastAPI TestClient to test endpoints without a running server.
    """

    def test_health_endpoint(self) -> None:
        """GET /health returns healthy when DB is reachable."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_db._get_conn.return_value = mock_conn

        original_db = main_module._db
        main_module._db = mock_db

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.get("/health")
            assert response.status_code == 200
            assert response.json() == {"status": "healthy"}
            assert "X-Request-ID" in response.headers
        finally:
            main_module._db = original_db

    def test_health_endpoint_not_initialized(self) -> None:
        """GET /health returns 503 when not initialized."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        original_db = main_module._db
        main_module._db = None

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.get("/health")
            assert response.status_code == 503
        finally:
            main_module._db = original_db

    def test_ready_endpoint_not_initialized(self) -> None:
        """GET /ready returns 503 when service is not initialized."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        # Ensure globals are None
        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        main_module._edgar_client = None
        main_module._kafka_producer = None

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.get("/ready")
            assert response.status_code == 503
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka

    def test_ready_endpoint_initialized(self) -> None:
        """GET /ready returns ready when all dependencies are set."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        original_db = main_module._db
        main_module._edgar_client = MagicMock()
        main_module._kafka_producer = MagicMock()
        main_module._db = MagicMock()

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.get("/ready")
            assert response.status_code == 200
            assert response.json() == {"status": "ready"}
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka
            main_module._db = original_db

    def test_ingest_not_initialized(self) -> None:
        """POST /v1/ingest returns 503 when not initialized."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        main_module._edgar_client = None
        main_module._kafka_producer = None

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.post("/v1/ingest", json={"tickers": ["AAPL"]})
            assert response.status_code == 503
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka

    def test_ingest_no_tickers_and_no_config(self) -> None:
        """POST /v1/ingest with no tickers and no config file returns 400."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        original_db = main_module._db
        main_module._edgar_client = _make_edgar_mock()
        main_module._kafka_producer = MagicMock()
        main_module._db = MagicMock()

        try:
            with patch("src.main.load_tickers", return_value=[]):
                client = TestClient(app=main_module.app, raise_server_exceptions=False)
                response = client.post("/v1/ingest", json={})
                assert response.status_code == 400
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka
            main_module._db = original_db

    def test_ingest_success_with_tickers(self) -> None:
        """POST /v1/ingest publishes new filings and records them in ingestion_log."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        filing = Filing(
            accession_number="0001-24-000001",
            ticker="AAPL",
            company_name="AAPL",
            filing_date="2024-11-01",
            filing_type="10-K",
            source_url="https://sec.gov/...",
            raw_text="Item 1...",
        )

        mock_edgar = _make_edgar_mock(filings=[filing])
        mock_kafka = MagicMock()
        mock_db = MagicMock()

        mock_kafka.publish_filing = MagicMock(return_value=PublishOutcome.DELIVERED)
        mock_kafka.flush = MagicMock()
        mock_db.is_already_ingested.return_value = False

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        original_db = main_module._db
        main_module._edgar_client = mock_edgar
        main_module._kafka_producer = mock_kafka
        main_module._db = mock_db

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.post("/v1/ingest", json={"tickers": ["AAPL"]})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["tickers_processed"] == ["AAPL"]
            assert data["filings_published"] == 1
            assert data["filings_skipped"] == 0
            assert data["filings_failed"] == 0
            assert data["facts_stored"] == 0
            assert data["errors"] == []
            mock_kafka.publish_filing.assert_called_once_with(filing)
            mock_db.record_ingestion.assert_called_once_with(filing)
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka
            main_module._db = original_db

    def test_ingest_stores_company_facts(self) -> None:
        """POST /v1/ingest fetches XBRL companyfacts and stores annual facts."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        company_facts = {
            "facts": {
                "us-gaap": {
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "start": "2023-10-01",
                                    "end": "2024-09-28",
                                    "val": 93_736_000_000,
                                    "form": "10-K",
                                    "fp": "FY",
                                    "fy": 2024,
                                    "filed": "2024-11-01",
                                }
                            ]
                        }
                    }
                }
            }
        }

        mock_edgar = MagicMock()
        mock_edgar.get_filings_for_ticker = AsyncMock(return_value=[])
        mock_edgar.resolve_cik = AsyncMock(return_value=(320193, "Apple Inc."))
        mock_edgar.fetch_company_facts = AsyncMock(return_value=company_facts)

        mock_kafka = MagicMock()
        mock_kafka.flush = MagicMock()
        mock_db = MagicMock()
        mock_db.store_financial_facts.return_value = 1

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        original_db = main_module._db
        main_module._edgar_client = mock_edgar
        main_module._kafka_producer = mock_kafka
        main_module._db = mock_db

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.post("/v1/ingest", json={"tickers": ["AAPL"]})
            assert response.status_code == 200
            data = response.json()
            assert data["facts_stored"] == 1
            stored_facts = mock_db.store_financial_facts.call_args[0][0]
            assert len(stored_facts) == 1
            assert stored_facts[0].concept == "NetIncomeLoss"
            assert stored_facts[0].ticker == "AAPL"
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka
            main_module._db = original_db

    def test_ingest_handles_ticker_error(self) -> None:
        """POST /v1/ingest captures per-ticker errors without crashing."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        mock_edgar = _make_edgar_mock()
        mock_edgar.get_filings_for_ticker = AsyncMock(side_effect=RuntimeError("EDGAR down"))
        mock_kafka = MagicMock()
        mock_kafka.flush = MagicMock()
        mock_db = MagicMock()

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        original_db = main_module._db
        main_module._edgar_client = mock_edgar
        main_module._kafka_producer = mock_kafka
        main_module._db = mock_db

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.post("/v1/ingest", json={"tickers": ["AAPL"]})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert len(data["errors"]) == 1
            assert "AAPL" in data["errors"][0]
            assert data["filings_published"] == 0
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka
            main_module._db = original_db

    def test_ingest_with_config_file_tickers(self) -> None:
        """POST /v1/ingest falls back to config/tickers.yml when no tickers in body."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        mock_edgar = _make_edgar_mock()
        mock_kafka = MagicMock()
        mock_kafka.flush = MagicMock()
        mock_db = MagicMock()

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        original_db = main_module._db
        main_module._edgar_client = mock_edgar
        main_module._kafka_producer = mock_kafka
        main_module._db = mock_db

        try:
            with patch("src.main.load_tickers", return_value=[
                {"symbol": "MSFT", "name": "Microsoft Corp."},
            ]):
                client = TestClient(app=main_module.app, raise_server_exceptions=False)
                response = client.post("/v1/ingest")
                assert response.status_code == 200
                data = response.json()
                assert data["tickers_processed"] == ["MSFT"]
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka
            main_module._db = original_db

    def test_ingest_skips_already_ingested_filing(self) -> None:
        """POST /v1/ingest skips filings already present in ingestion_log (FR-4)."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        filing = Filing(
            accession_number="0001-24-000001",
            ticker="AAPL",
            company_name="AAPL",
            filing_date="2024-11-01",
            filing_type="10-K",
            source_url="https://sec.gov/...",
            raw_text="Item 1...",
        )

        mock_edgar = _make_edgar_mock(filings=[filing])
        mock_kafka = MagicMock()
        mock_kafka.flush = MagicMock()
        mock_db = MagicMock()
        # Simulate filing already present in ingestion_log
        mock_db.is_already_ingested.return_value = True

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        original_db = main_module._db
        main_module._edgar_client = mock_edgar
        main_module._kafka_producer = mock_kafka
        main_module._db = mock_db

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.post("/v1/ingest", json={"tickers": ["AAPL"]})
            assert response.status_code == 200
            data = response.json()
            assert data["filings_published"] == 0
            assert data["filings_skipped"] == 1
            # Kafka and DB record must NOT be called for duplicate filings
            mock_kafka.publish_filing.assert_not_called()
            mock_db.record_ingestion.assert_not_called()
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka
            main_module._db = original_db

    def test_ingest_does_not_record_filing_kafka_never_acked(self) -> None:
        """A filing Kafka never acknowledged must stay eligible for re-ingestion.

        Recording it in ingestion_log would make is_already_ingested() skip it
        on every future run, losing the filing permanently even though no
        chunks were ever produced for it.
        """
        from fastapi.testclient import TestClient

        import src.main as main_module

        filing = _delivery_test_filing()

        mock_edgar = _make_edgar_mock(filings=[filing])
        mock_kafka = MagicMock()
        mock_kafka.publish_filing = MagicMock(return_value=PublishOutcome.FAILED)
        mock_kafka.flush = MagicMock()
        mock_db = MagicMock()
        mock_db.is_already_ingested.return_value = False

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        original_db = main_module._db
        main_module._edgar_client = mock_edgar
        main_module._kafka_producer = mock_kafka
        main_module._db = mock_db

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.post("/v1/ingest", json={"tickers": ["AAPL"]})
            assert response.status_code == 200
            data = response.json()

            # The filing counts as failed, never as published or merely skipped.
            assert data["filings_failed"] == 1
            assert data["filings_published"] == 0
            assert data["filings_skipped"] == 0
            # The critical assertion: nothing was written to ingestion_log.
            mock_db.record_ingestion.assert_not_called()
            # And the failure is visible to the caller, not swallowed.
            assert len(data["errors"]) == 1
            assert filing.accession_number in data["errors"][0]
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka
            main_module._db = original_db

    def test_ingest_records_filing_only_after_delivery_ack(self) -> None:
        """ingestion_log is written only once the broker has confirmed delivery."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        filing = _delivery_test_filing()
        call_order: list[str] = []

        mock_edgar = _make_edgar_mock(filings=[filing])
        mock_kafka = MagicMock()
        mock_db = MagicMock()
        mock_db.is_already_ingested.return_value = False

        def _publish(_filing: Filing) -> PublishOutcome:
            call_order.append("publish")
            return PublishOutcome.DELIVERED

        mock_kafka.publish_filing = MagicMock(side_effect=_publish)
        mock_kafka.flush = MagicMock()
        mock_db.record_ingestion = MagicMock(
            side_effect=lambda _f: call_order.append("record")
        )

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        original_db = main_module._db
        main_module._edgar_client = mock_edgar
        main_module._kafka_producer = mock_kafka
        main_module._db = mock_db

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.post("/v1/ingest", json={"tickers": ["AAPL"]})
            assert response.status_code == 200
            assert response.json()["filings_published"] == 1
            assert call_order == ["publish", "record"]
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka
            main_module._db = original_db

    def test_ingest_mixed_new_and_duplicate_filings(self) -> None:
        """POST /v1/ingest publishes new filings and skips duplicates in the same run."""
        from fastapi.testclient import TestClient

        import src.main as main_module

        new_filing = Filing(
            accession_number="0001-24-000001",
            ticker="AAPL",
            company_name="AAPL",
            filing_date="2024-11-01",
            filing_type="10-K",
            source_url="https://sec.gov/1",
            raw_text="Item 1...",
        )
        dup_filing = Filing(
            accession_number="0001-23-000001",
            ticker="AAPL",
            company_name="AAPL",
            filing_date="2023-11-01",
            filing_type="10-K",
            source_url="https://sec.gov/2",
            raw_text="Item 1 old...",
        )

        mock_edgar = _make_edgar_mock(filings=[new_filing, dup_filing])
        mock_kafka = MagicMock()
        mock_kafka.publish_filing = MagicMock(return_value=PublishOutcome.DELIVERED)
        mock_kafka.flush = MagicMock()
        mock_db = MagicMock()
        # new_filing is new, dup_filing is already ingested
        mock_db.is_already_ingested.side_effect = lambda acc: acc == dup_filing.accession_number

        original_edgar = main_module._edgar_client
        original_kafka = main_module._kafka_producer
        original_db = main_module._db
        main_module._edgar_client = mock_edgar
        main_module._kafka_producer = mock_kafka
        main_module._db = mock_db

        try:
            client = TestClient(app=main_module.app, raise_server_exceptions=False)
            response = client.post("/v1/ingest", json={"tickers": ["AAPL"]})
            assert response.status_code == 200
            data = response.json()
            assert data["filings_published"] == 1
            assert data["filings_skipped"] == 1
            mock_kafka.publish_filing.assert_called_once_with(new_filing)
            mock_db.record_ingestion.assert_called_once_with(new_filing)
        finally:
            main_module._edgar_client = original_edgar
            main_module._kafka_producer = original_kafka
            main_module._db = original_db
