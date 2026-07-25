"""SEC EDGAR client for fetching 10-K filings.

Resolution flow (one filing):
  1. Resolve ticker → CIK via the official company_tickers.json mapping,
     so we only ever ingest the company's own filings (never another
     filer that merely mentions the ticker in its text).
  2. List the company's 10-K filings via the submissions API
     (data.sec.gov/submissions/CIK##########.json).
  3. Fetch the filing's *primary document* (the actual 10-K HTML) rather
     than the full submission .txt, which bundles exhibits, XBRL and
     base64-encoded binaries.
  4. Parse the HTML to clean text (src.html_parser).

References:
  - TDD: FR-1 (fetch 10-K filings by ticker)
  - TDD: FR-5 (log and skip filings that fail to parse)
  - TDD: NFR-4 (respect SEC rate limit of 10 req/s)
  - TDD: Section 8.1.1 (EDGAR request duration histogram)
  - API docs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

import aiohttp
import structlog

from src.html_parser import extract_text
from src.metrics import EDGAR_REQUEST_DURATION, FILINGS_FETCHED_TOTAL

logger = structlog.get_logger()

# Official ticker → CIK mapping maintained by the SEC
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Filing history per company (10 years of data, parallel arrays)
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# XBRL structured facts per company
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Log a download-progress line every this many bytes
_PROGRESS_LOG_INTERVAL = 5 * 1024 * 1024  # 5 MB


@dataclass
class Filing:
    """Represents a single 10-K filing fetched from EDGAR."""

    accession_number: str
    ticker: str
    company_name: str
    filing_date: str
    filing_type: str
    source_url: str
    raw_text: str


@dataclass
class EdgarClient:
    """Async client for the SEC EDGAR submissions and archives APIs.

    Rate-limited via an asyncio.Semaphore to respect SEC's 10 req/s
    policy (TDD: NFR-4).  Retry logic uses exponential backoff with
    up to 3 attempts (TDD: Section 7.1).
    """

    user_agent: str
    rate_limit_rps: int = 10
    max_retries: int = 3
    filings_since: str = "2020-01-01"   # earliest filing date to ingest
    max_filings_per_ticker: int = 6     # newest first
    _semaphore: asyncio.Semaphore = field(init=False)
    _cik_cache: dict[str, tuple[int, str]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._semaphore = asyncio.Semaphore(self.rate_limit_rps)

    # ── Retry helper ─────────────────────────────────────────────

    async def _with_retries(
        self,
        operation: Callable[[], Awaitable[Any]],
        failure_value: Any,
        operation_name: str,
        **log_ctx: Any,
    ) -> Any:
        """Run an async operation with exponential-backoff retry.

        Retries on aiohttp.ClientError up to max_retries times (backoff: 2^attempt
        seconds: 2 s, 4 s, 8 s).  Any other exception causes an immediate return of
        failure_value.  Returns failure_value after all retries are exhausted.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return await operation()
            except aiohttp.ClientError as exc:
                if attempt == self.max_retries:
                    logger.error(f"{operation_name}_failed", **log_ctx, error=str(exc))
                    return failure_value
                wait = 2**attempt
                logger.warning(
                    f"{operation_name}_retry",
                    **log_ctx,
                    attempt=attempt,
                    wait_seconds=wait,
                    error=str(exc),
                )
                await asyncio.sleep(wait)
            except Exception as exc:
                logger.error(f"{operation_name}_failed", **log_ctx, error=str(exc))
                return failure_value
        return failure_value  # unreachable; keeps mypy happy

    # ── JSON fetch helper ────────────────────────────────────────

    async def _get_json(
        self,
        url: str,
        session: aiohttp.ClientSession,
        operation_name: str,
        **log_ctx: Any,
    ) -> dict[str, Any] | None:
        """GET a JSON document with rate limiting and retries."""
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}

        async def _do_get() -> dict[str, Any]:
            t0 = time.perf_counter()
            async with self._semaphore, session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                data: dict[str, Any] = await resp.json(content_type=None)
                await asyncio.sleep(1.0 / self.rate_limit_rps)
            logger.info(
                f"{operation_name}_complete",
                **log_ctx,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
            return data

        return await self._with_retries(  # type: ignore[no-any-return]
            _do_get, failure_value=None, operation_name=operation_name, **log_ctx
        )

    # ── Ticker → CIK resolution ──────────────────────────────────

    async def resolve_cik(
        self,
        ticker: str,
        session: aiohttp.ClientSession,
    ) -> tuple[int, str] | None:
        """Resolve a ticker symbol to (CIK, official company name).

        Uses the SEC's company_tickers.json mapping; the full mapping is
        cached in memory after the first call.  Returns None when the
        ticker is unknown or the mapping cannot be fetched.
        """
        symbol = ticker.upper()
        if not self._cik_cache:
            data = await self._get_json(
                COMPANY_TICKERS_URL, session, operation_name="edgar_ticker_map"
            )
            if data is None:
                return None
            for entry in data.values():
                self._cik_cache[str(entry["ticker"]).upper()] = (
                    int(entry["cik_str"]),
                    str(entry["title"]),
                )
            logger.info("edgar_ticker_map_loaded", entries=len(self._cik_cache))

        resolved = self._cik_cache.get(symbol)
        if resolved is None:
            logger.warning("edgar_unknown_ticker", ticker=symbol)
        return resolved

    # ── Filing metadata via the submissions API ──────────────────

    async def list_10k_filings(
        self,
        cik: int,
        session: aiohttp.ClientSession,
    ) -> list[dict[str, str]]:
        """List recent 10-K filings for a CIK via the submissions API.

        Returns dicts with accession_number, filing_date, primary_document
        (newest first), filtered to filings on/after ``filings_since`` and
        capped at ``max_filings_per_ticker``.  Amendments (10-K/A) are
        excluded — they would duplicate the original filing's content.
        """
        data = await self._get_json(
            SUBMISSIONS_URL.format(cik=cik),
            session,
            operation_name="edgar_submissions",
            cik=cik,
        )
        if data is None:
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms: list[str] = recent.get("form", [])
        accessions: list[str] = recent.get("accessionNumber", [])
        dates: list[str] = recent.get("filingDate", [])
        primary_docs: list[str] = recent.get("primaryDocument", [])

        filings: list[dict[str, str]] = []
        for form, accession, filing_date, primary_doc in zip(
            forms, accessions, dates, primary_docs, strict=False
        ):
            if form != "10-K" or filing_date < self.filings_since:
                continue
            if not accession or not primary_doc:
                logger.warning("edgar_incomplete_filing_entry", cik=cik, accession=accession)
                continue
            filings.append(
                {
                    "accession_number": accession,
                    "filing_date": filing_date,
                    "primary_document": primary_doc,
                }
            )
            if len(filings) >= self.max_filings_per_ticker:
                break

        logger.info("edgar_submissions_filtered", cik=cik, filings_found=len(filings))
        return filings

    # ── Fetch a filing document ──────────────────────────────────

    async def fetch_filing_document(
        self,
        document_url: str,
        session: aiohttp.ClientSession,
        ticker: str = "",
    ) -> str | None:
        """Fetch a filing's primary document from the EDGAR archives.

        Returns the raw document body (HTML or text), or None on failure (FR-5).
        """
        headers = {"User-Agent": self.user_agent}

        async def _do_fetch() -> str:
            t0 = time.perf_counter()
            byte_chunks: list[bytes] = []
            bytes_received = 0
            last_log_bytes = 0

            async with self._semaphore, session.get(document_url, headers=headers) as resp:
                resp.raise_for_status()
                content_length = resp.content_length  # may be None
                async for chunk in resp.content.iter_chunked(1024 * 64):  # 64 KB chunks
                    byte_chunks.append(chunk)
                    bytes_received += len(chunk)
                    if bytes_received - last_log_bytes >= _PROGRESS_LOG_INTERVAL:
                        logger.info(
                            "edgar_fetch_progress",
                            url=document_url,
                            received_mb=round(bytes_received / 1024 / 1024, 1),
                            total_mb=round(content_length / 1024 / 1024, 1) if content_length else None,
                        )
                        last_log_bytes = bytes_received
                await asyncio.sleep(1.0 / self.rate_limit_rps)

            raw_bytes = b"".join(byte_chunks)
            elapsed = time.perf_counter() - t0
            if ticker:
                EDGAR_REQUEST_DURATION.labels(ticker=ticker).observe(elapsed)
            logger.info(
                "edgar_fetch_complete",
                url=document_url,
                elapsed_ms=round(elapsed * 1000, 1),
                size_mb=round(len(raw_bytes) / 1024 / 1024, 2),
            )
            return raw_bytes.decode("utf-8", errors="replace")

        return await self._with_retries(  # type: ignore[no-any-return]
            _do_fetch, failure_value=None, operation_name="edgar_fetch", url=document_url
        )

    # ── XBRL company facts ───────────────────────────────────────

    async def fetch_company_facts(
        self,
        cik: int,
        session: aiohttp.ClientSession,
    ) -> dict[str, Any] | None:
        """Fetch the XBRL companyfacts document for a CIK.

        Returns the parsed JSON, or None on failure.
        """
        return await self._get_json(
            COMPANY_FACTS_URL.format(cik=cik),
            session,
            operation_name="edgar_company_facts",
            cik=cik,
        )

    # ── End-to-end per ticker ────────────────────────────────────

    async def get_filings_for_ticker(
        self,
        ticker: str,
        company_name: str,
        session: aiohttp.ClientSession,
    ) -> list[Filing]:
        """Resolve the ticker, list its 10-Ks, fetch and parse each document.

        Implements:
          - FR-1: Fetch 10-K filings from SEC EDGAR by ticker.
          - FR-5: Log and skip filings that fail; continue with rest.
        """
        t_ticker = time.perf_counter()

        resolved = await self.resolve_cik(ticker, session)
        if resolved is None:
            FILINGS_FETCHED_TOTAL.labels(ticker=ticker, status="skipped").inc()
            return []
        cik, official_name = resolved

        entries = await self.list_10k_filings(cik, session)
        filings: list[Filing] = []

        for entry in entries:
            accession = entry["accession_number"]
            accession_no_dashes = accession.replace("-", "")
            source_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}"
                f"/{accession_no_dashes}/{entry['primary_document']}"
            )

            logger.info(
                "edgar_fetch_started",
                ticker=ticker,
                accession=accession,
                filing_date=entry["filing_date"],
                url=source_url,
            )
            document = await self.fetch_filing_document(source_url, session, ticker=ticker)
            if document is None:
                logger.warning("edgar_fetch_skipped", ticker=ticker, accession=accession)
                FILINGS_FETCHED_TOTAL.labels(ticker=ticker, status="skipped").inc()
                continue

            clean_text = extract_text(document)
            if not clean_text:
                logger.warning("edgar_parse_empty", ticker=ticker, accession=accession)
                FILINGS_FETCHED_TOTAL.labels(ticker=ticker, status="skipped").inc()
                continue

            logger.info(
                "edgar_document_parsed",
                ticker=ticker,
                accession=accession,
                raw_kb=round(len(document) / 1024, 1),
                clean_kb=round(len(clean_text) / 1024, 1),
            )

            filings.append(
                Filing(
                    accession_number=accession,
                    ticker=ticker,
                    company_name=official_name or company_name,
                    filing_date=entry["filing_date"],
                    filing_type="10-K",
                    source_url=source_url,
                    raw_text=clean_text,
                )
            )

        logger.info(
            "edgar_ticker_complete",
            ticker=ticker,
            filings_found=len(filings),
            elapsed_ms=round((time.perf_counter() - t_ticker) * 1000, 1),
        )
        return filings
