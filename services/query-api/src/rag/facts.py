"""Structured XBRL facts — lookup and injection for numeric questions.

Financial figures ("what was total revenue in FY2024?") are answered far
more reliably from XBRL companyfacts than from prose retrieval, so the
ingestion service stores curated annual facts in the ``financial_facts``
table and this module injects the relevant ones into the RAG context.

Flow (per query): detect financial-metric intent from the question text,
look up matching facts for the requested ticker and fiscal years, and
return them as high-relevance RetrievedChunk entries that the generator
prepends to the retrieved context.  Questions without a ticker filter or
without metric keywords are unaffected.

References:
  - db/migrations/002_hybrid_search_and_facts.sql (financial_facts schema)
  - services/ingestion/src/facts.py (extraction side)
"""

from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

import psycopg2
import structlog

from src.rag.prompts import RetrievedChunk

logger = structlog.get_logger()

# Keyword → prioritised us-gaap concept list.  The first concept in a group
# that has data for the requested year wins (filers migrated revenue tags
# over the years, so several aliases may exist).
# Order matters: more specific phrases must precede generic ones
# ("earnings per share" before "earnings"/"net income").
_METRIC_KEYWORDS: list[tuple[tuple[str, ...], list[str]]] = [
    (
        ("earnings per share", "eps", "diluted earnings"),
        ["EarningsPerShareDiluted"],
    ),
    (
        ("research and development", "r&d"),
        ["ResearchAndDevelopmentExpense"],
    ),
    (
        ("operating income", "income from operations", "operating profit"),
        ["OperatingIncomeLoss"],
    ),
    (
        ("operating expense",),
        ["OperatingExpenses"],
    ),
    (
        ("gross profit", "gross margin"),
        ["GrossProfit"],
    ),
    (
        ("cost of revenue", "cost of sales", "cost of goods"),
        ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    ),
    (
        ("net income", "net profit", "net earnings", "bottom line"),
        ["NetIncomeLoss"],
    ),
    (
        ("total assets",),
        ["Assets"],
    ),
    (
        ("stockholders' equity", "stockholders equity", "shareholders' equity", "shareholders equity"),
        ["StockholdersEquity"],
    ),
    (
        ("cash and cash equivalents",),
        ["CashAndCashEquivalentsAtCarryingValue"],
    ),
    (
        ("revenue", "net sales", "total sales", "turnover"),
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ],
    ),
]

_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")

# Cap the number of injected facts so they cannot crowd out prose context.
_MAX_FACTS = 8


def detect_metric_concepts(question: str) -> list[list[str]]:
    """Return prioritised concept groups whose keywords appear in the question."""
    lowered = question.lower()
    return [
        concepts
        for keywords, concepts in _METRIC_KEYWORDS
        if any(kw in lowered for kw in keywords)
    ]


def extract_years(question: str) -> list[int]:
    """Extract four-digit years (fiscal year references) from the question."""
    return [int(y) for y in _YEAR_PATTERN.findall(question)]


def format_fact_value(value: float, unit: str) -> str:
    """Format an XBRL value for the prompt ("$391,035,000,000" / "$6.08 per share")."""
    if unit == "USD/shares":
        return f"${value:,.2f} per share"
    if value == int(value):
        return f"${int(value):,}"
    return f"${value:,.2f}"


class FactsRepository:
    """Reads curated annual XBRL facts from the financial_facts table."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn: psycopg2.extensions.connection | None = None

    def connect(self) -> None:
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = True
        logger.info("facts_repository_connected")

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
        cur = self._get_conn().cursor()
        try:
            yield cur
        finally:
            cur.close()

    def _lookup_group(
        self,
        ticker: str,
        concepts: list[str],
        years: list[int],
    ) -> list[tuple[Any, ...]]:
        """Fetch facts for one concept group; per fiscal year, the highest-priority
        concept with data wins."""
        sql = """
            SELECT concept, label, unit, fiscal_year, period_end, value
            FROM financial_facts
            WHERE ticker = %s AND concept = ANY(%s)
            ORDER BY fiscal_year DESC
        """
        with self._cursor() as cur:
            cur.execute(sql, (ticker, concepts))
            rows = cur.fetchall()

        by_year: dict[int, tuple[Any, ...]] = {}
        for row in rows:
            year = int(row[3])
            current = by_year.get(year)
            if current is None or concepts.index(row[0]) < concepts.index(current[0]):
                by_year[year] = row

        if years:
            selected_years = [y for y in years if y in by_year]
        elif by_year:
            selected_years = [max(by_year)]  # no year asked → most recent
        else:
            selected_years = []

        return [by_year[y] for y in selected_years]

    def facts_for_question(
        self,
        question: str,
        ticker: str | None,
    ) -> list[RetrievedChunk]:
        """Return XBRL facts relevant to the question as context chunks.

        Empty unless a ticker filter is set and the question names a
        tracked financial metric.  Lookup failures degrade to no facts —
        never to a failed query.
        """
        if not ticker:
            return []
        concept_groups = detect_metric_concepts(question)
        if not concept_groups:
            return []
        years = extract_years(question)

        chunks: list[RetrievedChunk] = []
        try:
            for concepts in concept_groups:
                for concept, label, unit, fiscal_year, period_end, value in self._lookup_group(
                    ticker, concepts, years
                ):
                    text = (
                        f"{ticker} {label}, fiscal year {fiscal_year} "
                        f"(period ending {period_end}): {format_fact_value(float(value), unit)}. "
                        f"Source: XBRL us-gaap:{concept}, SEC companyfacts (authoritative)."
                    )
                    chunks.append(
                        RetrievedChunk(
                            chunk_id=f"xbrl:{ticker}:{concept}:{fiscal_year}",
                            ticker=ticker,
                            filing_date=str(period_end),
                            section="XBRL Financial Facts",
                            relevance_score=1.0,
                            text=text,
                        )
                    )
        except Exception as exc:
            logger.warning("facts_lookup_failed", ticker=ticker, error=str(exc))
            return []

        if chunks:
            logger.info("facts_injected", ticker=ticker, facts=len(chunks))
        return chunks[:_MAX_FACTS]
