"""Extraction of structured financial facts from XBRL companyfacts JSON.

Financial figures ("what was total revenue in FY2024?") live in XBRL, not
in filing prose, so the ingestion service stores a small curated set of
annual us-gaap facts in the ``financial_facts`` table.  The Query API
injects them as authoritative context for numeric questions.

References:
  - SEC companyfacts API: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
  - db/migrations/002_hybrid_search_and_facts.sql (financial_facts schema)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import structlog

logger = structlog.get_logger()

# Curated us-gaap concepts worth storing, with human-readable labels.
# Several revenue tags exist because filers migrated tags over the years.
TRACKED_CONCEPTS: dict[str, str] = {
    "RevenueFromContractWithCustomerExcludingAssessedTax": "Total revenue (net sales)",
    "Revenues": "Total revenue",
    "SalesRevenueNet": "Total revenue (net sales)",
    "CostOfRevenue": "Cost of revenue",
    "CostOfGoodsAndServicesSold": "Cost of sales",
    "GrossProfit": "Gross profit",
    "ResearchAndDevelopmentExpense": "Research and development expense",
    "OperatingExpenses": "Operating expenses",
    "OperatingIncomeLoss": "Operating income",
    "NetIncomeLoss": "Net income",
    "EarningsPerShareDiluted": "Diluted earnings per share",
    "Assets": "Total assets",
    "StockholdersEquity": "Total stockholders' equity",
    "CashAndCashEquivalentsAtCarryingValue": "Cash and cash equivalents",
}

# Accepted XBRL units per concept kind.
_ACCEPTED_UNITS = ("USD", "USD/shares")

# A "duration" fact must cover roughly one fiscal year.
_MIN_PERIOD_DAYS = 340
_MAX_PERIOD_DAYS = 380


@dataclass
class FinancialFact:
    """One annual financial fact extracted from XBRL companyfacts."""

    ticker: str
    cik: int
    concept: str          # us-gaap tag, e.g. "NetIncomeLoss"
    label: str            # human-readable, e.g. "Net income"
    unit: str             # "USD" or "USD/shares"
    fiscal_year: int      # year of the fiscal period end
    period_end: str       # ISO date
    value: float
    filed: str            # ISO date the containing filing was filed


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _is_annual_period(item: dict[str, Any]) -> bool:
    """True for instant facts, or duration facts spanning ~1 fiscal year."""
    start = item.get("start")
    if start is None:
        return True  # instant concept (balance sheet)
    start_d = _parse_date(str(start))
    end_d = _parse_date(str(item.get("end", "")))
    if start_d is None or end_d is None:
        return False
    return timedelta(days=_MIN_PERIOD_DAYS) <= (end_d - start_d) <= timedelta(days=_MAX_PERIOD_DAYS)


def extract_annual_facts(
    company_facts: dict[str, Any],
    ticker: str,
    cik: int,
) -> list[FinancialFact]:
    """Extract tracked annual facts from a companyfacts JSON document.

    Keeps facts reported in 10-K filings for full fiscal years (fp == "FY").
    The same period is re-reported in later filings (comparative columns);
    the most recently filed value wins so restatements take precedence.
    The fiscal year is labelled with the calendar year of the period end.
    """
    facts_obj = company_facts.get("facts")
    us_gaap: dict[str, Any] = facts_obj.get("us-gaap", {}) if isinstance(facts_obj, dict) else {}
    if not us_gaap:
        return []

    # (concept, unit, period_end) → best fact seen so far
    best: dict[tuple[str, str, str], FinancialFact] = {}

    for concept, label in TRACKED_CONCEPTS.items():
        units = us_gaap.get(concept, {}).get("units", {})
        for unit, items in units.items():
            if unit not in _ACCEPTED_UNITS:
                continue
            for item in items:
                if item.get("form") != "10-K" or item.get("fp") != "FY":
                    continue
                if not _is_annual_period(item):
                    continue
                end_d = _parse_date(str(item.get("end", "")))
                value = item.get("val")
                filed = item.get("filed")
                if end_d is None or filed is None or not isinstance(value, (int, float)):
                    continue

                fact = FinancialFact(
                    ticker=ticker,
                    cik=cik,
                    concept=concept,
                    label=label,
                    unit=unit,
                    fiscal_year=end_d.year,
                    period_end=end_d.isoformat(),
                    value=float(value),
                    filed=str(filed),
                )
                key = (concept, unit, fact.period_end)
                current = best.get(key)
                if current is None or fact.filed > current.filed:
                    best[key] = fact

    facts = sorted(best.values(), key=lambda f: (f.concept, f.period_end))
    logger.info("xbrl_facts_extracted", ticker=ticker, cik=cik, facts=len(facts))
    return facts
