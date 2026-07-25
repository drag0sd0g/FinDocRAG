"""Unit tests for XBRL facts lookup and injection (src/rag/facts.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from src.rag.facts import (
    FactsRepository,
    detect_metric_concepts,
    extract_years,
    format_fact_value,
)

# ── Metric intent detection ──────────────────────────────────────


class TestDetectMetricConcepts:
    def test_revenue_keywords(self) -> None:
        groups = detect_metric_concepts("What was Apple's total revenue in 2024?")
        assert ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"] in groups

    def test_net_sales_maps_to_revenue(self) -> None:
        groups = detect_metric_concepts("Report net sales for fiscal 2023")
        assert any("Revenues" in g for g in groups)

    def test_eps_beats_net_income(self) -> None:
        """'diluted earnings per share' must match EPS, and the EPS group
        must come first (more specific keyword group)."""
        groups = detect_metric_concepts("What was diluted earnings per share?")
        assert groups[0] == ["EarningsPerShareDiluted"]

    def test_rnd_abbreviation(self) -> None:
        groups = detect_metric_concepts("How much did MSFT spend on R&D?")
        assert ["ResearchAndDevelopmentExpense"] in groups

    def test_non_financial_question_matches_nothing(self) -> None:
        assert detect_metric_concepts("What are the main supply chain risks?") == []

    def test_case_insensitive(self) -> None:
        assert detect_metric_concepts("NET INCOME please") == [["NetIncomeLoss"]]


class TestExtractYears:
    def test_single_year(self) -> None:
        assert extract_years("revenue in 2024") == [2024]

    def test_multiple_years(self) -> None:
        assert extract_years("compare 2022 and 2024") == [2022, 2024]

    def test_no_years(self) -> None:
        assert extract_years("latest revenue") == []

    def test_ignores_non_year_numbers(self) -> None:
        assert extract_years("top 100 customers, 391035 million") == []


class TestFormatFactValue:
    def test_large_usd_value(self) -> None:
        assert format_fact_value(391_035_000_000.0, "USD") == "$391,035,000,000"

    def test_per_share_value(self) -> None:
        assert format_fact_value(6.08, "USD/shares") == "$6.08 per share"

    def test_fractional_usd(self) -> None:
        assert format_fact_value(1234.5, "USD") == "$1,234.50"


# ── FactsRepository ──────────────────────────────────────────────


def _repo_with_rows(rows: list[tuple[Any, ...]]) -> tuple[FactsRepository, MagicMock]:
    repo = FactsRepository(dsn="postgresql://fake")
    mock_cur = MagicMock()
    mock_cur.fetchall.return_value = rows
    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_conn.closed = False
    repo._conn = mock_conn
    return repo, mock_cur


# Row shape: (concept, label, unit, fiscal_year, period_end, value)
_REVENUE_2024 = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Total revenue",
    "USD",
    2024,
    "2024-09-28",
    391_035_000_000.0,
)
_REVENUE_2023 = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Total revenue",
    "USD",
    2023,
    "2023-09-30",
    383_285_000_000.0,
)


class TestFactsForQuestion:
    def test_no_ticker_returns_empty(self) -> None:
        repo, mock_cur = _repo_with_rows([_REVENUE_2024])
        assert repo.facts_for_question("What was total revenue in 2024?", None) == []
        mock_cur.execute.assert_not_called()

    def test_no_metric_keywords_returns_empty(self) -> None:
        repo, mock_cur = _repo_with_rows([_REVENUE_2024])
        assert repo.facts_for_question("Summarise the risk factors", "AAPL") == []
        mock_cur.execute.assert_not_called()

    def test_injects_fact_chunk(self) -> None:
        repo, _ = _repo_with_rows([_REVENUE_2024])
        chunks = repo.facts_for_question("What was Apple's revenue in 2024?", "AAPL")
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_id == "xbrl:AAPL:RevenueFromContractWithCustomerExcludingAssessedTax:2024"
        assert chunk.section == "XBRL Financial Facts"
        assert chunk.relevance_score == 1.0
        assert "$391,035,000,000" in chunk.text
        assert "fiscal year 2024" in chunk.text

    def test_no_year_asked_returns_most_recent(self) -> None:
        repo, _ = _repo_with_rows([_REVENUE_2024, _REVENUE_2023])
        chunks = repo.facts_for_question("What is Apple's latest revenue?", "AAPL")
        assert len(chunks) == 1
        assert "2024" in chunks[0].chunk_id

    def test_specific_years_selected(self) -> None:
        repo, _ = _repo_with_rows([_REVENUE_2024, _REVENUE_2023])
        chunks = repo.facts_for_question("Compare revenue in 2023 and 2024", "AAPL")
        years = {c.chunk_id.rsplit(":", 1)[1] for c in chunks}
        assert years == {"2023", "2024"}

    def test_higher_priority_concept_wins_per_year(self) -> None:
        """When both revenue aliases exist for a year, the first concept in
        the group's priority order must win."""
        fallback = ("Revenues", "Total revenue", "USD", 2024, "2024-09-28", 1.0)
        repo, _ = _repo_with_rows([fallback, _REVENUE_2024])
        chunks = repo.facts_for_question("Revenue in 2024?", "AAPL")
        assert len(chunks) == 1
        assert "RevenueFromContractWithCustomerExcludingAssessedTax" in chunks[0].chunk_id

    def test_db_error_degrades_to_no_facts(self) -> None:
        repo, mock_cur = _repo_with_rows([])
        mock_cur.execute.side_effect = RuntimeError("relation financial_facts does not exist")
        assert repo.facts_for_question("Revenue in 2024?", "AAPL") == []

    def test_facts_capped_at_max(self) -> None:
        rows = [
            ("RevenueFromContractWithCustomerExcludingAssessedTax", "Total revenue", "USD", 2015 + i,
             f"{2015 + i}-09-30", float(i))
            for i in range(12)
        ]
        repo, _ = _repo_with_rows(rows)
        question = "Revenue in " + ", ".join(str(2015 + i) for i in range(12))
        chunks = repo.facts_for_question(question, "AAPL")
        assert len(chunks) == 8

    def test_query_filters_by_ticker(self) -> None:
        repo, mock_cur = _repo_with_rows([_REVENUE_2024])
        repo.facts_for_question("Revenue in 2024?", "AAPL")
        sql, params = mock_cur.execute.call_args[0]
        assert "ticker = %s" in sql
        assert params[0] == "AAPL"


class TestRepositoryLifecycle:
    def test_close_without_connect_is_noop(self) -> None:
        repo = FactsRepository(dsn="postgresql://fake")
        repo.close()  # must not raise
        assert repo._conn is None

    def test_close_closes_connection(self) -> None:
        repo, _ = _repo_with_rows([])
        conn = repo._conn
        repo.close()
        assert conn is not None
        conn.close.assert_called_once()
        assert repo._conn is None
