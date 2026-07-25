"""Unit tests for XBRL companyfacts extraction."""

from __future__ import annotations

from typing import Any

from src.facts import FinancialFact, extract_annual_facts


def _companyfacts(concept_items: dict[str, list[dict[str, Any]]], unit: str = "USD") -> dict[str, Any]:
    """Build a minimal companyfacts JSON document."""
    return {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                concept: {"units": {unit: items}}
                for concept, items in concept_items.items()
            }
        },
    }


def _annual_item(
    end: str,
    val: float,
    filed: str,
    start: str | None = None,
    form: str = "10-K",
    fp: str = "FY",
) -> dict[str, Any]:
    item: dict[str, Any] = {"end": end, "val": val, "filed": filed, "form": form, "fp": fp, "fy": 2024}
    if start is not None:
        item["start"] = start
    return item


class TestExtractAnnualFacts:
    def test_extracts_annual_duration_fact(self) -> None:
        doc = _companyfacts({
            "NetIncomeLoss": [
                _annual_item(start="2023-10-01", end="2024-09-28", val=93_736_000_000, filed="2024-11-01"),
            ],
        })
        facts = extract_annual_facts(doc, ticker="AAPL", cik=320193)
        assert len(facts) == 1
        fact = facts[0]
        assert fact.concept == "NetIncomeLoss"
        assert fact.fiscal_year == 2024
        assert fact.period_end == "2024-09-28"
        assert fact.value == 93_736_000_000
        assert fact.ticker == "AAPL"
        assert fact.cik == 320193

    def test_skips_quarterly_periods(self) -> None:
        doc = _companyfacts({
            "NetIncomeLoss": [
                # ~3 month duration — not an annual figure
                _annual_item(start="2024-06-30", end="2024-09-28", val=14_736_000_000, filed="2024-11-01"),
            ],
        })
        assert extract_annual_facts(doc, ticker="AAPL", cik=320193) == []

    def test_skips_non_10k_forms(self) -> None:
        doc = _companyfacts({
            "NetIncomeLoss": [
                _annual_item(
                    start="2023-10-01", end="2024-09-28", val=1.0,
                    filed="2024-08-01", form="10-Q", fp="Q3",
                ),
            ],
        })
        assert extract_annual_facts(doc, ticker="AAPL", cik=320193) == []

    def test_instant_facts_need_no_duration(self) -> None:
        doc = _companyfacts({
            "Assets": [_annual_item(end="2024-09-28", val=364_980_000_000, filed="2024-11-01")],
        })
        facts = extract_annual_facts(doc, ticker="AAPL", cik=320193)
        assert len(facts) == 1
        assert facts[0].concept == "Assets"

    def test_latest_filed_value_wins_for_same_period(self) -> None:
        """Comparative columns re-report the same period; restatements must win."""
        doc = _companyfacts({
            "NetIncomeLoss": [
                _annual_item(start="2022-09-25", end="2023-09-30", val=96_995_000_000, filed="2023-11-03"),
                _annual_item(start="2022-09-25", end="2023-09-30", val=97_000_000_000, filed="2024-11-01"),
            ],
        })
        facts = extract_annual_facts(doc, ticker="AAPL", cik=320193)
        assert len(facts) == 1
        assert facts[0].value == 97_000_000_000
        assert facts[0].filed == "2024-11-01"

    def test_untracked_concepts_are_ignored(self) -> None:
        doc = _companyfacts({
            "SomeObscureConcept": [
                _annual_item(start="2023-10-01", end="2024-09-28", val=1.0, filed="2024-11-01"),
            ],
        })
        assert extract_annual_facts(doc, ticker="AAPL", cik=320193) == []

    def test_unaccepted_units_are_ignored(self) -> None:
        doc = _companyfacts(
            {"NetIncomeLoss": [
                _annual_item(start="2023-10-01", end="2024-09-28", val=1.0, filed="2024-11-01"),
            ]},
            unit="EUR",
        )
        assert extract_annual_facts(doc, ticker="AAPL", cik=320193) == []

    def test_eps_in_usd_per_share(self) -> None:
        doc = _companyfacts(
            {"EarningsPerShareDiluted": [
                _annual_item(start="2023-10-01", end="2024-09-28", val=6.08, filed="2024-11-01"),
            ]},
            unit="USD/shares",
        )
        facts = extract_annual_facts(doc, ticker="AAPL", cik=320193)
        assert len(facts) == 1
        assert facts[0].unit == "USD/shares"
        assert facts[0].value == 6.08

    def test_missing_fields_are_skipped(self) -> None:
        doc = _companyfacts({
            "NetIncomeLoss": [
                {"end": "2024-09-28", "form": "10-K", "fp": "FY"},          # no val/filed
                {"val": 1.0, "form": "10-K", "fp": "FY", "filed": "2024-11-01"},  # no end
            ],
        })
        assert extract_annual_facts(doc, ticker="AAPL", cik=320193) == []

    def test_empty_document(self) -> None:
        assert extract_annual_facts({}, ticker="AAPL", cik=320193) == []

    def test_multiple_years_sorted(self) -> None:
        doc = _companyfacts({
            "Revenues": [
                _annual_item(start="2023-10-01", end="2024-09-28", val=391.0, filed="2024-11-01"),
                _annual_item(start="2022-09-25", end="2023-09-30", val=383.0, filed="2023-11-03"),
            ],
        })
        facts = extract_annual_facts(doc, ticker="AAPL", cik=320193)
        assert [f.fiscal_year for f in facts] == [2023, 2024]


class TestFinancialFactDataclass:
    def test_stores_all_fields(self) -> None:
        fact = FinancialFact(
            ticker="AAPL",
            cik=320193,
            concept="NetIncomeLoss",
            label="Net income",
            unit="USD",
            fiscal_year=2024,
            period_end="2024-09-28",
            value=93_736_000_000.0,
            filed="2024-11-01",
        )
        assert fact.label == "Net income"
        assert fact.fiscal_year == 2024
