"""Unit tests for the EDGAR HTML → clean text parser."""

from __future__ import annotations

from src.html_parser import extract_text

# ── Plain-text passthrough ───────────────────────────────────────


class TestPlainTextPassthrough:
    def test_empty_input(self) -> None:
        assert extract_text("") == ""

    def test_plain_text_is_normalised_not_parsed(self) -> None:
        text = "Item 1.  Business\n\n\n\nWe   are a company."
        result = extract_text(text)
        assert result == "Item 1. Business\n\nWe are a company."

    def test_plain_text_preserves_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        assert extract_text(text) == text


# ── HTML parsing ─────────────────────────────────────────────────


class TestHtmlParsing:
    def test_strips_tags(self) -> None:
        html = "<html><body><p>Item 1. Business</p><p>We sell devices.</p></body></html>"
        result = extract_text(html)
        assert "<p>" not in result
        assert "Item 1. Business" in result
        assert "We sell devices." in result

    def test_paragraphs_become_blank_line_separated(self) -> None:
        html = "<html><body><p>First.</p><p>Second.</p></body></html>"
        result = extract_text(html)
        assert "First.\n\nSecond." in result

    def test_drops_script_and_style(self) -> None:
        html = (
            "<html><head><style>p { color: red }</style></head>"
            "<body><script>alert('x')</script><p>Visible text.</p></body></html>"
        )
        result = extract_text(html)
        assert "Visible text." in result
        assert "alert" not in result
        assert "color" not in result

    def test_drops_inline_xbrl_hidden_header(self) -> None:
        html = (
            "<html><body>"
            "<ix:header><ix:hidden>MACHINE-ONLY-FACT</ix:hidden></ix:header>"
            "<p>Human readable.</p></body></html>"
        )
        result = extract_text(html)
        assert "Human readable." in result
        assert "MACHINE-ONLY-FACT" not in result

    def test_inline_spans_do_not_break_words(self) -> None:
        # EDGAR filings wrap runs of text in adjacent spans.
        html = "<html><body><div><span>Net </span><span>sales</span></div></body></html>"
        result = extract_text(html)
        assert "Net sales" in result

    def test_nbsp_normalised_to_space(self) -> None:
        html = "<html><body><p>Total&nbsp;revenue</p></body></html>"
        assert "Total revenue" in extract_text(html)

    def test_no_triple_blank_lines(self) -> None:
        html = (
            "<html><body><div><div><div><p>A</p></div></div></div>"
            "<div><p>B</p></div></body></html>"
        )
        result = extract_text(html)
        assert "\n\n\n" not in result


# ── Table rendering ──────────────────────────────────────────────


class TestTableRendering:
    def test_table_rows_become_pipe_separated(self) -> None:
        html = (
            "<html><body><table>"
            "<tr><td>Net sales</td><td>391,035</td><td>383,285</td></tr>"
            "<tr><td>Operating income</td><td>123,216</td><td>114,301</td></tr>"
            "</table></body></html>"
        )
        result = extract_text(html)
        assert "Net sales | 391,035 | 383,285" in result
        assert "Operating income | 123,216 | 114,301" in result

    def test_empty_cells_are_dropped(self) -> None:
        html = (
            "<html><body><table>"
            "<tr><td></td><td>2024</td><td></td><td>2023</td></tr>"
            "</table></body></html>"
        )
        assert "2024 | 2023" in extract_text(html)

    def test_nested_markup_in_cells_is_flattened(self) -> None:
        html = (
            "<html><body><table>"
            "<tr><td><span>R&amp;D</span> <b>expense</b></td><td><div>31,370</div></td></tr>"
            "</table></body></html>"
        )
        assert "R&D expense | 31,370" in extract_text(html)

    def test_header_cells_included(self) -> None:
        html = (
            "<html><body><table>"
            "<tr><th>Segment</th><th>Revenue</th></tr>"
            "<tr><td>iPhone</td><td>201,183</td></tr>"
            "</table></body></html>"
        )
        result = extract_text(html)
        assert "Segment | Revenue" in result
        assert "iPhone | 201,183" in result


# ── Realistic 10-K shape ─────────────────────────────────────────


class TestSectionHeadersSurviveParsing:
    def test_item_headers_stay_on_their_own_lines(self) -> None:
        """The embedding worker's section splitter needs 'Item N.' at line starts."""
        html = (
            "<html><body>"
            "<div><span>Item 1A.</span><span> Risk Factors</span></div>"
            "<p>The Company faces risks.</p>"
            "<div>Item 7. Management's Discussion</div>"
            "<p>Revenue grew.</p>"
            "</body></html>"
        )
        result = extract_text(html)
        lines = result.split("\n")
        assert any(line.startswith("Item 1A.") for line in lines)
        assert any(line.startswith("Item 7.") for line in lines)
