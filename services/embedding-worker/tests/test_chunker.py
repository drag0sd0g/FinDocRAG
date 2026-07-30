"""Unit tests for the 10-K filing chunker.

Tests all three stages of the chunking pipeline plus the
end-to-end chunk_filing function.
"""

from __future__ import annotations

from src.chunker import (
    Chunk,
    chunk_filing,
    count_tokens,
    make_chunk_id,
    pack_paragraphs,
    split_by_token_window,
    split_into_paragraphs,
    split_into_sections,
)

# ── Token counting ───────────────────────────────────────────────

class TestCountTokens:
    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_simple_sentence(self) -> None:
        tokens = count_tokens("Hello world")
        assert tokens >= 2  # at least 2 tokens

    def test_longer_text(self) -> None:
        tokens = count_tokens("The quick brown fox jumps over the lazy dog.")
        assert tokens > 5


# ── Chunk ID ─────────────────────────────────────────────────────

class TestMakeChunkId:
    def test_deterministic(self) -> None:
        id1 = make_chunk_id("ACC001", "Item 1", 0)
        id2 = make_chunk_id("ACC001", "Item 1", 0)
        assert id1 == id2

    def test_different_inputs_produce_different_ids(self) -> None:
        id1 = make_chunk_id("ACC001", "Item 1", 0)
        id2 = make_chunk_id("ACC001", "Item 1", 1)
        assert id1 != id2

    def test_is_64_char_hex(self) -> None:
        cid = make_chunk_id("ACC001", "Item 1", 0)
        assert len(cid) == 64
        int(cid, 16)  # should not raise


# ── Stage 1: Section split ──────────────────────────────────────

class TestSplitIntoSections:
    def test_no_items_returns_full_document(self) -> None:
        text = "This is a plain document with no item headers."
        sections = split_into_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "Full Document"

    def test_single_item(self) -> None:
        text = "Preamble text.\nItem 1. Business\nWe are a company."
        sections = split_into_sections(text)
        names = [s[0] for s in sections]
        assert "Preamble" in names
        assert "Item 1" in names

    def test_multiple_items(self) -> None:
        text = (
            "Cover page.\n"
            "Item 1. Business\nWe sell things.\n"
            "Item 1A. Risk Factors\nThere are risks.\n"
            "Item 7. MD&A\nRevenue grew."
        )
        sections = split_into_sections(text)
        names = [s[0] for s in sections]
        assert "Item 1" in names
        assert "Item 1A" in names
        assert "Item 7" in names

    def test_case_insensitive_item(self) -> None:
        text = "ITEM 1. BUSINESS\nWe are a company."
        sections = split_into_sections(text)
        names = [s[0] for s in sections]
        assert "Item 1" in names


# ── Stage 1: table-of-contents handling ─────────────────────────

def _filing_with_toc() -> str:
    """A miniature 10-K: cover page, table of contents, then the real body.

    Every real filing names each item twice — once as a TOC row with a page
    number, once as the header of the actual section.
    """
    return (
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n"
        "Annual Report on Form 10-K\n"
        "\n"
        "TABLE OF CONTENTS\n"
        "Item 1. Business 3\n"
        "Item 1A. Risk Factors 12\n"
        "Item 7. Management's Discussion and Analysis 40\n"
        "\n"
        "Item 1. Business\n"
        "The Company designs and sells consumer electronics worldwide. " * 12 + "\n"
        "Item 1A. Risk Factors\n"
        "The Company depends on a concentrated supply chain in Asia. " * 12 + "\n"
        "Item 7. Management's Discussion and Analysis\n"
        "Total net sales increased 2% year over year to $391.0 billion. " * 12
    )


class TestTableOfContentsHandling:
    def test_each_item_yields_exactly_one_section(self) -> None:
        """TOC rows must not create a second section for the same item."""
        sections = split_into_sections(_filing_with_toc())
        names = [name for name, _ in sections]
        for item in ("Item 1", "Item 1A", "Item 7"):
            assert names.count(item) == 1, f"{item} appeared {names.count(item)} times"

    def test_sections_hold_body_text_not_toc_rows(self) -> None:
        """The kept boundary must be the body header, not the TOC row."""
        sections = dict(split_into_sections(_filing_with_toc()))
        assert "concentrated supply chain" in sections["Item 1A"]
        assert "Total net sales increased" in sections["Item 7"]

    def test_toc_rows_land_in_the_preamble(self) -> None:
        """Losing boundaries are absorbed, never dropped — no text is lost."""
        sections = dict(split_into_sections(_filing_with_toc()))
        assert "TABLE OF CONTENTS" in sections["Preamble"]
        # The page-numbered TOC row stays with the TOC, out of the body sections.
        assert "Risk Factors 12" in sections["Preamble"]
        assert "Risk Factors 12" not in sections["Item 1A"]

    def test_body_text_is_not_mislabelled_by_a_toc_row(self) -> None:
        """Regression: body prose must carry its own item's section name.

        With every TOC row treated as a boundary, the last TOC row swallowed
        the text that followed it, so real Item 1 prose was labelled with
        whichever item the TOC happened to list last.
        """
        sections = dict(split_into_sections(_filing_with_toc()))
        assert "designs and sells consumer electronics" in sections["Item 1"]

    def test_no_text_is_lost(self) -> None:
        sections = split_into_sections(_filing_with_toc())
        recombined = "".join(text for _, text in sections)
        assert "designs and sells consumer electronics" in recombined
        assert "concentrated supply chain" in recombined
        assert "Total net sales increased" in recombined

    def test_duplicate_item_keeps_the_longer_span(self) -> None:
        """A passing mention loses to the section that actually holds the body."""
        text = (
            "Cover.\n"
            "Item 3. Legal Proceedings 5\n"
            "Item 3. Legal Proceedings\n"
            + "The Company is party to various legal proceedings. " * 20
        )
        sections = dict(split_into_sections(text))
        assert "The Company is party" in sections["Item 3"]
        # The TOC row (title plus page number) stayed behind in the Preamble.
        assert "Legal Proceedings 5" not in sections["Item 3"]
        assert "Legal Proceedings 5" in sections["Preamble"]

    def test_single_occurrence_is_unaffected(self) -> None:
        """Documents without a TOC keep their original one-section-per-item split."""
        text = (
            "Cover page.\n"
            "Item 1. Business\nWe sell things.\n"
            "Item 7. MD&A\nRevenue grew."
        )
        names = [name for name, _ in split_into_sections(text)]
        assert names == ["Preamble", "Item 1", "Item 7"]


# ── Stage 2: Paragraph split ────────────────────────────────────

class TestSplitIntoParagraphs:
    def test_single_paragraph(self) -> None:
        assert split_into_paragraphs("No double newlines here.") == ["No double newlines here."]

    def test_multiple_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        result = split_into_paragraphs(text)
        assert len(result) == 3

    def test_ignores_empty_paragraphs(self) -> None:
        text = "A.\n\n\n\nB."
        result = split_into_paragraphs(text)
        assert len(result) == 2


# ── Stage 3: Token windowing ────────────────────────────────────

class TestSplitByTokenWindow:
    def test_short_text_returns_single_window(self) -> None:
        result = split_by_token_window("Hello world", max_tokens=512)
        assert len(result) == 1
        assert result[0] == "Hello world"

    def test_long_text_produces_multiple_windows(self) -> None:
        # Create text that is definitely > 10 tokens
        long_text = " ".join(["word"] * 200)
        result = split_by_token_window(long_text, max_tokens=10, overlap=2)
        assert len(result) > 1

    def test_overlap_produces_more_windows(self) -> None:
        long_text = " ".join(["word"] * 200)
        no_overlap = split_by_token_window(long_text, max_tokens=50, overlap=0)
        with_overlap = split_by_token_window(long_text, max_tokens=50, overlap=25)
        assert len(with_overlap) > len(no_overlap)


# ── End-to-end: chunk_filing ─────────────────────────────────────

class TestChunkFiling:
    def test_produces_chunks_with_metadata(self) -> None:
        text = "Item 1. Business\nWe are a company that does things.\n\nWe have products."
        chunks = chunk_filing(
            raw_text=text,
            accession_number="ACC001",
            ticker="AAPL",
            filing_date="2024-11-01",
        )
        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, Chunk)
            assert c.accession_number == "ACC001"
            assert c.ticker == "AAPL"
            assert c.filing_date == "2024-11-01"
            assert c.token_count > 0
            assert len(c.chunk_id) == 64

    def test_empty_text_returns_no_chunks(self) -> None:
        chunks = chunk_filing(
            raw_text="",
            accession_number="ACC001",
            ticker="AAPL",
            filing_date="2024-11-01",
        )
        assert chunks == []

    def test_chunk_ids_are_unique(self) -> None:
        text = (
            "Item 1. Business\nFirst paragraph.\n\nSecond paragraph.\n"
            "Item 7. MD&A\nRevenue discussion.\n\nExpense discussion."
        )
        chunks = chunk_filing(
            raw_text=text,
            accession_number="ACC001",
            ticker="AAPL",
            filing_date="2024-11-01",
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))  # all unique


# ── Paragraph packing ────────────────────────────────────────────

class TestPackParagraphs:
    def test_empty_list(self) -> None:
        assert pack_paragraphs([]) == []

    def test_short_paragraphs_are_merged(self) -> None:
        paragraphs = ["Item 7. MD&A", "Revenue grew 5% year over year.", "Margins expanded."]
        groups = pack_paragraphs(paragraphs, max_tokens=512)
        assert len(groups) == 1
        assert groups[0] == "\n\n".join(paragraphs)

    def test_budget_is_respected(self) -> None:
        paragraph = "word " * 50  # ~50 tokens
        groups = pack_paragraphs([paragraph.strip()] * 10, max_tokens=120)
        assert len(groups) > 1
        for group in groups:
            assert count_tokens(group) <= 120

    def test_oversize_paragraph_emitted_alone(self) -> None:
        small = "A short line."
        huge = "token " * 600  # exceeds the budget on its own
        groups = pack_paragraphs([small, huge.strip(), small], max_tokens=512)
        assert len(groups) == 3
        assert groups[0] == small
        assert groups[1] == huge.strip()
        assert groups[2] == small

    def test_order_is_preserved(self) -> None:
        paragraphs = [f"Paragraph number {i}." for i in range(20)]
        groups = pack_paragraphs(paragraphs, max_tokens=30)
        assert "\n\n".join(groups) == "\n\n".join(paragraphs)


# ── Contextual embedding text ────────────────────────────────────

class TestEmbeddingText:
    def _chunk(self) -> Chunk:
        return Chunk(
            chunk_id="x" * 64,
            accession_number="ACC001",
            ticker="AAPL",
            filing_date="2024-11-01",
            section_name="Item 1A",
            chunk_index=0,
            text="The Company faces supply chain risks.",
            token_count=7,
        )

    def test_header_contains_filing_identity(self) -> None:
        chunk = self._chunk()
        header, body = chunk.embedding_text.split("\n", 1)
        assert header == "[AAPL | 10-K | filed 2024-11-01 | Item 1A]"
        assert body == chunk.text

    def test_stored_text_is_unchanged(self) -> None:
        chunk = self._chunk()
        _ = chunk.embedding_text
        assert chunk.text == "The Company faces supply chain risks."
