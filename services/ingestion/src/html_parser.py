"""HTML → clean text extraction for SEC EDGAR filing documents.

EDGAR primary documents are (inline-XBRL) HTML. Chunking and embedding
raw HTML wastes tokens on markup and destroys financial tables, so this
module converts a filing document to clean plain text:

  - <script>/<style>/<head> and the hidden <ix:header> XBRL block are dropped
  - <table> elements are rendered as pipe-separated rows so financial
    figures stay readable and adjacent to their row/column labels
  - block-level elements become paragraph breaks (double newline), which
    is exactly what the embedding worker's paragraph splitter expects
  - whitespace and non-breaking spaces are normalised

Plain-text documents (pre-2001 filings) pass through with whitespace
normalisation only.

References:
  - TDD: FR-1 (fetch 10-K filings), FR-5 (skip filings that fail to parse)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from bs4.element import Tag

# Elements whose content must never reach the text output.
_STRIP_TAGS = ["script", "style", "head", "title", "noscript"]

# Elements that terminate a paragraph when rendered.
_BLOCK_TAGS = [
    "p", "div", "br", "hr", "table", "tr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "ul", "ol", "section", "article",
]

# Cheap heuristic: does this look like an HTML document?
_HTML_MARKER = re.compile(r"<(?:html|body|div|p|table|span)[\s>]", re.IGNORECASE)


def _render_table(table: Tag) -> str:
    """Render an HTML table as pipe-separated text rows.

    Keeps figures next to their row labels ("Net sales | 391,035 | 383,285")
    so both the embedder and the LLM can read them.
    """
    lines: list[str] = []
    for row in table.find_all("tr"):
        cells = [
            cell.get_text(" ", strip=True)
            for cell in row.find_all(["td", "th"])
        ]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines into readable paragraphs."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(document: str) -> str:
    """Convert an EDGAR filing document (HTML or plain text) to clean text.

    Returns normalised plain text with paragraphs separated by blank lines
    and tables rendered as pipe-separated rows.
    """
    if not document:
        return ""

    if not _HTML_MARKER.search(document):
        # Plain-text filing — nothing to parse.
        return _normalise_whitespace(document)

    soup = BeautifulSoup(document, "lxml")

    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    # Inline-XBRL hidden header: machine-readable facts not shown to readers.
    for tag in soup.find_all(re.compile(r"^ix:header$", re.IGNORECASE)):
        tag.decompose()

    # Replace tables with their pre-rendered text before flattening,
    # so nested markup inside cells cannot fragment the rows.
    for table in soup.find_all("table"):
        rendered = _render_table(table)
        table.replace_with(soup.new_string(f"\n\n{rendered}\n\n"))

    # Mark paragraph boundaries: every block element appends a blank line.
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append(soup.new_string("\n\n"))

    return _normalise_whitespace(soup.get_text())
