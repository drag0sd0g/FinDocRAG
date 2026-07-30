"""Section-aware chunking for 10-K filings.

Implements the chunking strategy from TDD Section 5.2.2:
  1. Section split — by 10-K item headers (Item 1, 1A, 7, etc.), keeping
     only the occurrence of each item that holds its actual body text
     (every 10-K lists all of its items twice: once in the table of
     contents, once as the real section header)
  2. Paragraph split — by double newlines within each section
  3. Paragraph packing — consecutive paragraphs are greedily packed into
     chunks of up to 512 tokens, so short paragraphs (headings, single
     sentences) never become their own low-signal chunks
  4. Token-based windowing — only paragraphs that alone exceed 512 tokens
     are split into 512-token windows with 64-token overlap

Each chunk also exposes ``embedding_text`` — the chunk text prefixed with
a contextual header (ticker, filing date, section) that is embedded but
not stored, which measurably improves retrieval on corpus-wide queries.

References:
  - TDD: FR-7 (section-aware splitting with 512/64 token window)
  - TDD: FR-8 (chunk metadata: ticker, filing_date, accession, section, index)
  - TDD: Section 5.2.2 Chunking Strategy
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken

# ── Constants ────────────────────────────────────────────────────

# 10-K section headers we recognise (TDD Section 5.2.2 step 1)
_ITEM_PATTERN = re.compile(
    r"(?:^|\n)"                   # start of string or newline
    r"(?:ITEM|Item)\s+"           # "Item" or "ITEM"
    r"(1A|1B|1|2|3|4|5|6|7A|7|8|9A|9B|9|10|11|12|13|14|15)"
    r"[.\s:—\-]",                 # followed by punctuation/space
    re.MULTILINE,
)

DEFAULT_CHUNK_SIZE = 512   # tokens
DEFAULT_OVERLAP = 64       # tokens
TIKTOKEN_ENCODING = "cl100k_base"


@dataclass
class Chunk:
    """A single text chunk with metadata."""

    chunk_id: str          # SHA256(accession_number + section_name + chunk_index)
    accession_number: str
    ticker: str
    filing_date: str
    section_name: str
    chunk_index: int
    text: str
    token_count: int

    @property
    def embedding_text(self) -> str:
        """Chunk text prefixed with a contextual header, used for embedding only.

        The header anchors the vector to the filing's identity so that
        queries like "Apple supply chain risks" match AAPL chunks even when
        the chunk body never repeats the company name.  The stored/displayed
        text (``self.text``) is unchanged.
        """
        return (
            f"[{self.ticker} | 10-K | filed {self.filing_date} | {self.section_name}]\n"
            f"{self.text}"
        )


# ── Tokeniser (cached) ──────────────────────────────────────────

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Return the cl100k_base encoder (lazily initialised)."""
    global _encoder  # noqa: PLW0603
    if _encoder is None:
        _encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding (TDD Section 5.4)."""
    return len(_get_encoder().encode(text))


# ── Chunk ID generation ─────────────────────────────────────────

def make_chunk_id(accession_number: str, section_name: str, chunk_index: int) -> str:
    """Deterministic chunk ID: SHA256(accession + section + index).

    TDD Section 5.2.2 step 4.
    """
    raw = f"{accession_number}{section_name}{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Stage 1: Section split ──────────────────────────────────────

def _select_section_boundaries(
    text: str,
    matches: list[re.Match[str]],
) -> list[re.Match[str]]:
    """Choose which Item matches are real section headers.

    Every 10-K names each of its items at least twice: once as a row in the
    table of contents near the front, and once as the header of the actual
    section.  Treating all of them as boundaries produces ~20 junk sections
    holding nothing but TOC rows, and — worse — mislabels real body text with
    whichever TOC row happened to precede it.

    For each item number we therefore keep only the occurrence that owns the
    most text, which is the body header rather than the one-line TOC row.
    Ties go to the later occurrence, since the body always follows the TOC.
    Matches that lose are not boundaries, so their text is absorbed into the
    preceding section instead of being dropped — the TOC ends up in the
    Preamble, where it belongs.
    """
    best: dict[str, tuple[re.Match[str], int]] = {}

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        span = end - match.end()
        item_number = match.group(1)
        current = best.get(item_number)
        # >= so that, at equal length, the later (body) occurrence wins.
        if current is None or span >= current[1]:
            best[item_number] = (match, span)

    return sorted((match for match, _ in best.values()), key=lambda m: m.start())


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split a 10-K filing into (section_name, section_text) pairs.

    Uses regex to find Item headers, keeping one boundary per item number
    (see _select_section_boundaries). Text before the first item is
    labelled "Preamble". If no items are found, the entire document
    is returned as a single "Full Document" section.
    """
    matches = list(_ITEM_PATTERN.finditer(text))

    if not matches:
        return [("Full Document", text)]

    boundaries = _select_section_boundaries(text, matches)

    sections: list[tuple[str, str]] = []

    # Text before the first boundary → Preamble (cover page + table of contents)
    if boundaries[0].start() > 0:
        preamble = text[: boundaries[0].start()].strip()
        if preamble:
            sections.append(("Preamble", preamble))

    for i, match in enumerate(boundaries):
        item_number = match.group(1)
        section_name = f"Item {item_number}"
        start = match.end()
        end = boundaries[i + 1].start() if i + 1 < len(boundaries) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((section_name, section_text))

    return sections


# ── Stage 2: Paragraph split ────────────────────────────────────

def split_into_paragraphs(text: str) -> list[str]:
    """Split section text by double newlines (TDD step 2).

    Returns non-empty paragraphs only.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


# ── Stage 3: Token-based windowing ──────────────────────────────

def split_by_token_window(
    text: str,
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split text into token-count windows with overlap (TDD step 3).

    If the text fits within max_tokens, returns it as a single window.
    """
    encoder = _get_encoder()
    tokens = encoder.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    windows: list[str] = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        window_tokens = tokens[start:end]
        windows.append(encoder.decode(window_tokens))
        # Advance by (max_tokens - overlap) so the next window overlaps
        start += max_tokens - overlap

    return windows


# ── Stage 2b: Paragraph packing ─────────────────────────────────

def pack_paragraphs(paragraphs: list[str], max_tokens: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Greedily pack consecutive paragraphs into groups of ≤ max_tokens.

    Short paragraphs (headings, one-liners, table fragments) are merged
    with their neighbours instead of becoming their own low-signal chunks.
    A paragraph that alone exceeds max_tokens is emitted as its own group
    (the caller window-splits it).
    """
    groups: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        tokens = count_tokens(paragraph)

        if tokens > max_tokens:
            if current:
                groups.append("\n\n".join(current))
                current, current_tokens = [], 0
            groups.append(paragraph)  # oversize — window-split downstream
            continue

        # +1 accounts for the "\n\n" joiner between paragraphs.
        if current and current_tokens + tokens + 1 > max_tokens:
            groups.append("\n\n".join(current))
            current, current_tokens = [], 0

        current.append(paragraph)
        current_tokens += tokens + 1

    if current:
        groups.append("\n\n".join(current))

    return groups


# ── Public API: full chunking pipeline ───────────────────────────

def chunk_filing(
    raw_text: str,
    accession_number: str,
    ticker: str,
    filing_date: str,
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Run the full chunking pipeline on a filing.

    Returns a list of Chunk objects ready for embedding and storage.
    Implements FR-7, FR-8.
    """
    chunks: list[Chunk] = []
    global_index = 0

    sections = split_into_sections(raw_text)

    for section_name, section_text in sections:
        paragraphs = split_into_paragraphs(section_text)
        groups = pack_paragraphs(paragraphs, max_tokens)

        for group in groups:
            windows = split_by_token_window(group, max_tokens, overlap)

            for window_text in windows:
                token_count = count_tokens(window_text)
                chunk_id = make_chunk_id(accession_number, section_name, global_index)
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        accession_number=accession_number,
                        ticker=ticker,
                        filing_date=filing_date,
                        section_name=section_name,
                        chunk_index=global_index,
                        text=window_text,
                        token_count=token_count,
                    )
                )
                global_index += 1

    return chunks
