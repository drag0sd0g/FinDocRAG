"""Pydantic request and response models for the Query API.

Matches the API contract in TDD Section 5.2.3 exactly.

References:
  - TDD: FR-12 (POST /v1/query request)
  - TDD: FR-17 (response with answer, sources, timing)
  - TDD: FR-19, FR-20 (GET /v1/documents)
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, Field, field_validator

# ── Query endpoint ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    """POST /v1/query request body (FR-12)."""

    question: str
    ticker_filter: str | None = None
    # Optional filing-date window (ISO dates). Useful when several years of
    # 10-Ks are ingested and the question targets a specific period.
    filing_date_from: str | None = None
    filing_date_to: str | None = None
    # When true, each source includes the full chunk text alongside the
    # 200-char preview (used by the evaluation harness for ragas scoring).
    include_source_text: bool = False
    # Upper bound of 20 is a context-window budget: at ~512 tokens per chunk,
    # 20 chunks consume ~10 K tokens, leaving headroom for the system prompt,
    # question, and answer inside a 32 K-token context window.  Raise the cap
    # only if you switch to a model with a larger context window.
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty or whitespace")
        return v.strip()

    @field_validator("filing_date_from", "filing_date_to")
    @classmethod
    def date_must_be_iso(cls, v: str | None) -> str | None:
        if v is None:
            return v
        datetime.date.fromisoformat(v)  # raises ValueError if malformed
        return v


class SourceChunk(BaseModel):
    """A single source chunk returned alongside the answer (FR-17)."""

    chunk_id: str
    ticker: str
    filing_date: str
    section: str
    relevance_score: float
    text_preview: str  # first 200 characters
    text: str | None = None  # full chunk text, only when include_source_text


class TimingInfo(BaseModel):
    """Latency breakdown returned in query responses (FR-17)."""

    embedding_ms: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    """POST /v1/query response body (FR-17)."""

    answer: str | None
    sources: list[SourceChunk]
    model: str
    timing: TimingInfo
    degraded: bool = False
    request_id: str = ""


# ── Documents endpoint ───────────────────────────────────────────

class DocumentInfo(BaseModel):
    """A single ingested filing (FR-19)."""

    accession_number: str
    ticker: str
    company_name: str
    filing_date: str
    filing_type: str
    chunk_count: int | None
    ingested_at: str


class DocumentListResponse(BaseModel):
    """GET /v1/documents response (FR-19, FR-20)."""

    documents: list[DocumentInfo]
    total: int
    limit: int
    offset: int
