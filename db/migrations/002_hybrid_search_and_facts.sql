-- FinDocDRAG Migration 002 — Hybrid search + structured financial facts
-- See: docs/technical-design-document.md Section 5.3
--
-- 1. Full-text search support on document_chunks (lexical leg of hybrid
--    retrieval — fused with vector search via Reciprocal Rank Fusion).
-- 2. financial_facts table for curated annual XBRL facts, injected into
--    query context for numeric financial questions.
--
-- All statements are idempotent: this file may be re-applied safely.

-- ============================================================
-- Full-text search column + GIN index
-- ============================================================
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS chunk_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_tsv
    ON document_chunks USING GIN (chunk_tsv);

-- ============================================================
-- Structured financial facts (XBRL companyfacts)
-- ============================================================
CREATE TABLE IF NOT EXISTS financial_facts (
    ticker       VARCHAR(10)  NOT NULL,
    cik          BIGINT       NOT NULL,
    concept      VARCHAR(120) NOT NULL,   -- us-gaap tag, e.g. NetIncomeLoss
    label        VARCHAR(120) NOT NULL,   -- human-readable label
    unit         VARCHAR(20)  NOT NULL,   -- USD | USD/shares
    fiscal_year  INTEGER      NOT NULL,   -- year of the fiscal period end
    period_end   DATE         NOT NULL,
    value        NUMERIC      NOT NULL,
    filed        DATE         NOT NULL,   -- filing date of the reporting 10-K
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, concept, unit, period_end)
);

CREATE INDEX IF NOT EXISTS idx_facts_ticker_year
    ON financial_facts(ticker, fiscal_year);
