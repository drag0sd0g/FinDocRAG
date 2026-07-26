-- FinDocDRAG Migration 003 — Embedding model upgrade: 384 -> 768 dims
-- See: docs/technical-design-document.md Section 5.2.2
--
-- Switches the dense embedding from all-MiniLM-L6-v2 (384-dim, 256-token
-- truncation) to nomic-embed-text-v1.5 (768-dim, 8192-token context). The old
-- 384-dim vectors cannot be cast to 768 dims and must be regenerated, so this
-- migration CLEARS document_chunks and ingestion_log to force a full
-- re-ingestion/re-embed. financial_facts are dimension-independent and kept.
--
-- Guarded + idempotent: the body only runs while the column is still
-- vector(384); on a database already at 768 (fresh install, or a re-run of the
-- migration set) it is a no-op, so it is safe to re-apply on every startup.

DO $$
DECLARE
    current_dim integer;
BEGIN
    -- pgvector stores the declared dimension directly in atttypmod.
    SELECT atttypmod INTO current_dim
    FROM pg_attribute
    WHERE attrelid = 'document_chunks'::regclass
      AND attname = 'embedding'
      AND NOT attisdropped;

    IF current_dim = 384 THEN
        RAISE NOTICE 'Migration 003: embedding vector(384) -> vector(768); clearing chunks for re-embed';

        -- The HNSW index is bound to the column type; drop before altering.
        DROP INDEX IF EXISTS idx_chunks_embedding;

        -- 384-dim data is invalid at 768; clear and re-ingest. CASCADE covers
        -- the document_chunks -> ingestion_log foreign key.
        TRUNCATE document_chunks, ingestion_log CASCADE;

        ALTER TABLE document_chunks
            ALTER COLUMN embedding TYPE vector(768);

        CREATE INDEX idx_chunks_embedding ON document_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200);

        RAISE NOTICE 'Migration 003: complete';
    ELSE
        RAISE NOTICE 'Migration 003: embedding already vector(%), skipping', current_dim;
    END IF;
END $$;
