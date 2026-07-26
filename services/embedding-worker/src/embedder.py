"""Embedding generation using sentence-transformers.

References:
  - TDD: FR-9 (dense embeddings for each chunk)
  - TDD: Section 5.2.2 Embedding (batch of 64, loaded once at startup)

Model: nomic-ai/nomic-embed-text-v1.5 (768-dim, 8192-token context).
Chunks are packed to a 512-token budget with a contextual header, so the
model's long context means the *entire* chunk is embedded — all-MiniLM-L6-v2
silently truncated at 256 tokens, dropping roughly half of every chunk.

nomic is an asymmetric retrieval model: documents must be prefixed with
``search_document:`` and queries with ``search_query:`` (the query side lives
in the query-api retriever). Getting the prefix wrong degrades retrieval, so
the document prefix is applied here, once, for every chunk we embed.
"""

from __future__ import annotations

import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_BATCH_SIZE = 64

# nomic asymmetric-retrieval task prefix for the passage/document side.
DEFAULT_DOCUMENT_PREFIX = "search_document: "


class Embedder:
    """Wraps a sentence-transformers model for batch embedding.

    The model is loaded once on construction and held in memory
    (TDD Section 5.2.2).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        document_prefix: str = DEFAULT_DOCUMENT_PREFIX,
    ) -> None:
        logger.info("embedder_loading_model", model=model_name)
        # trust_remote_code: nomic ships a custom BERT (nomic-bert-2048); the
        # flag is ignored by standard sentence-transformers models.
        self._model = SentenceTransformer(model_name, trust_remote_code=True)
        self._dimension = self._model.get_sentence_embedding_dimension()
        self._document_prefix = document_prefix
        logger.info(
            "embedder_model_loaded",
            model=model_name,
            dimension=self._dimension,
            max_seq_length=self._model.max_seq_length,
            document_prefix=document_prefix,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension (768 for nomic-embed-text)."""
        return int(self._dimension or 0)

    def embed(self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Each text is prefixed with the asymmetric ``search_document:`` task
        prefix before encoding. Returns a list of float vectors, one per input
        text. Uses batch_size=64 for throughput (TDD Section 5.2.2).
        """
        prefixed = [f"{self._document_prefix}{text}" for text in texts]
        embeddings = self._model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vec.tolist() for vec in embeddings]
