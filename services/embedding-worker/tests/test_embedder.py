"""Unit tests for the embedder.

The actual sentence-transformers model is NOT loaded in tests —
we mock it to keep tests fast and CPU-light.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.embedder import DEFAULT_DOCUMENT_PREFIX, Embedder


class TestEmbedder:
    """Tests for the Embedder wrapper."""

    @patch("src.embedder.SentenceTransformer")
    def test_dimension_property(self, mock_st_class: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st_class.return_value = mock_model

        embedder = Embedder(model_name="test-model")
        assert embedder.dimension == 768

    @patch("src.embedder.SentenceTransformer")
    def test_loads_model_with_trust_remote_code(self, mock_st_class: MagicMock) -> None:
        # nomic ships a custom nomic-bert model that requires trust_remote_code.
        mock_st_class.return_value = MagicMock()
        Embedder(model_name="nomic-ai/nomic-embed-text-v1.5")
        mock_st_class.assert_called_once_with(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )

    @patch("src.embedder.SentenceTransformer")
    def test_embed_returns_list_of_lists(self, mock_st_class: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        # encode returns numpy arrays
        mock_model.encode.return_value = np.array([
            [0.1] * 768,
            [0.2] * 768,
        ])
        mock_st_class.return_value = mock_model

        embedder = Embedder(model_name="test-model")
        result = embedder.embed(["hello", "world"])

        assert len(result) == 2
        assert len(result[0]) == 768
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)

    @patch("src.embedder.SentenceTransformer")
    def test_embed_applies_document_prefix(self, mock_st_class: MagicMock) -> None:
        # Asymmetric retrieval: every chunk is prefixed with search_document:.
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768, [0.2] * 768])
        mock_st_class.return_value = mock_model

        embedder = Embedder(model_name="test-model")
        embedder.embed(["hello", "world"])

        prefixed = mock_model.encode.call_args.args[0]
        assert prefixed == [
            f"{DEFAULT_DOCUMENT_PREFIX}hello",
            f"{DEFAULT_DOCUMENT_PREFIX}world",
        ]

    @patch("src.embedder.SentenceTransformer")
    def test_embed_passes_batch_size(self, mock_st_class: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.encode.return_value = np.array([[0.1] * 768])
        mock_st_class.return_value = mock_model

        embedder = Embedder(model_name="test-model")
        embedder.embed(["hello"], batch_size=32)

        mock_model.encode.assert_called_once_with(
            [f"{DEFAULT_DOCUMENT_PREFIX}hello"],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
