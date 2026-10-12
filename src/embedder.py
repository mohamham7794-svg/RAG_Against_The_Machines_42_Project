"""Semantic embeddings (bonus): a small CPU sentence-embedding model.

The model is only needed for the optional ``index --semantic`` and
``--mode hybrid`` paths, so ``sentence_transformers`` is imported lazily:
the default lexical pipeline never touches it.
"""

from __future__ import annotations

from typing import Any, Iterable, List

import numpy as np
from tqdm import tqdm

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256
ENCODE_SLICE = 256


class Embedder:
    """Turns text into L2-normalised vectors (dot product == cosine).

    Attributes:
        model_name: Sentence-transformers model identifier.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Remember the model name; the weights are loaded on first use.

        Args:
            model_name: Sentence-transformers model identifier.
        """
        self.model_name = model_name
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        """Load the model on first use.

        Raises:
            RuntimeError: If sentence-transformers or the model weights
                are not available.
        """
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for semantic search. "
                "Add it to the project dependencies and run `uv sync`."
            ) from exc
        try:
            model = SentenceTransformer(self.model_name, device="cpu")
        except Exception as exc:  # noqa: BLE001 - offline, bad name, ...
            raise RuntimeError(
                f"Could not load embedding model {self.model_name}: {exc}"
            ) from exc
        model.max_seq_length = MAX_SEQ_LENGTH
        self._model = model

    def encode(
        self, texts: List[str], show_progress: bool = False
    ) -> np.ndarray:
        """Embed ``texts``.

        Args:
            texts: The texts to embed.
            show_progress: Display a tqdm progress bar.

        Returns:
            A float32 array of shape ``(len(texts), dim)``.
        """
        self._ensure_loaded()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        parts: List[np.ndarray] = []
        starts: List[int] = list(range(0, len(texts), ENCODE_SLICE))
        batches: Iterable[int] = starts
        if show_progress:
            batches = tqdm(starts, desc="Embedding", unit="batch")
        for start in batches:
            batch = texts[start:start + ENCODE_SLICE]
            vectors = self._model.encode(
                batch,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            parts.append(np.asarray(vectors, dtype=np.float32))
        return np.vstack(parts)
