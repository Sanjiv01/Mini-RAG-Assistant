"""Sentence-Transformers embedder, lazy singleton on CPU."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # CPU keeps GPU VRAM free for the LLM; bge-small is fast on CPU.
    return SentenceTransformer(settings.embedding_model, device="cpu")


def encode(texts: list[str], *, query: bool = False) -> np.ndarray:
    """Return L2-normalised float32 embeddings (cosine via inner product)."""
    if not texts:
        return np.zeros((0, dim()), dtype="float32")
    # bge models are trained with an instruction prefix for queries.
    if query and settings.embedding_model.lower().startswith("baai/bge"):
        texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
    vecs = _get_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vecs.astype("float32", copy=False)


def dim() -> int:
    m = _get_model()
    # The method was renamed in newer sentence-transformers; support both.
    getter = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
    return getter()
