"""Cross-encoder reranker. Largest single quality win in the pipeline."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from sentence_transformers import CrossEncoder

from .config import settings
from .retrieve import RetrievedChunk


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model, device="cpu")


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score (query, chunk) pairs with a cross-encoder, return top-k by score."""
    if not candidates:
        return []
    # Use embed_text so the reranker sees [source — section] context; this is a
    # cheap, deterministic stand-in for LLM-based Contextual Retrieval.
    pairs = [(query, c.chunk.embed_text) for c in candidates]
    scores = _get_reranker().predict(pairs, show_progress_bar=False)
    scored = [replace(c, rerank_score=float(s)) for c, s in zip(candidates, scores)]
    scored.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)
    return scored[:top_k]
