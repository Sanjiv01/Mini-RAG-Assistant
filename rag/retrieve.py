"""Hybrid retrieval: dense + BM25 fused via Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

from .ingest import Chunk
from .store import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float | None = None


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    *,
    k_rrf: int = 60,
) -> dict[int, float]:
    """Combine multiple rankings of (idx, score) into a single RRF score per idx."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (idx, _score) in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    return fused


def hybrid_search(
    store: VectorStore,
    query: str,
    k: int,
    *,
    k_each: int | None = None,
    k_rrf: int = 60,
) -> list[RetrievedChunk]:
    """Run dense and BM25 in parallel, fuse via RRF, return top-k candidates."""
    k_each = k_each or max(k * 2, k)
    dense = store.dense_search(query, k_each)
    sparse = store.sparse_search(query, k_each)

    fused = reciprocal_rank_fusion([dense, sparse], k_rrf=k_rrf)
    if not fused:
        return []

    dense_map = dict(dense)
    sparse_map = dict(sparse)
    top_ids = sorted(fused, key=fused.get, reverse=True)[:k]

    return [
        RetrievedChunk(
            chunk=store.chunks[i],
            dense_score=dense_map.get(i, 0.0),
            sparse_score=sparse_map.get(i, 0.0),
            rrf_score=fused[i],
        )
        for i in top_ids
    ]
