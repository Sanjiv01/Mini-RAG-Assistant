"""FAISS (dense) + BM25 (sparse) + parallel chunk metadata, persisted to disk."""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .config import settings
from .embed import dim, encode
from .ingest import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class VectorStore:
    """Combined dense + sparse index over `Chunk` objects.

    FAISS `IndexFlatIP` on L2-normalised vectors gives cosine similarity. We don't
    use IVF/HNSW because corpora here are small (≤ tens of thousands of chunks)
    and flat search latency is negligible compared to LLM generation.
    """

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []
        self._seen_ids: set[str] = set()

    # --- mutation ----------------------------------------------------------

    def add(self, chunks: list[Chunk]) -> int:
        """Add new chunks, skipping duplicates by `chunk_id`. Returns count added."""
        new = [c for c in chunks if c.chunk_id not in self._seen_ids]
        if not new:
            return 0

        vecs = encode([c.embed_text for c in new], query=False)
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim())
        self.index.add(vecs)

        self.chunks.extend(new)
        self._seen_ids.update(c.chunk_id for c in new)
        self._tokenized.extend(_tokenize(c.embed_text) for c in new)
        self.bm25 = BM25Okapi(self._tokenized)
        return len(new)

    def clear(self) -> None:
        self.chunks.clear()
        self.index = None
        self.bm25 = None
        self._tokenized.clear()
        self._seen_ids.clear()

    # --- query -------------------------------------------------------------

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = encode([query], query=True)
        k = min(k, self.index.ntotal)
        scores, idxs = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.bm25 or not self._tokenized:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        if scores.size == 0:
            return []
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    # --- persistence -------------------------------------------------------

    def save(self, cache_dir: Path | None = None) -> None:
        cache_dir = cache_dir or settings.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(cache_dir / "faiss.index"))
        with (cache_dir / "chunks.json").open("w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)
        with (cache_dir / "bm25.pkl").open("wb") as f:
            pickle.dump(self._tokenized, f)

    def load(self, cache_dir: Path | None = None) -> bool:
        cache_dir = cache_dir or settings.cache_dir
        idx_path = cache_dir / "faiss.index"
        meta_path = cache_dir / "chunks.json"
        bm_path = cache_dir / "bm25.pkl"
        if not (idx_path.exists() and meta_path.exists() and bm_path.exists()):
            return False
        self.index = faiss.read_index(str(idx_path))
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = [Chunk(**c) for c in data]
        self._seen_ids = {c.chunk_id for c in self.chunks}
        with bm_path.open("rb") as f:
            self._tokenized = pickle.load(f)
        self.bm25 = BM25Okapi(self._tokenized) if self._tokenized else None
        return True

    # --- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self.chunks)

    def stats(self) -> dict:
        sources: dict[str, int] = {}
        for c in self.chunks:
            sources[c.source] = sources.get(c.source, 0) + 1
        return {"chunks": len(self.chunks), "sources": sources}
