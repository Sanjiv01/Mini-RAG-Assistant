"""Round-trip tests for store + hybrid retrieval. These hit a real (small)
embedding model, so they exercise the integration end-to-end."""

from pathlib import Path

import pytest

from rag.ingest import chunk_text
from rag.retrieve import hybrid_search, reciprocal_rank_fusion
from rag.store import VectorStore


@pytest.fixture(scope="module")
def store() -> VectorStore:
    docs = {
        "policies.md": (
            "# Data Retention\n\nClient engagement records are retained for seven (7) years."
            "\n\n# PII\n\nPII is purged within 90 days of project closure."
        ),
        "products.md": (
            "# Pricing\n\nThe Team plan costs $1,499 per month.\n\n"
            "# SLA\n\nEnterprise customers receive a 99.95% monthly uptime SLA."
        ),
        "unrelated.md": "Cats are small carnivorous mammals popular as pets.",
    }
    chunks = []
    for name, text in docs.items():
        chunks.extend(chunk_text(text, source=name, chunk_size=200, overlap=20))
    s = VectorStore()
    s.add(chunks)
    return s


def test_store_dedupes(store: VectorStore):
    before = len(store)
    added = store.add(store.chunks)  # same chunk_ids
    assert added == 0
    assert len(store) == before


def test_persistence_roundtrip(tmp_path: Path, store: VectorStore):
    store.save(cache_dir=tmp_path)
    reloaded = VectorStore()
    assert reloaded.load(cache_dir=tmp_path)
    assert len(reloaded) == len(store)
    assert reloaded.chunks[0].text == store.chunks[0].text


def test_dense_finds_paraphrase(store: VectorStore):
    # 'retention' isn't in the corpus chunk for retention; the body says
    # 'retained for seven years'. Dense should still surface it.
    hits = store.dense_search("how long do we keep records?", k=3)
    sources = [store.chunks[i].source for i, _ in hits]
    assert "policies.md" in sources


def test_sparse_finds_exact_term(store: VectorStore):
    hits = store.sparse_search("99.95%", k=3)
    assert hits, "BM25 should find the exact SLA token"
    assert store.chunks[hits[0][0]].source == "products.md"


def test_hybrid_search_returns_results(store: VectorStore):
    results = hybrid_search(store, "what is the Team plan price?", k=3)
    assert len(results) > 0
    assert any(r.chunk.source == "products.md" for r in results)


def test_rrf_combines_rankings():
    a = [(0, 1.0), (1, 0.5)]
    b = [(1, 2.0), (0, 1.0)]
    fused = reciprocal_rank_fusion([a, b])
    # idx 0 ranks 1st then 2nd; idx 1 ranks 2nd then 1st — should be similar but not equal
    assert set(fused.keys()) == {0, 1}
    assert all(v > 0 for v in fused.values())
