"""Retrieval-only evaluation: precision@k, recall@k, grounding accuracy.

Also runs an ablation: dense-only vs hybrid (BM25+dense) vs hybrid+rerank,
so we can quantify what each architectural choice buys us. LLM-free — fast,
deterministic, runs in CI.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import settings  # noqa: E402
from rag.ingest import load_and_chunk  # noqa: E402
from rag.rerank import rerank  # noqa: E402
from rag.retrieve import RetrievedChunk, hybrid_search  # noqa: E402
from rag.store import VectorStore  # noqa: E402

SAMPLE_DIR = ROOT / "data" / "sample_corpus"
EVAL_FILE = ROOT / "eval" / "eval_questions.json"
RESULTS_FILE = ROOT / "eval" / "results.md"


def precision_at_k(retrieved: list[RetrievedChunk], expected_sources: list[str]) -> float:
    if not retrieved or not expected_sources:
        return 0.0
    hits = sum(1 for r in retrieved if r.chunk.source in expected_sources)
    return hits / len(retrieved)


def recall_at_k(retrieved: list[RetrievedChunk], expected_sources: list[str]) -> float:
    if not expected_sources:
        return 0.0
    found = {r.chunk.source for r in retrieved} & set(expected_sources)
    return len(found) / len(set(expected_sources))


def grounding_accuracy(retrieved: list[RetrievedChunk], expected_substrings: list[str]) -> float:
    if not expected_substrings:
        return 0.0
    ctx = " ".join(r.chunk.text for r in retrieved).lower()
    hits = sum(1 for s in expected_substrings if s.lower() in ctx)
    return hits / len(expected_substrings)


def build_store() -> VectorStore:
    chunks = load_and_chunk(sorted(SAMPLE_DIR.glob("*.md")))
    store = VectorStore()
    store.add(chunks)
    return store


def run_strategy(
    store: VectorStore,
    questions: list[dict],
    k: int,
    *,
    fetch: Callable[[str], list[RetrievedChunk]],
    label: str,
) -> dict:
    in_corpus = [q for q in questions if not q["is_off_topic"]]
    off_topic = [q for q in questions if q["is_off_topic"]]

    precisions, recalls, groundings = [], [], []
    for q in in_corpus:
        retrieved = fetch(q["question"])[:k]
        precisions.append(precision_at_k(retrieved, q["expected_source_files"]))
        recalls.append(recall_at_k(retrieved, q["expected_source_files"]))
        groundings.append(grounding_accuracy(retrieved, q["expected_substrings"]))

    # Guardrail: count off-topic questions whose best score falls below threshold.
    guardrail_passes = 0
    for q in off_topic:
        retrieved = fetch(q["question"])[:k]
        top = retrieved[0] if retrieved else None
        score = (top.rerank_score if top and top.rerank_score is not None
                 else (top.dense_score if top else 0.0))
        if not top or score < settings.crag_threshold:
            guardrail_passes += 1

    return {
        "label": label,
        "k": k,
        "precision@k": _mean(precisions),
        "recall@k": _mean(recalls),
        "grounding": _mean(groundings),
        "guardrail": guardrail_passes / max(len(off_topic), 1),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    print("Building index from sample corpus...")
    store = build_store()
    print(f"  -> {len(store)} chunks indexed")

    with EVAL_FILE.open() as f:
        questions = json.load(f)["questions"]
    print(f"Loaded {len(questions)} eval questions "
          f"({sum(1 for q in questions if q['is_off_topic'])} off-topic).")

    k = settings.final_k

    strategies = [
        run_strategy(
            store, questions, k,
            fetch=lambda q: [
                RetrievedChunk(chunk=store.chunks[i], dense_score=s)
                for i, s in store.dense_search(q, settings.retrieve_k)
            ],
            label="Dense only",
        ),
        run_strategy(
            store, questions, k,
            fetch=lambda q: hybrid_search(store, q, settings.retrieve_k),
            label="Hybrid (BM25+dense+RRF)",
        ),
        run_strategy(
            store, questions, k,
            fetch=lambda q: rerank(
                q, hybrid_search(store, q, settings.retrieve_k), top_k=settings.retrieve_k
            ),
            label="Hybrid + Reranker",
        ),
    ]

    # Format and emit.
    lines = [
        "# Retrieval Evaluation",
        "",
        f"_Corpus_: `data/sample_corpus/*.md` · _Questions_: {len(questions)} "
        f"({sum(1 for q in questions if q['is_off_topic'])} off-topic)",
        f"_k_ = {k}, _retrieve\\_k_ = {settings.retrieve_k}",
        "",
        "| Strategy | Precision@k | Recall@k | Grounding | Guardrail |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in strategies:
        lines.append(
            f"| {s['label']} | {s['precision@k']:.2f} | {s['recall@k']:.2f} | "
            f"{s['grounding']:.2f} | {s['guardrail']:.2f} |"
        )

    body = "\n".join(lines) + "\n"
    RESULTS_FILE.write_text(body, encoding="utf-8")
    print()
    print(body)
    print(f"Wrote {RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
