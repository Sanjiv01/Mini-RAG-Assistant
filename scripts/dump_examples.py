"""Run the RAG over the eval questions and dump answers + citations + scores
to examples/sample_responses.json so reviewers can see expected behaviour
without launching the app. Loads the configured LLM (slow first run).

Usage:
    python scripts/dump_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.pipeline import build_rag  # noqa: E402

SAMPLE_DIR = ROOT / "data" / "sample_corpus"
EVAL_FILE = ROOT / "eval" / "eval_questions.json"
OUT_FILE = ROOT / "examples" / "sample_responses.json"


def main() -> None:
    rag = build_rag()
    rag.ingest_paths(
        sorted(SAMPLE_DIR.glob("*.md")),
        contextual=False,  # keep snapshot fast + deterministic
    )

    with EVAL_FILE.open() as f:
        questions = json.load(f)["questions"]

    out = {"backend": rag.llm.name, "responses": []}
    for q in questions:
        ans = rag.ask(q["question"])
        out["responses"].append({
            "id": q["id"],
            "question": q["question"],
            "is_off_topic": q["is_off_topic"],
            "answer": ans.text,
            "refused": ans.refused,
            "confidence": {
                "retrieval": round(ans.confidence.retrieval, 3),
                "grounding": round(ans.confidence.grounding, 3),
                "final": round(ans.confidence.final, 3),
                "label": ans.confidence.label,
            },
            "sources": [
                {
                    "source": r.chunk.source,
                    "section": r.chunk.section,
                    "rerank_score": round(r.rerank_score, 3) if r.rerank_score is not None else None,
                    "snippet": r.chunk.text[:200] + ("..." if len(r.chunk.text) > 200 else ""),
                }
                for r in ans.retrieved
            ],
        })

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT_FILE.relative_to(ROOT)} ({len(out['responses'])} responses, backend={rag.llm.name})")


if __name__ == "__main__":
    main()
