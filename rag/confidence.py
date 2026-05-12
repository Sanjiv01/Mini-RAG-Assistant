"""Confidence scoring + CRAG-style gate.

Two signals are blended into a 0-100% score:
  1. Retrieval confidence  — sigmoid of the top reranker logit.
  2. Grounding confidence  — fraction of answer 4-grams (stop-words removed)
                              that also appear in the concatenated retrieved
                              context. Deterministic, no extra model call.
Final = 0.6 · retrieval + 0.4 · grounding, capped at retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieve import RetrievedChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have he her him his i in is it its of on or
    she that the their them they this to was we were will with you your""".split()
)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


def _ngrams(tokens: list[str], n: int = 4) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def retrieval_confidence(retrieved: list[RetrievedChunk]) -> float:
    """Top reranker probability. bge-reranker emits values already in [0, 1].

    Falls back to dense cosine if no reranker score is present.
    """
    if not retrieved:
        return 0.0
    top = retrieved[0]
    if top.rerank_score is not None:
        return max(0.0, min(1.0, top.rerank_score))
    # No reranker: dense cosine sim is in [-1, 1]; map to [0, 1].
    return max(0.0, min(1.0, (top.dense_score + 1.0) / 2.0))


def grounding_confidence(answer: str, retrieved: list[RetrievedChunk]) -> float:
    if not answer.strip() or not retrieved:
        return 0.0
    ans_tokens = _tokens(answer)
    if not ans_tokens:
        return 0.0
    ctx_tokens = _tokens(" ".join(c.chunk.text for c in retrieved))
    ans_grams = _ngrams(ans_tokens)
    ctx_grams = _ngrams(ctx_tokens)
    if not ans_grams:
        return 0.0
    overlap = len(ans_grams & ctx_grams) / len(ans_grams)
    return overlap


@dataclass(frozen=True)
class ConfidenceReport:
    retrieval: float
    grounding: float
    final: float
    label: str  # "High" | "Medium" | "Low"


def _label(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def score_answer(answer: str, retrieved: list[RetrievedChunk]) -> ConfidenceReport:
    r = retrieval_confidence(retrieved)
    g = grounding_confidence(answer, retrieved)
    final = min(r, 0.6 * r + 0.4 * g)
    return ConfidenceReport(retrieval=r, grounding=g, final=final, label=_label(final))


def crag_gate(retrieved: list[RetrievedChunk], threshold: float) -> bool:
    """True ⇒ the top retrieved chunk is good enough to proceed."""
    if not retrieved:
        return False
    top_score = retrieved[0].rerank_score
    if top_score is None:
        return retrieval_confidence(retrieved) > 0.0
    return top_score >= threshold
