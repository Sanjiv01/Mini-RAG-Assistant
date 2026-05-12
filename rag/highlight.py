"""Find substrings from retrieved chunks that appear in the answer (or vice versa),
used to highlight grounded spans in the UI."""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _normalize(text: str) -> str:
    return text.lower()


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def find_matched_spans(source_text: str, answer: str, min_words: int = 4) -> list[Span]:
    """Return spans in `source_text` whose ≥`min_words`-word phrases appear in `answer`.

    Greedy longest-first scan over the source; case-insensitive whole-word match.
    Used purely for UI highlighting — no semantic meaning, just visual aid.
    """
    if not source_text or not answer:
        return []
    src_words = list(_WORD_RE.finditer(source_text))
    if len(src_words) < min_words:
        return []
    ans_norm = _normalize(answer)

    spans: list[Span] = []
    i = 0
    while i <= len(src_words) - min_words:
        # Try the longest match starting at i, shrink until we find one or give up.
        for j in range(len(src_words), i + min_words - 1, -1):
            start = src_words[i].start()
            end = src_words[j - 1].end()
            phrase = _normalize(source_text[start:end])
            if phrase in ans_norm:
                spans.append(Span(start, end))
                i = j
                break
        else:
            i += 1
    return spans


def render_with_highlights(text: str, spans: list[Span]) -> str:
    """Wrap each span in <mark>…</mark>. Spans must be non-overlapping and sorted."""
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for s in spans:
        out.append(text[cursor : s.start])
        out.append("<mark>")
        out.append(text[s.start : s.end])
        out.append("</mark>")
        cursor = s.end
    out.append(text[cursor:])
    return "".join(out)
