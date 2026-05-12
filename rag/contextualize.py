"""Anthropic-style Contextual Retrieval: prepend a short LLM-generated locator
to each chunk *before* embedding.

The locator names the document, section, and key concepts the chunk covers,
which lets a query like 'data retention?' match a chunk whose body uses
'records shall be destroyed after 7 years' without ever using the word retention.

This runs once at ingestion and is cheap with a local model. Defaults ON for
small corpora, off-able from the UI for large uploads.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from .generate import LLM
from .ingest import Chunk

log = logging.getLogger(__name__)

_SYSTEM = (
    "You write a one-sentence preamble that explains where a passage sits inside a "
    "document so that it can be retrieved by semantic search. Be specific about the "
    "topic, section, and key terms. Output the preamble only — no preface, no quotes."
)

_USER_TEMPLATE = """<document filename="{source}">
{document}
</document>

Here is a passage from that document:

<passage>
{passage}
</passage>

Write a single-sentence preamble (≤ 35 words) that situates this passage within the
document. Mention the document topic, the relevant section if obvious, and the key
terms the passage covers. Output the sentence only."""


def contextualize_chunks(chunks: list[Chunk], llm: LLM, *, progress=None) -> list[Chunk]:
    """Return a new list of chunks each with a `context` preamble populated."""
    if not chunks:
        return chunks

    # Group chunks by source so each gets shown the same parent document.
    by_source: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c)

    # Reconstruct a (possibly truncated) parent document per source for the prompt.
    parents = {src: _stitch(cs) for src, cs in by_source.items()}

    out: list[Chunk] = []
    total = len(chunks)
    for i, c in enumerate(chunks):
        parent = parents[c.source]
        try:
            preamble = llm.complete(
                _SYSTEM,
                [{"role": "user", "content": _USER_TEMPLATE.format(
                    source=c.source, document=parent[:6000], passage=c.text
                )}],
            ).splitlines()[0].strip()
        except Exception as exc:  # never fail ingestion on a single chunk
            log.warning("contextualize failed for %s: %s", c.source, exc)
            preamble = ""
        out.append(replace(c, context=preamble))
        if progress is not None:
            progress((i + 1) / total)
    return out


def _stitch(chunks: list[Chunk]) -> str:
    """Reassemble approximate parent text from chunks (overlap-tolerant)."""
    return "\n\n".join(c.text for c in chunks)
