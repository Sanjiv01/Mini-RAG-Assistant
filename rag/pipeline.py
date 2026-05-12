"""High-level orchestrator: ask(query, history) → grounded answer + citations + score."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import settings
from .confidence import ConfidenceReport, crag_gate, score_answer
from .contextualize import contextualize_chunks
from .generate import LLM, make_llm
from .ingest import Chunk, chunk_text, load_and_chunk
from .url_loader import load_url_as_text
from .rerank import rerank
from .retrieve import RetrievedChunk, hybrid_search
from .store import VectorStore


REFUSAL = (
    "I don't have enough information in the indexed documents to answer that. "
    "Try uploading a relevant document or rephrasing the question."
)

_SYSTEM = (
    "You are a precise assistant that answers questions strictly from the provided "
    "numbered sources. Cite each fact inline as [Source N]. If the answer is not "
    "in the sources, reply exactly: \"I don't have enough information in the "
    "indexed documents to answer that.\""
)


@dataclass
class Answer:
    text: str
    retrieved: list[RetrievedChunk]
    confidence: ConfidenceReport
    refused: bool = False
    debug: dict = field(default_factory=dict)


class RAG:
    """Stateful RAG session: an index, an LLM, and an ask() method."""

    def __init__(self, llm: LLM) -> None:
        self.store = VectorStore()
        self.llm: LLM = llm

    # --- ingestion --------------------------------------------------------

    def ingest_paths(
        self,
        paths: list[Path],
        *,
        contextual: bool | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> int:
        chunks = load_and_chunk(paths)
        return self.ingest_chunks(chunks, contextual=contextual, progress=progress)

    def ingest_url(
        self,
        url: str,
        *,
        contextual: bool | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> tuple[int, str]:
        """Fetch a URL (PDF or HTML), chunk it, add to the store.

        Returns (chunks_added, display_name). Raises RuntimeError on
        network failure or unsupported content.
        """
        text, name = load_url_as_text(url)
        chunks = chunk_text(text, source=f"url:{name}")
        added = self.ingest_chunks(chunks, contextual=contextual, progress=progress)
        return added, name

    def ingest_chunks(
        self,
        chunks: list[Chunk],
        *,
        contextual: bool | None = None,
        progress: Callable[[float], None] | None = None,
    ) -> int:
        if not chunks:
            return 0
        use_ctx = settings.contextual_retrieval if contextual is None else contextual
        if use_ctx:
            chunks = contextualize_chunks(chunks, self.llm, progress=progress)
        return self.store.add(chunks)

    def clear(self) -> None:
        self.store.clear()

    # --- query ------------------------------------------------------------

    def ask(self, query: str, history: list[dict] | None = None) -> Answer:
        query = query.strip()
        if not query:
            return Answer(text="", retrieved=[], confidence=score_answer("", []), refused=True)
        if len(self.store) == 0:
            return Answer(
                text="No documents have been indexed yet. Upload a file or load the sample corpus.",
                retrieved=[],
                confidence=score_answer("", []),
                refused=True,
            )

        candidates = hybrid_search(self.store, query, k=settings.retrieve_k)
        top = rerank(query, candidates, top_k=settings.final_k)

        if not crag_gate(top, threshold=settings.crag_threshold):
            return Answer(
                text=REFUSAL,
                retrieved=top,
                confidence=score_answer("", top),
                refused=True,
                debug={"reason": "crag_gate_failed"},
            )

        prompt = self._build_user_prompt(query, top)
        messages = self._with_history(history, prompt)
        try:
            answer_text = self.llm.complete(_SYSTEM, messages).strip()
        except Exception as exc:
            answer_text = f"(LLM generation failed: {exc})"

        if not answer_text:
            answer_text = REFUSAL
            refused = True
        else:
            refused = REFUSAL.split(".")[0].lower() in answer_text.lower()

        return Answer(
            text=answer_text,
            retrieved=top,
            confidence=score_answer(answer_text, top),
            refused=refused,
        )

    # --- internals --------------------------------------------------------

    @staticmethod
    def _build_user_prompt(query: str, retrieved: list[RetrievedChunk]) -> str:
        blocks = []
        for i, r in enumerate(retrieved, start=1):
            head = f"[Source {i}] ({r.chunk.source}"
            if r.chunk.section:
                head += f" — {r.chunk.section}"
            head += ")"
            blocks.append(f"{head}\n{r.chunk.text}")
        context = "\n\n".join(blocks)
        return f"{context}\n\n---\nQuestion: {query}"

    @staticmethod
    def _with_history(history: list[dict] | None, new_user: str) -> list[dict]:
        msgs: list[dict] = []
        if history:
            # Keep the last 4 turns (8 messages) to bound context.
            msgs.extend(history[-8:])
        msgs.append({"role": "user", "content": new_user})
        return msgs


def build_rag() -> RAG:
    """Construct a RAG with the configured Transformers LLM (loaded eagerly)."""
    return RAG(llm=make_llm())
