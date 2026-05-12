"""End-to-end pipeline with a stub LLM so we don't load Qwen in CI."""

from pathlib import Path

import pytest

from rag.generate import LLM
from rag.pipeline import RAG, REFUSAL

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_corpus"


class _EchoLLM:
    """Stub: always echoes the first retrieved chunk so we can test grounding."""

    name = "stub"

    def complete(self, system: str, messages: list[dict]) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "[Source 1]" in user:
            after = user.split("[Source 1]", 1)[1]
            # Take the first 200 chars after the marker as a faux answer.
            cleaned = after.split("\n", 1)[1] if "\n" in after else after
            return cleaned[:200].strip()
        return "no answer"


@pytest.fixture(scope="module")
def rag() -> RAG:
    r = RAG(llm=_EchoLLM())
    r.ingest_paths(sorted(SAMPLE_DIR.glob("*.md")), contextual=False)
    return r


def test_ingestion_populates_store(rag: RAG):
    assert len(rag.store) > 0
    assert rag.store.stats()["sources"]  # at least one source indexed


def test_in_corpus_question_returns_grounded_answer(rag: RAG):
    ans = rag.ask("How long are client engagement records retained?")
    assert not ans.refused
    assert ans.retrieved
    assert any(r.chunk.source == "policies.md" for r in ans.retrieved)
    # The stub echoes the source chunk, so grounding should be near 1.0
    assert ans.confidence.grounding > 0.5


def test_off_topic_question_refuses(rag: RAG):
    ans = rag.ask("Who won the 2014 FIFA World Cup?")
    # Either the CRAG gate refused, or the answer is the explicit refusal text.
    assert ans.refused or REFUSAL in ans.text


def test_empty_query_refuses(rag: RAG):
    ans = rag.ask("   ")
    assert ans.refused


def test_no_documents_refuses():
    r = RAG(llm=_EchoLLM())
    ans = r.ask("anything")
    assert ans.refused
