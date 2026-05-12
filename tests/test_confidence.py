from rag.confidence import (
    crag_gate,
    grounding_confidence,
    retrieval_confidence,
    score_answer,
)
from rag.ingest import Chunk
from rag.retrieve import RetrievedChunk


def _rc(text: str, rerank: float | None = None, dense: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(text=text, source="t.md", chunk_id="x"),
        rerank_score=rerank,
        dense_score=dense,
    )


def test_retrieval_confidence_empty():
    assert retrieval_confidence([]) == 0.0


def test_retrieval_confidence_passes_through_probability():
    # bge-reranker emits scores already in [0, 1] — use them directly.
    assert abs(retrieval_confidence([_rc("foo", rerank=0.5)]) - 0.5) < 1e-9
    assert retrieval_confidence([_rc("foo", rerank=0.98)]) > 0.97


def test_retrieval_confidence_clamps():
    assert retrieval_confidence([_rc("foo", rerank=-0.2)]) == 0.0
    assert retrieval_confidence([_rc("foo", rerank=1.5)]) == 1.0


def test_grounding_full_overlap():
    chunk = _rc("the firm retains client records for seven years after closure")
    # Use a long enough answer for 4-grams to form.
    answer = "the firm retains client records for seven years"
    val = grounding_confidence(answer, [chunk])
    assert val > 0.5


def test_grounding_no_overlap():
    chunk = _rc("cats and dogs make popular pets")
    answer = "the firm retains client records for seven years"
    val = grounding_confidence(answer, [chunk])
    assert val == 0.0


def test_grounding_empty_inputs():
    assert grounding_confidence("", [_rc("x")]) == 0.0
    assert grounding_confidence("nonempty", []) == 0.0


def test_score_capped_at_retrieval():
    # retrieval ≈ 0; grounding high should still produce low final.
    chunks = [_rc("the answer is forty two and we are sure", rerank=0.0)]
    report = score_answer("the answer is forty two and we are sure", chunks)
    assert report.final <= report.retrieval + 1e-9


def test_crag_gate_below_threshold():
    assert crag_gate([_rc("foo", rerank=0.0005)], threshold=0.001) is False


def test_crag_gate_above_threshold():
    assert crag_gate([_rc("foo", rerank=0.005)], threshold=0.001) is True


def test_crag_gate_no_results():
    assert crag_gate([], threshold=0.001) is False


def test_label_thresholds():
    chunks = [_rc("text matching answer here in this passage about retention", rerank=0.95)]
    high = score_answer("text matching answer here in this passage about retention", chunks)
    assert high.label == "High"
    low = score_answer("totally unrelated phrase about cats and dogs", [_rc("foo", rerank=0.02)])
    assert low.label == "Low"
