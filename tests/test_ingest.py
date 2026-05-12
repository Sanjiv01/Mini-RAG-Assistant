from rag.ingest import chunk_text, load_document
from pathlib import Path


def test_chunk_basic_size_bound():
    text = "para one.\n\n" + ("word " * 500) + "\n\nfinal."
    chunks = chunk_text(text, source="t.md", chunk_size=200, overlap=30)
    assert chunks, "should produce at least one chunk"
    # No chunk should exceed chunk_size + overlap (overlap is prepended).
    for c in chunks:
        assert len(c.text) <= 200 + 30


def test_chunk_overlap_is_applied():
    text = "alpha beta gamma delta epsilon zeta eta theta " * 30
    chunks = chunk_text(text, source="t.md", chunk_size=100, overlap=20)
    if len(chunks) >= 2:
        # The tail of chunk i should appear at the head of chunk i+1.
        tail = chunks[0].text[-20:]
        # not strict equality because of stripping; just check substring presence
        assert any(tail.strip() and tail.strip()[:10] in c.text for c in chunks[1:])


def test_chunk_stable_ids():
    text = "Hello world. " * 10
    a = chunk_text(text, source="t.md", chunk_size=50, overlap=10)
    b = chunk_text(text, source="t.md", chunk_size=50, overlap=10)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_chunk_empty_text():
    assert chunk_text("", source="x") == []
    assert chunk_text("   \n\n  ", source="x") == []


def test_load_text_file(tmp_path: Path):
    p = tmp_path / "doc.md"
    p.write_text("# Heading\n\nBody paragraph.", encoding="utf-8")
    text = load_document(p)
    assert "Heading" in text and "Body paragraph" in text


def test_unsupported_suffix(tmp_path: Path):
    p = tmp_path / "doc.docx"
    p.write_bytes(b"PK\x03\x04")
    try:
        load_document(p)
    except ValueError as e:
        assert "Unsupported" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_section_attribution():
    text = (
        "# Top\n\nintro paragraph.\n\n"
        "## Retention\n\nrecords kept for seven years."
    )
    chunks = chunk_text(text, source="p.md", chunk_size=60, overlap=10)
    # At least one chunk should carry the 'Retention' heading.
    sections = {c.section for c in chunks}
    assert any("Retention" in s for s in sections)
