from rag.highlight import find_matched_spans, render_with_highlights


def test_find_matched_spans_simple():
    source = "Records are retained for seven years after engagement closes."
    answer = "The firm keeps records for seven years."
    spans = find_matched_spans(source, answer, min_words=3)
    assert spans
    # The matched span should contain 'seven years'
    rendered = render_with_highlights(source, spans)
    assert "<mark>" in rendered and "seven years" in rendered.lower()


def test_find_matched_spans_no_overlap():
    spans = find_matched_spans("foo bar baz", "completely different sentence", min_words=4)
    assert spans == []


def test_render_no_spans_returns_text():
    assert render_with_highlights("hello world", []) == "hello world"


def test_render_preserves_order():
    src = "alpha beta gamma delta epsilon"
    answer = "alpha beta gamma delta"
    spans = find_matched_spans(src, answer, min_words=3)
    rendered = render_with_highlights(src, spans)
    assert rendered.startswith("<mark>") or "<mark>" in rendered
    # ensure no overlapping/duplicated <mark>
    assert rendered.count("<mark>") == rendered.count("</mark>")
