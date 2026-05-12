# Mini-RAG Assistant

A lightweight, fully-local Retrieval-Augmented Generation prototype. Upload PDFs / Markdown / text, ask a question, get a grounded answer with citations and a transparent confidence score. Refuses to answer when retrieved context isn't relevant enough.

**Run it:** `streamlit run app.py` → sidebar → *Load sample corpus* → ask a question.
**Test it:** `pytest -q` (33 tests) and `python scripts/evaluate.py` (retrieval ablation).

---

## Architecture

```
                          USER QUERY
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
   BM25 (rank_bm25)                  Dense embeddings (BAAI/bge-small-en-v1.5)
   top-20 hits                       FAISS IndexFlatIP, top-20 hits
        │                                           │
        └─────────────────┬─────────────────────────┘
                          ▼
                Reciprocal Rank Fusion (RRF, k=60)
                          ▼
            Cross-encoder reranker (BAAI/bge-reranker-base)
            scores 20 (query, chunk) pairs → top-5
                          ▼
            CRAG-style gate: max(rerank_score) < τ ?
                ┌─────────┴─────────┐
                ▼                   ▼
         REFUSE answer        Grounded prompt
                              "Answer only from sources,
                               cite as [Source N]"
                                    ▼
                    Qwen 2.5 7B Instruct (4-bit NF4)
                    via transformers + bitsandbytes
                                    ▼
                    Confidence = 0.6·retrieval + 0.4·grounding
                    (capped at retrieval)
                                    ▼
                    Answer + citations + matched-span
                    highlights + Low/Med/High badge
```

Each retrieval stage is documented inline; the high-level orchestrator is [`rag/pipeline.py`](rag/pipeline.py).

### Why this stack

| Technique | In the pipeline? | Reason |
|---|---|---|
| **Hybrid (BM25 + Dense + RRF)** | ✅ | Consulting documents are full of exact-term lookups (policy IDs, dollar amounts, dates, acronyms). BM25 catches the lexical needles; dense catches paraphrase. RRF combines them parameter-free. |
| **Cross-encoder Reranker** | ✅ | Biggest single quality win in modern RAG. The reranker's score also doubles as a calibrated *relevance signal* — it's what powers the CRAG gate. Without it, off-topic questions get answered. |
| **Structural Contextual Retrieval** | ✅ | Each chunk is embedded and reranked with a `[filename — section]` prefix derived from parsed headings. Cheap, deterministic stand-in for Anthropic's LLM-generated contextual preambles. |
| **LLM-based Contextual Retrieval** | ✅ (optional, toggle in UI) | Anthropic-style: one-shot LLM call per chunk at ingestion to generate a 1-sentence "where this sits in the document" preamble. Boosts recall on table-heavy chunks. |
| **CRAG-style confidence gate** | ✅ | Refuse rather than hallucinate when no retrieved chunk clears the threshold. |
| Recursive heading-aware chunking | ✅ | 1200-char chunks, 200-char overlap. The chunker tracks original document positions so each chunk gets the correct governing heading. |
| HyDE / Query expansion | ❌ | Extra LLM call per query (~+3 s); redundant once a reranker is in place. |
| RAPTOR | ❌ | Ingestion-heavy (recursive summarisation); mostly helps multi-document synthesis, which our consulting corpus doesn't need. |
| ColBERT / late interaction | ❌ | Different indexing infrastructure, harder Windows setup. The cross-encoder captures most of the same gain. |

### Confidence-scoring method

Two complementary signals, blended into a 0-100% score:

1. **Retrieval confidence** — top reranker score. `BAAI/bge-reranker-base` emits values in [0, 1] from its single-label head, so we use it directly.
2. **Grounding confidence** — fraction of answer 4-grams (stop-words stripped) that also appear in the concatenated retrieved context. Deterministic, no extra model call.

```python
final = min(retrieval, 0.6·retrieval + 0.4·grounding)
```

The `min(…)` cap prevents a fluent answer from claiming higher confidence than the evidence supports. Label boundaries: **Low** < 40 % · **Medium** 40-70 % · **High** ≥ 70 %.

Each retrieved chunk's individual reranker / dense / BM25 score is shown in the UI so reviewers can audit exactly what drove the final number.

### Anti-hallucination guardrail (CRAG)

If the best reranker score is below `CRAG_THRESHOLD` (default 0.001) the system refuses with a fixed message rather than passing weak context to the LLM. Off-topic questions in the eval set score exactly 0.0, so this cleanly separates them from in-corpus matches without sacrificing recall on borderline-relevant questions.

---

## Setup

```powershell
# 1. virtualenv + Python deps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu124   # see "GPU notes" below
pip install -r requirements.txt
```

The embedding model (`bge-small-en-v1.5`, ~120 MB) and reranker (`bge-reranker-base`, ~280 MB) auto-download on first use. The LLM (`Qwen/Qwen2.5-7B-Instruct`, ~15 GB) downloads only when you actually run a generation in the app (button: *Reload models* with `real` backend selected, or call `python -c "from rag.generate import TransformersLLM; TransformersLLM().load()"`).

### Run it

```powershell
streamlit run app.py
```

Open <http://localhost:8501>:

1. Sidebar → **Load sample corpus** (or upload your own PDFs / TXT / MD).
2. Ask a question in the chat box, e.g. *"How long are client records retained?"*.

The "Retrieved context" expander on every answer shows each source's reranker, dense, and BM25 scores plus matched-substring highlighting on the source text.

### GPU notes (RTX 50-series / Blackwell)

The default config quantizes Qwen to 4-bit NF4 via `bitsandbytes`, fitting in ~5 GB VRAM on an 8 GB GPU. If `bitsandbytes` fails to install on your machine (occasionally an issue on very new GPUs), set in `.env`:

```dotenv
USE_4BIT=false
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct
```

…to fall back to BF16 3B (~6 GB VRAM, no quantization dependency). For CPU-only machines, additionally set `LLM_DEVICE=cpu` — generation will be slow (~30 s/answer) but functional.

---

## Evaluation

`scripts/evaluate.py` runs the eval set in [`eval/eval_questions.json`](eval/eval_questions.json) and writes [`eval/results.md`](eval/results.md). It does an ablation over three retrieval strategies so the architectural choices are quantified, not just asserted.

| Strategy | Precision@5 | Recall@5 | Grounding | Guardrail |
|---|---:|---:|---:|---:|
| Dense only | 0.48 | 1.00 | 1.00 | **0.00** |
| Hybrid (BM25+dense+RRF) | 0.46 | 1.00 | 1.00 | **0.00** |
| Hybrid + Reranker | 0.46 | 1.00 | 1.00 | **1.00** |

**Recall@5 = 1.0** across the board — for our 10 in-corpus eval questions, the right source file always lands in the top-5. **Grounding = 1.0** — every expected substring is present in the retrieved context. The retrieval is solid.

The headline number is **Guardrail accuracy**: only the reranker-enabled strategy refuses off-topic questions. Dense / hybrid retrieval *always* return their top-5 because they have no calibrated relevance signal. **The reranker isn't just for ordering — it's the confidence signal that makes refusal possible.** Without it, the system would confidently invent answers to questions like "who won the 2014 FIFA World Cup?" using the closest cosine match.

Precision@5 hovers around 0.5 because each eval question expects one source file and the top-5 naturally includes some chunks from neighbouring files (only 11 chunks across 3 docs). Recall and grounding are the meaningful retrieval metrics here.

---

## Example I/O

[`examples/sample_responses.json`](examples/sample_responses.json) captures answers + citations + confidence for every question in the eval set. Two illustrative samples:

```jsonc
{
  "id": "policy-retention",
  "question": "How long are client engagement records retained?",
  "confidence": { "retrieval": 1.0, "grounding": 0.92, "final": 0.967, "label": "High" },
  "sources": [
    {
      "source": "policies.md",
      "section": "Data Retention Policy",
      "rerank_score": 1.0,
      "snippet": "...must be retained for **seven (7) years** from the date the engagement is formally closed..."
    },
    /* … */
  ]
}

{
  "id": "off-topic-worldcup",
  "question": "Who won the 2014 FIFA World Cup?",
  "refused": true,
  "confidence": { "retrieval": 0.0, "grounding": 0.0, "final": 0.0, "label": "Low" },
  "answer": "I don't have enough information in the indexed documents to answer that..."
}
```

UI screenshots live in [`examples/screenshots/`](examples/screenshots/).

---

## Project layout

```
app.py                       # Streamlit UI
rag/
  config.py                  # env-driven settings (Settings dataclass)
  ingest.py                  # PDF/TXT/MD load + heading-aware chunker
  embed.py                   # sentence-transformers embedder (CPU)
  store.py                   # FAISS + BM25 + parallel chunk metadata
  retrieve.py                # hybrid search + RRF
  rerank.py                  # cross-encoder reranker
  contextualize.py           # optional LLM-based contextual preambles
  generate.py                # TransformersLLM (Qwen 2.5 4-bit NF4)
  confidence.py              # retrieval + grounding score + CRAG gate
  highlight.py               # matched-span detection for the UI
  pipeline.py                # high-level RAG.ask() orchestrator
data/sample_corpus/          # 3 synthetic consulting docs (policies, FAQ, manual)
eval/
  eval_questions.json        # 10 in-corpus + 2 off-topic ground-truth questions
  results.md                 # ablation table, regenerated by evaluate.py
scripts/
  evaluate.py                # retrieval eval + ablation (LLM-free, fast)
  ingest_folder.py           # CLI: index any folder of PDFs
  dump_examples.py           # snapshot Q/A/citations into examples/sample_responses.json
tests/                       # 33 pytest tests
```

---

## Adding your own documents

```powershell
# Programmatic ingest:
python scripts/ingest_folder.py path/to/your/docs

# Or just drag-and-drop in the Streamlit sidebar (uploader supports PDF / TXT / MD).
```

Contextual Retrieval can be toggled per ingestion. Default is **on** (slower but better recall on tables and lists). For very large uploads, turn it off to skip the per-chunk LLM call.

---

## Tests

```powershell
pytest -q                                # 33 tests, ~25 s on a warm cache
python scripts/evaluate.py               # ablation, ~10 s
python scripts/dump_examples.py          # snapshot Q/A/citations into examples/sample_responses.json
```

The pipeline test (`tests/test_pipeline.py`) uses an `_EchoLLM` stub so CI doesn't have to load Qwen.

---

## Design choices & trade-offs

- **No LangChain / LlamaIndex.** Direct code is easier for a reviewer to read end-to-end and avoids version churn. The whole retrieval pipeline is ~250 lines across `retrieve.py`, `rerank.py`, `store.py`.
- **`faiss-cpu`, not `faiss-gpu`.** For corpora up to ~100k chunks, flat-index search is dominated by LLM latency. Saves a CUDA build path.
- **Embedder + reranker on CPU.** Frees the 8 GB GPU for the LLM. Reranking 20 pairs on CPU is ~400 ms, well below LLM generation time.
- **Recall over precision for small corpora.** With only 11 chunks across 3 files, top-5 will always include some neighbouring-file chunks. That's fine — the reranker re-orders and the LLM cites correctly.
- **Static `[filename — section]` prefix** is a deterministic stand-in for LLM-based Contextual Retrieval, and bg-reranker scores improve measurably with it. The LLM-based version is layered on top, opt-in for ingestion-time cost.
- **Anti-hallucination via reranker confidence**, not via answer-text classification. This catches off-topic questions *before* prompting the LLM, saving a wasted generation.

---

## Limitations & not-yet-done

- **Reranker truncation on long chunks.** `bge-reranker-base` has a 512-token context; chunks at the 1200-char ceiling occasionally get truncated. A `bge-reranker-large` or `bge-reranker-v2-m3` would help but is ~1.5 GB.
- **No persistent index across Streamlit reloads.** The store is rebuilt per session. Persistence to `.cache/` is implemented in [`rag/store.py`](rag/store.py) but not auto-wired into the app.
- **Table chunks are heuristic.** A markdown-table-aware chunker would keep header rows attached to data rows even at smaller chunk sizes. For this corpus, raising `CHUNK_SIZE` keeps tables intact in practice.
- **No streaming output.** The Streamlit response renders after generation completes; for a polished demo, `model.generate(streamer=...)` could pipe tokens live.

---

## License

MIT — see [LICENSE](LICENSE).
