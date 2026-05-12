# Retrieval Evaluation

_Corpus_: `data/sample_corpus/*.md` · _Questions_: 12 (2 off-topic)
_k_ = 5, _retrieve\_k_ = 20

| Strategy | Precision@k | Recall@k | Grounding | Guardrail |
|---|---:|---:|---:|---:|
| Dense only | 0.48 | 1.00 | 1.00 | 0.00 |
| Hybrid (BM25+dense+RRF) | 0.46 | 1.00 | 1.00 | 0.00 |
| Hybrid + Reranker | 0.46 | 1.00 | 1.00 | 1.00 |
