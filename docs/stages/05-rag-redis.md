# Stage 5: RAG And Redis Cache

## Tech Stack

- PubMedBERT embeddings
- PostgreSQL pgvector
- PostgreSQL keyword search
- hybrid retrieval
- Gemini answer generation
- Redis exact cache
- Redis semantic cache
- MLflow/Dagshub metrics hooks

## Key Steps

1. Added RAG document ingestion.
2. Embedded chunks with local PubMedBERT embeddings.
3. Stored embeddings in PostgreSQL pgvector.
4. Added semantic search and keyword search.
5. Added hybrid retrieval.
6. Added grounded answer generation.
7. Added Redis exact cache.
8. Added Redis semantic cache.
9. Added RAG metrics and reports.
10. Added selected-document and all-document retrieval modes.

## What Was Implemented

RAG lets users ask questions over uploaded and trusted TMF documents. It is separate from classification:

- classifier predicts TMF document class
- RAG retrieves relevant chunks and generates answers

The answer generator is expected to use retrieved chunks and citations rather than inventing answers.

## Retrieval Scope

Supported retrieval modes include:

- selected document
- master/trusted data
- verified documents
- all authorized documents
- class-scoped retrieval

## Redis Metrics

| Metric | Value |
| --- | ---: |
| Average uncached latency | 4,816.40 ms |
| Average cached latency | 779.75 ms |
| Exact cache speedup | 7.05x |
| Semantic cache speedup | 5.61x |
| Overall latency reduction | 83.81% |
| Total cache time saved | 24,222 ms |

## Files

| Path | Purpose |
| --- | --- |
| `src/rag/vector_store.py` | pgvector storage/search |
| `src/rag/service.py` | RAG orchestration |
| `src/rag/retriever.py` | Hybrid retrieval |
| `src/rag/semantic_cache.py` | Redis exact/semantic cache |
| `src/rag/evaluation_report.py` | RAG metrics report generation |
| `scripts/index_master_data.py` | Trusted data indexing |
| `scripts/reset_rag_vector_store.py` | Reset vector store |
| `reports/rag_metrics_summary.txt` | RAG/Redis metrics |

## Limitations

- RAG evaluation set is small.
- Semantic cache threshold tuning can improve hit rate.
- Production RAG needs stronger hallucination and source coverage evaluation.
