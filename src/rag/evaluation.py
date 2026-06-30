"""Basic RAG evaluation metric placeholders and simple custom metrics."""

from __future__ import annotations

from typing import Any


def retrieval_metrics(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int) -> dict[str, float]:
    top_k = retrieved_chunk_ids[:k]
    relevant = set(relevant_chunk_ids)
    if not relevant:
        return {
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
        }
    hits = [chunk_id for chunk_id in top_k if chunk_id in relevant]
    first_hit_rank = next((index + 1 for index, chunk_id in enumerate(top_k) if chunk_id in relevant), None)
    return {
        "recall_at_k": len(hits) / len(relevant),
        "precision_at_k": len(hits) / max(len(top_k), 1),
        "hit_rate_at_k": 1.0 if hits else 0.0,
        "hit_rate": 1.0 if hits else 0.0,
        "mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
    }


def generation_metric_placeholders() -> dict[str, Any]:
    return {
        "answer_relevance": None,
        "faithfulness": None,
        "answer_relevancy": None,
        "context_relevancy": None,
        "citation_accuracy": None,
        "hallucination_rate": None,
        "note": "Hook RAGAS or judged evaluation here when labeled QA data is available.",
    }
