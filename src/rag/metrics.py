"""Basic in-process RAG metrics and MLflow logging helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.mlflow_tracking import log_metrics, log_params, start_mlflow_run, end_mlflow_run


@dataclass
class RAGMetricsTracker:
    total_questions: int = 0
    cache_hits: int = 0
    semantic_cache_hits: int = 0
    exact_cache_hits: int = 0
    llm_calls_saved: int = 0
    latencies_ms: list[int] = field(default_factory=list)
    retrieval_latencies_ms: list[int] = field(default_factory=list)
    generation_latencies_ms: list[int] = field(default_factory=list)
    reranking_latencies_ms: list[int] = field(default_factory=list)

    def record(
        self,
        latency_ms: int,
        cache_type: str,
        retrieval_latency_ms: int | None = None,
        generation_latency_ms: int | None = None,
        reranking_latency_ms: int | None = None,
    ) -> None:
        self.total_questions += 1
        self.latencies_ms.append(latency_ms)
        if retrieval_latency_ms is not None:
            self.retrieval_latencies_ms.append(retrieval_latency_ms)
        if generation_latency_ms is not None:
            self.generation_latencies_ms.append(generation_latency_ms)
        if reranking_latency_ms is not None:
            self.reranking_latencies_ms.append(reranking_latency_ms)
        if cache_type != "none":
            self.cache_hits += 1
            self.llm_calls_saved += 1
        if cache_type == "semantic":
            self.semantic_cache_hits += 1
        if cache_type == "exact":
            self.exact_cache_hits += 1

    def snapshot(self) -> dict[str, Any]:
        avg_latency = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0
        avg_retrieval_latency = (
            sum(self.retrieval_latencies_ms) / len(self.retrieval_latencies_ms)
            if self.retrieval_latencies_ms
            else 0.0
        )
        avg_generation_latency = (
            sum(self.generation_latencies_ms) / len(self.generation_latencies_ms)
            if self.generation_latencies_ms
            else 0.0
        )
        avg_reranking_latency = (
            sum(self.reranking_latencies_ms) / len(self.reranking_latencies_ms)
            if self.reranking_latencies_ms
            else 0.0
        )
        total = self.total_questions or 1
        return {
            "total_questions": self.total_questions,
            "cache_hit_rate": self.cache_hits / total,
            "semantic_cache_hit_rate": self.semantic_cache_hits / total,
            "exact_cache_hit_rate": self.exact_cache_hits / total,
            "llm_api_calls_saved": self.llm_calls_saved,
            "avg_latency_ms": avg_latency,
            "avg_retrieval_latency_ms": avg_retrieval_latency,
            "avg_reranking_latency_ms": avg_reranking_latency,
            "avg_generation_latency_ms": avg_generation_latency,
            "token_usage": None,
            "estimated_cost_reduction": None,
        }


rag_metrics = RAGMetricsTracker()


def log_rag_metrics_to_mlflow(params: dict[str, Any], metrics: dict[str, Any]) -> None:
    run = start_mlflow_run(run_name="rag_metrics")
    if run is None:
        return
    try:
        log_params(params)
        log_metrics(metrics, prefix="rag")
    finally:
        end_mlflow_run()
