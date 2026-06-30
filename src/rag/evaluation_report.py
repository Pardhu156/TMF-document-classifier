"""Reusable reporting utilities for RAG + Redis evaluation runs.

This module is intentionally offline-friendly: it accepts already collected API
responses, computes aggregate metrics, writes JSON/text reports, and keeps a
small run history for trend comparison. It does not call the RAG API, Redis,
PostgreSQL, S3, Gemini, or the classifier by itself.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any

from src.logger import logger
from src.utils import ensure_dir, get_git_commit_hash, save_json


CACHE_TYPES = ("exact", "semantic", "none")


def safe_mean(values: list[float | int | None]) -> float | None:
    """Return a mean for numeric values, or None when no values are available."""
    clean_values = [float(value) for value in values if isinstance(value, (int, float))]
    if not clean_values:
        return None
    return float(statistics.mean(clean_values))


def percent(numerator: float, denominator: float) -> float:
    """Return a percentage while avoiding division-by-zero."""
    if denominator <= 0:
        return 0.0
    return float((numerator / denominator) * 100)


def latency_ms(response: dict[str, Any]) -> float:
    """Prefer server-reported latency and fall back to client-measured latency."""
    value = response.get("latency_ms", response.get("client_elapsed_ms", 0))
    return float(value or 0)


def speedup_factor(baseline_latency_ms: float, comparison_latency_ms: float) -> float | None:
    """Return how many times faster the comparison is versus the baseline."""
    if baseline_latency_ms <= 0 or comparison_latency_ms <= 0:
        return None
    return float(baseline_latency_ms / comparison_latency_ms)


def latency_reduction_percent(baseline_latency_ms: float, comparison_latency_ms: float) -> float | None:
    """Return latency reduction percentage from baseline to comparison."""
    if baseline_latency_ms <= 0:
        return None
    return float(max(0.0, (baseline_latency_ms - comparison_latency_ms) / baseline_latency_ms) * 100)


def chunk_scores(chunks: list[dict[str, Any]]) -> list[float]:
    """Collect available retrieval scores from retrieved chunks."""
    scores: list[float] = []
    for chunk in chunks:
        score = chunk.get("hybrid_score")
        if score is None:
            score = chunk.get("semantic_score")
        if score is None:
            score = chunk.get("keyword_score")
        if isinstance(score, (int, float)):
            scores.append(float(score))
    return scores


def extract_retrieval_metrics(response: dict[str, Any]) -> dict[str, Any]:
    """Extract retrieval diagnostics from a /rag/ask response."""
    chunks = response.get("retrieved_chunks") or []
    sources = response.get("sources") or []
    scores = chunk_scores(chunks)
    top_chunk = chunks[0] if chunks else {}

    retrieved_file_names = sorted(
        {
            str(chunk.get("file_name") or source.get("file_name"))
            for chunk, source in zip(chunks, sources + [{}] * len(chunks))
            if chunk.get("file_name") or source.get("file_name")
        }
    )
    retrieved_document_ids = sorted(
        {
            str(chunk.get("document_id") or source.get("document_id"))
            for chunk, source in zip(chunks, sources + [{}] * len(chunks))
            if chunk.get("document_id") or source.get("document_id")
        }
    )
    verified_sources = [
        chunk
        for chunk in chunks
        if str(chunk.get("verification_status", "")).lower() in {"verified", "approved", "master"}
    ]

    return {
        "retrieval_scope": response.get("retrieval_scope"),
        "matched_file_name": response.get("matched_file_name"),
        "retrieval_latency_ms": response.get("retrieval_latency_ms"),
        "total_latency_ms": latency_ms(response),
        "number_of_retrieved_chunks": len(chunks),
        "retrieved_file_names": retrieved_file_names,
        "retrieved_document_ids": retrieved_document_ids,
        "top_retrieved_file": top_chunk.get("file_name"),
        "top_chunk_preview": str(top_chunk.get("chunk_text", ""))[:300],
        "source_type": top_chunk.get("source_type"),
        "verification_status": top_chunk.get("verification_status"),
        "hybrid_score": top_chunk.get("hybrid_score"),
        "semantic_score": top_chunk.get("semantic_score"),
        "keyword_score": top_chunk.get("keyword_score"),
        "top_chunk_score": scores[0] if scores else None,
        "average_retrieval_score": safe_mean(scores),
        "source_verification_success": bool(chunks) and len(verified_sources) == len(chunks),
        "retrieved_chunk_ids": [
            str(chunk.get("chunk_id") or chunk.get("id"))
            for chunk in chunks
            if chunk.get("chunk_id") or chunk.get("id")
        ],
    }


def answer_quality_metrics(response: dict[str, Any], expected_keywords: list[str]) -> dict[str, Any]:
    """Compute lightweight answer quality checks without an LLM judge."""
    answer = str(response.get("answer") or "")
    lower_answer = answer.lower()
    matched_keywords = [keyword for keyword in expected_keywords if keyword.lower() in lower_answer]
    return {
        "keyword_coverage_score": len(matched_keywords) / max(len(expected_keywords), 1),
        "keyword_coverage_score_percent": percent(len(matched_keywords), max(len(expected_keywords), 1)),
        "matched_keywords": matched_keywords,
        "citation_present": bool(response.get("sources")),
        "answer_non_empty": bool(answer.strip()),
    }


def precision_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str] | None,
    k: int,
) -> dict[str, float | None]:
    """Compute Precision@K and Recall@K when ground truth IDs are provided."""
    if not relevant_ids:
        return {"precision_at_k": None, "recall_at_k": None}
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    hits = [chunk_id for chunk_id in top_k if chunk_id in relevant]
    return {
        "precision_at_k": len(hits) / max(len(top_k), 1),
        "recall_at_k": len(hits) / max(len(relevant), 1),
    }


def build_case_metrics(
    test_case: dict[str, Any],
    first_run: dict[str, Any],
    second_run: dict[str, Any],
    semantic_run: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """Build one normalized metrics record for a question/paraphrase test case."""
    first_latency = latency_ms(first_run)
    second_latency = latency_ms(second_run)
    semantic_latency = latency_ms(semantic_run)
    retrieval = extract_retrieval_metrics(first_run)
    quality = answer_quality_metrics(first_run, test_case.get("expected_keywords", []))
    pr_at_k = precision_recall_at_k(
        retrieval.get("retrieved_chunk_ids", []),
        test_case.get("relevant_chunk_ids"),
        top_k,
    )

    return {
        "question": test_case.get("question"),
        "paraphrase": test_case.get("paraphrase"),
        "expected_keywords": test_case.get("expected_keywords", []),
        "first_run": {
            "cache_hit": bool(first_run.get("cache_hit")),
            "cache_type": first_run.get("cache_type", "none"),
            "latency_ms": first_latency,
        },
        "second_run": {
            "cache_hit": bool(second_run.get("cache_hit")),
            "cache_type": second_run.get("cache_type", "none"),
            "latency_ms": second_latency,
            "exact_cache_speedup_factor": speedup_factor(first_latency, second_latency),
            "latency_reduction_percent": latency_reduction_percent(first_latency, second_latency),
        },
        "semantic_run": {
            "cache_hit": bool(semantic_run.get("cache_hit")),
            "cache_type": semantic_run.get("cache_type", "none"),
            "latency_ms": semantic_latency,
            "semantic_cache_speedup_factor": speedup_factor(first_latency, semantic_latency),
            "latency_reduction_percent": latency_reduction_percent(first_latency, semantic_latency),
        },
        "retrieval": retrieval,
        "answer_quality": quality,
        "precision_at_k": pr_at_k["precision_at_k"],
        "recall_at_k": pr_at_k["recall_at_k"],
    }


def _cache_type_counts(case_metrics: list[dict[str, Any]]) -> Counter[str]:
    cache_types: list[str] = []
    for case in case_metrics:
        cache_types.extend(
            [
                str(case["first_run"].get("cache_type", "none")),
                str(case["second_run"].get("cache_type", "none")),
                str(case["semantic_run"].get("cache_type", "none")),
            ]
        )
    return Counter(cache_types)


def aggregate_metrics(case_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all per-case records into a dashboard-ready summary."""
    total_questions = len(case_metrics)
    cache_counts = _cache_type_counts(case_metrics)
    total_cache_observations = sum(cache_counts.values())

    first_latencies = [case["first_run"]["latency_ms"] for case in case_metrics]
    second_latencies = [case["second_run"]["latency_ms"] for case in case_metrics]
    semantic_latencies = [case["semantic_run"]["latency_ms"] for case in case_metrics]
    all_runs = [
        case[run_key]
        for case in case_metrics
        for run_key in ("first_run", "second_run", "semantic_run")
    ]
    uncached_latencies = [run["latency_ms"] for run in all_runs if run.get("cache_type") == "none"]
    cached_latencies = [run["latency_ms"] for run in all_runs if run.get("cache_type") != "none"]
    exact_latencies = [run["latency_ms"] for run in all_runs if run.get("cache_type") == "exact"]
    semantic_hit_latencies = [run["latency_ms"] for run in all_runs if run.get("cache_type") == "semantic"]

    exact_speedups = [case["second_run"]["exact_cache_speedup_factor"] for case in case_metrics]
    semantic_speedups = [case["semantic_run"]["semantic_cache_speedup_factor"] for case in case_metrics]

    cache_time_saved_ms = 0.0
    cache_time_baseline_ms = 0.0
    for case in case_metrics:
        baseline = case["first_run"]["latency_ms"]
        for run_key in ("second_run", "semantic_run"):
            run = case[run_key]
            if run.get("cache_type") != "none":
                cache_time_baseline_ms += baseline
                cache_time_saved_ms += max(0.0, baseline - run["latency_ms"])

    retrievals = [case["retrieval"] for case in case_metrics]
    qualities = [case["answer_quality"] for case in case_metrics]
    precision_scores = [case["precision_at_k"] for case in case_metrics]
    recall_scores = [case["recall_at_k"] for case in case_metrics]

    avg_uncached_latency = safe_mean(uncached_latencies)
    avg_cached_latency = safe_mean(cached_latencies)

    summary = {
        "total_questions": total_questions,
        "total_api_calls": total_cache_observations,
        "cache_distribution": {
            cache_type: int(cache_counts.get(cache_type, 0)) for cache_type in CACHE_TYPES
        },
        "exact_cache_hit_rate_percent": percent(cache_counts.get("exact", 0), total_cache_observations),
        "semantic_cache_hit_rate_percent": percent(cache_counts.get("semantic", 0), total_cache_observations),
        "cache_miss_rate_percent": percent(cache_counts.get("none", 0), total_cache_observations),
        "avg_first_run_latency_ms": safe_mean(first_latencies),
        "avg_second_run_latency_ms": safe_mean(second_latencies),
        "avg_uncached_latency_ms": avg_uncached_latency,
        "avg_cached_latency_ms": avg_cached_latency,
        "avg_exact_cache_latency_ms": safe_mean(exact_latencies),
        "avg_semantic_cache_latency_ms": safe_mean(semantic_hit_latencies),
        "avg_semantic_run_latency_ms": safe_mean(semantic_latencies),
        "exact_cache_speedup_factor": safe_mean(exact_speedups),
        "semantic_cache_speedup_factor": safe_mean(semantic_speedups),
        "overall_latency_reduction_percent": (
            latency_reduction_percent(avg_uncached_latency, avg_cached_latency)
            if avg_uncached_latency is not None and avg_cached_latency is not None
            else None
        ),
        "total_cache_time_saved_ms": cache_time_saved_ms,
        "total_cache_time_saved_percent": percent(cache_time_saved_ms, cache_time_baseline_ms),
        "avg_retrieval_latency_ms": safe_mean(
            [retrieval.get("retrieval_latency_ms") for retrieval in retrievals]
        ),
        "avg_retrieved_chunks": safe_mean(
            [retrieval.get("number_of_retrieved_chunks") for retrieval in retrievals]
        ),
        "avg_top_chunk_score": safe_mean([retrieval.get("top_chunk_score") for retrieval in retrievals]),
        "avg_retrieval_score": safe_mean(
            [retrieval.get("average_retrieval_score") for retrieval in retrievals]
        ),
        "avg_keyword_coverage_score_percent": safe_mean(
            [quality.get("keyword_coverage_score_percent") for quality in qualities]
        ),
        "citation_presence_rate_percent": percent(
            sum(1 for quality in qualities if quality.get("citation_present")),
            total_questions,
        ),
        "answer_availability_rate_percent": percent(
            sum(1 for quality in qualities if quality.get("answer_non_empty")),
            total_questions,
        ),
        "source_verification_success_rate_percent": percent(
            sum(1 for retrieval in retrievals if retrieval.get("source_verification_success")),
            total_questions,
        ),
        "precision_at_k": safe_mean(precision_scores),
        "recall_at_k": safe_mean(recall_scores),
    }
    return summary


def build_interpretations(summary: dict[str, Any]) -> list[str]:
    """Create human-readable interpretations from aggregate metrics."""
    interpretations: list[str] = []
    latency_reduction = summary.get("overall_latency_reduction_percent")
    exact_speedup = summary.get("exact_cache_speedup_factor")
    semantic_speedup = summary.get("semantic_cache_speedup_factor")
    semantic_hit_rate = float(summary.get("semantic_cache_hit_rate_percent") or 0)
    exact_hit_rate = float(summary.get("exact_cache_hit_rate_percent") or 0)

    if isinstance(latency_reduction, (int, float)):
        interpretations.append(f"Redis reduced average cached latency by {latency_reduction:.1f}%.")
    if isinstance(exact_speedup, (int, float)):
        interpretations.append(f"Exact cache achieved {exact_speedup:.2f}x average speedup.")
    if isinstance(semantic_speedup, (int, float)):
        interpretations.append(f"Semantic cache achieved {semantic_speedup:.2f}x average speedup.")

    if exact_hit_rate >= 25:
        interpretations.append("Exact cache is working as expected for repeated questions.")
    else:
        interpretations.append("Exact cache hit rate is low; confirm repeated queries are normalized consistently.")

    if semantic_hit_rate >= 30:
        interpretations.append("Semantic cache hit rate is acceptable for paraphrased questions.")
    elif semantic_hit_rate >= 10:
        interpretations.append("Semantic cache hit rate is improving but still worth tuning.")
    else:
        interpretations.append("Semantic cache hit rate needs improvement; tune threshold or embedding coverage.")

    keyword_coverage = summary.get("avg_keyword_coverage_score_percent")
    if isinstance(keyword_coverage, (int, float)):
        interpretations.append(f"Average keyword coverage is {keyword_coverage:.1f}%.")
    return interpretations


def build_report(
    case_metrics: list[dict[str, Any]],
    *,
    top_k: int,
    api_url: str,
    clear_cache: bool,
) -> dict[str, Any]:
    """Build the complete JSON-serializable report."""
    summary = aggregate_metrics(case_metrics)
    return {
        "run_metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit_hash": get_git_commit_hash(),
            "api_url": api_url,
            "top_k": top_k,
            "clear_cache": clear_cache,
        },
        "summary": summary,
        "interpretations": build_interpretations(summary),
        "cases": case_metrics,
    }


def _format_value(value: Any, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def format_dashboard(report: dict[str, Any]) -> str:
    """Return a terminal-friendly dashboard report."""
    summary = report["summary"]
    lines = [
        "\n================ RAG + Redis Metrics Dashboard ================",
        f"created_at: {report['run_metadata']['created_at']}",
        f"total_questions: {summary['total_questions']}",
        f"total_api_calls: {summary['total_api_calls']}",
        "",
        "Cache performance",
        f"  exact_cache_hit_rate: {_format_value(summary['exact_cache_hit_rate_percent'], '%')}",
        f"  semantic_cache_hit_rate: {_format_value(summary['semantic_cache_hit_rate_percent'], '%')}",
        f"  cache_miss_rate: {_format_value(summary['cache_miss_rate_percent'], '%')}",
        f"  distribution: {summary['cache_distribution']}",
        "",
        "Latency",
        f"  avg_uncached_latency_ms: {_format_value(summary['avg_uncached_latency_ms'])}",
        f"  avg_cached_latency_ms: {_format_value(summary['avg_cached_latency_ms'])}",
        f"  avg_semantic_cache_latency_ms: {_format_value(summary['avg_semantic_cache_latency_ms'])}",
        f"  exact_cache_speedup_factor: {_format_value(summary['exact_cache_speedup_factor'], 'x')}",
        f"  semantic_cache_speedup_factor: {_format_value(summary['semantic_cache_speedup_factor'], 'x')}",
        f"  overall_latency_reduction: {_format_value(summary['overall_latency_reduction_percent'], '%')}",
        f"  total_cache_time_saved_ms: {_format_value(summary['total_cache_time_saved_ms'])}",
        f"  total_cache_time_saved_percent: {_format_value(summary['total_cache_time_saved_percent'], '%')}",
        "",
        "Retrieval",
        f"  avg_retrieval_latency_ms: {_format_value(summary['avg_retrieval_latency_ms'])}",
        f"  avg_retrieved_chunks: {_format_value(summary['avg_retrieved_chunks'])}",
        f"  avg_top_chunk_score: {_format_value(summary['avg_top_chunk_score'])}",
        f"  avg_retrieval_score: {_format_value(summary['avg_retrieval_score'])}",
        f"  source_verification_success_rate: {_format_value(summary['source_verification_success_rate_percent'], '%')}",
        "",
        "Answer sanity checks",
        f"  avg_keyword_coverage_score: {_format_value(summary['avg_keyword_coverage_score_percent'], '%')}",
        f"  citation_presence_rate: {_format_value(summary['citation_presence_rate_percent'], '%')}",
        f"  answer_availability_rate: {_format_value(summary['answer_availability_rate_percent'], '%')}",
        f"  precision_at_k: {_format_value(summary['precision_at_k'])}",
        f"  recall_at_k: {_format_value(summary['recall_at_k'])}",
        "",
        "Interpretation",
    ]
    lines.extend([f"  - {interpretation}" for interpretation in report["interpretations"]])

    lines.append("\nPer-question sanity check")
    for index, case in enumerate(report["cases"], start=1):
        retrieval = case["retrieval"]
        quality = case["answer_quality"]
        lines.extend(
            [
                f"\n  {index}. {case['question']}",
                f"     first={case['first_run']['cache_type']} "
                f"{case['first_run']['latency_ms']:.0f}ms | "
                f"second={case['second_run']['cache_type']} "
                f"{case['second_run']['latency_ms']:.0f}ms | "
                f"semantic={case['semantic_run']['cache_type']} "
                f"{case['semantic_run']['latency_ms']:.0f}ms",
                f"     top_file={retrieval['top_retrieved_file']} "
                f"score={retrieval['top_chunk_score']} chunks={retrieval['number_of_retrieved_chunks']}",
                f"     source_type={retrieval['source_type']} "
                f"verification={retrieval['verification_status']} "
                f"keyword_coverage={quality['keyword_coverage_score_percent']:.1f}%",
                f"     preview={retrieval['top_chunk_preview']}",
            ]
        )
    return "\n".join(lines)


def append_run_history(summary: dict[str, Any], history_path: Path) -> Path:
    """Append compact summary metrics for trend/regression tracking."""
    ensure_dir(history_path.parent)
    history: list[dict[str, Any]]
    if history_path.exists():
        try:
            loaded = json.loads(history_path.read_text(encoding="utf-8"))
            history = loaded if isinstance(loaded, list) else []
        except json.JSONDecodeError:
            logger.warning("Existing RAG metrics history is invalid JSON; starting a new history file.")
            history = []
    else:
        history = []
    history.append(summary)
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return history_path


def save_report(report: dict[str, Any], output_dir: Path | str = "reports") -> dict[str, Path]:
    """Save JSON, text, timestamped JSON, and compact run history."""
    directory = ensure_dir(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    latest_json = directory / "rag_metrics_summary.json"
    latest_text = directory / "rag_metrics_summary.txt"
    runs_dir = ensure_dir(directory / "rag_metrics_runs")
    timestamped_json = runs_dir / f"rag_metrics_{timestamp}.json"
    history_path = directory / "rag_metrics_history.json"

    dashboard = format_dashboard(report)
    save_json(report, latest_json)
    save_json(report, timestamped_json)
    latest_text.write_text(dashboard, encoding="utf-8")
    append_run_history(
        {
            "created_at": report["run_metadata"]["created_at"],
            **report["summary"],
        },
        history_path,
    )
    logger.info("Saved RAG metrics report to %s", latest_json)
    return {
        "latest_json": latest_json,
        "latest_text": latest_text,
        "timestamped_json": timestamped_json,
        "history_json": history_path,
    }
