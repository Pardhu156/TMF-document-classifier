"""Comprehensive RAG + Redis metrics smoke test.

Run with FastAPI already running:
    python scripts/test_rag_metrics.py
    python scripts/test_rag_metrics.py --clear-cache true --top-k 5

The script intentionally uses a small hardcoded question set first. A JSON
evaluation dataset can be plugged in later without changing the metrics module.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import logger
from src.rag.evaluation_report import build_case_metrics, build_report, format_dashboard, save_report


DEFAULT_API_URL = "http://127.0.0.1:8000/rag/ask"

TEST_CASES = [
    {
        "question": "What is the study objective?",
        "paraphrase": "What is the aim of the study?",
        "expected_keywords": ["objective", "study"],
    },
    {
        "question": "What are the inclusion criteria?",
        "paraphrase": "Who is eligible for the study?",
        "expected_keywords": ["inclusion", "criteria", "eligible"],
    },
    {
        "question": "What are the adverse event reporting requirements?",
        "paraphrase": "How should adverse events be reported?",
        "expected_keywords": ["adverse", "event", "reported"],
    },
]


def parse_bool(value: str) -> bool:
    """Parse a CLI boolean in a beginner-friendly way."""
    return value.strip().lower() in {"1", "true", "yes", "y"}


def request_rag(api_url: str, question: str, top_k: int, scope: str = "master") -> dict[str, Any]:
    """Call /rag/ask and normalize the response shape used by the metrics module."""
    payload = {"question": question, "scope": scope}
    start = time.perf_counter()
    response = requests.post(api_url, json=payload, timeout=120)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    try:
        data = response.json()
    except Exception:
        data = {"raw_response": response.text}

    if response.status_code >= 400:
        return {
            "error": data,
            "status_code": response.status_code,
            "question": question,
            "client_elapsed_ms": elapsed_ms,
            "cache_hit": False,
            "cache_type": "none",
        }

    data["question"] = question
    data["client_elapsed_ms"] = elapsed_ms
    data["retrieved_chunks"] = (data.get("retrieved_chunks") or [])[:top_k]
    return data


def clear_redis_cache() -> str:
    """Clear local Redis cache for a clean before/after measurement."""
    try:
        import redis

        client = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
        client.flushdb()
        logger.info("Redis cache cleared before RAG metrics run.")
        return "Redis cache cleared with FLUSHDB."
    except Exception as error:
        logger.warning("Redis cache was not cleared: %s", error)
        return f"Redis cache was not cleared: {error}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure RAG retrieval and Redis cache performance.")
    parser.add_argument("--clear-cache", default="false", help="true/false. Flush Redis DB before running.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of returned chunks to inspect.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="RAG ask endpoint URL.")
    parser.add_argument("--output-dir", default="reports", help="Directory for text/JSON metrics reports.")
    args = parser.parse_args()

    clear_cache = parse_bool(args.clear_cache)
    preface_lines: list[str] = []
    if clear_cache:
        preface_lines.append(clear_redis_cache())

    logger.info("Starting RAG + Redis metrics run against %s", args.api_url)
    case_metrics: list[dict[str, Any]] = []
    error_lines: list[str] = []

    for index, test_case in enumerate(TEST_CASES, start=1):
        question = test_case["question"]
        paraphrase = test_case["paraphrase"]
        logger.info("Running RAG metrics test case %s: %s", index, question)

        first_run = request_rag(args.api_url, question, args.top_k)
        second_run = request_rag(args.api_url, question, args.top_k)
        semantic_run = request_rag(args.api_url, paraphrase, args.top_k)

        errors = {
            "first": first_run.get("error"),
            "second": second_run.get("error"),
            "semantic": semantic_run.get("error"),
        }
        if any(errors.values()):
            error_lines.append(f"Test case {index} failed: {errors}")
            logger.error("RAG metrics test case %s failed: %s", index, errors)
            continue

        case_metrics.append(
            build_case_metrics(
                test_case=test_case,
                first_run=first_run,
                second_run=second_run,
                semantic_run=semantic_run,
                top_k=args.top_k,
            )
        )

    if not case_metrics:
        raise RuntimeError(
            "No RAG metrics could be computed. Make sure FastAPI, Redis, PostgreSQL/pgvector, "
            "and the Gemini generation key are available."
        )

    report = build_report(
        case_metrics,
        top_k=args.top_k,
        api_url=args.api_url,
        clear_cache=clear_cache,
    )
    saved_paths = save_report(report, output_dir=Path(args.output_dir))
    dashboard = format_dashboard(report)

    output_lines = preface_lines + error_lines + [dashboard]
    print("\n".join(line for line in output_lines if line))
    print("\nSaved metrics:")
    for label, path in saved_paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
