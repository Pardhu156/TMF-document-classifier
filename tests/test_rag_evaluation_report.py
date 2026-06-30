import pytest

from src.rag.evaluation_report import build_case_metrics, build_report


def _response(cache_type: str, latency_ms: int, answer: str = "The study objective is safety.") -> dict:
    return {
        "answer": answer,
        "sources": [{"document_id": "doc_1", "file_name": "protocol.pdf", "chunk_id": "chunk_1"}],
        "cache_hit": cache_type != "none",
        "cache_type": cache_type,
        "latency_ms": latency_ms,
        "retrieval_scope": "master",
        "matched_file_name": None,
        "retrieved_chunks": [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "file_name": "protocol.pdf",
                "chunk_text": "The primary objective of the study is to evaluate safety.",
                "source_type": "MASTER_DATA",
                "verification_status": "verified",
                "hybrid_score": 0.91,
                "semantic_score": 0.88,
                "keyword_score": 0.79,
            }
        ],
    }


def test_build_case_metrics_computes_cache_and_quality_metrics() -> None:
    test_case = {
        "question": "What is the study objective?",
        "paraphrase": "What is the aim of the study?",
        "expected_keywords": ["objective", "study"],
        "relevant_chunk_ids": ["chunk_1"],
    }

    case = build_case_metrics(
        test_case=test_case,
        first_run=_response("none", 1000),
        second_run=_response("exact", 100),
        semantic_run=_response("semantic", 250),
        top_k=5,
    )

    assert case["second_run"]["exact_cache_speedup_factor"] == 10
    assert case["semantic_run"]["semantic_cache_speedup_factor"] == 4
    assert case["answer_quality"]["keyword_coverage_score_percent"] == 100
    assert case["retrieval"]["source_verification_success"] is True
    assert case["precision_at_k"] == 1
    assert case["recall_at_k"] == 1


def test_build_report_aggregates_dashboard_metrics() -> None:
    test_case = {
        "question": "What is the study objective?",
        "paraphrase": "What is the aim of the study?",
        "expected_keywords": ["objective", "study"],
    }
    case = build_case_metrics(
        test_case=test_case,
        first_run=_response("none", 1000),
        second_run=_response("exact", 100),
        semantic_run=_response("semantic", 200),
        top_k=5,
    )

    report = build_report([case], top_k=5, api_url="http://test/rag/ask", clear_cache=True)
    summary = report["summary"]

    assert summary["exact_cache_hit_rate_percent"] == pytest.approx(100 / 3)
    assert summary["semantic_cache_hit_rate_percent"] == pytest.approx(100 / 3)
    assert summary["cache_miss_rate_percent"] == pytest.approx(100 / 3)
    assert summary["avg_cached_latency_ms"] == 150
    assert summary["overall_latency_reduction_percent"] == 85
    assert summary["total_cache_time_saved_ms"] == 1700
    assert report["interpretations"]
