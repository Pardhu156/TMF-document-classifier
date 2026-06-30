from src.rag.evaluation import generation_metric_placeholders, retrieval_metrics


def test_retrieval_metrics_calculates_hit_rate_and_recall() -> None:
    metrics = retrieval_metrics(
        retrieved_chunk_ids=["a", "b", "c"],
        relevant_chunk_ids=["b", "d"],
        k=3,
    )

    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["recall_at_k"] == 0.5
    assert metrics["precision_at_k"] == 1 / 3


def test_retrieval_metrics_handles_no_relevant_chunks() -> None:
    metrics = retrieval_metrics(
        retrieved_chunk_ids=["a", "b", "c"],
        relevant_chunk_ids=[],
        k=3,
    )

    assert metrics["hit_rate_at_k"] == 0.0
    assert metrics["recall_at_k"] == 0.0


def test_generation_metric_placeholders_are_present() -> None:
    metrics = generation_metric_placeholders()

    assert "answer_relevance" in metrics
    assert "faithfulness" in metrics
    assert metrics["answer_relevance"] is None
