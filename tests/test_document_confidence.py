import pytest

from src.utils import document_confidence_summary


def _chunk(label: str, confidence: float) -> dict[str, object]:
    return {"predicted_label": label, "confidence": confidence}


def test_document_confidence_strong_agreement_case() -> None:
    results = [_chunk("protocol", 0.90) for _ in range(40)]
    results += [_chunk("sap", 0.80) for _ in range(10)]

    summary = document_confidence_summary(results, ["protocol", "sap"])

    assert summary["predicted_label"] == "protocol"
    assert summary["vote_confidence"] == pytest.approx(0.80)
    assert summary["margin_confidence"] == pytest.approx(0.60)
    assert summary["confidence"] == pytest.approx(1.0)
    assert summary["decision_status"] == "auto_classify"


def test_document_confidence_close_vote_case_is_penalized() -> None:
    results = [_chunk("protocol", 0.90) for _ in range(32)]
    results += [_chunk("sap", 0.88) for _ in range(30)]

    summary = document_confidence_summary(results, ["protocol", "sap"])

    assert summary["predicted_label"] == "protocol"
    assert summary["vote_confidence"] == pytest.approx(32 / 62)
    assert summary["margin_confidence"] == pytest.approx(2 / 62)
    assert summary["confidence"] < 0.50
    assert summary["requires_review"] is True
    assert summary["decision_status"] == "human_review"


def test_document_confidence_single_class_dominance_case() -> None:
    results = [_chunk("safety_report", 0.70) for _ in range(12)]

    summary = document_confidence_summary(results, ["protocol", "safety_report", "sap"])

    assert summary["predicted_label"] == "safety_report"
    assert summary["vote_confidence"] == pytest.approx(1.0)
    assert summary["margin_confidence"] == pytest.approx(1.0)
    assert summary["confidence"] == pytest.approx(1.0)
    assert summary["decision_status"] == "auto_classify"


def test_document_confidence_multiclass_scenario_more_than_three_classes() -> None:
    results = [_chunk("class_a", 0.82) for _ in range(12)]
    results += [_chunk("class_b", 0.77) for _ in range(8)]
    results += [_chunk("class_c", 0.74) for _ in range(5)]
    results += [_chunk("class_d", 0.91) for _ in range(2)]

    summary = document_confidence_summary(
        results,
        ["class_a", "class_b", "class_c", "class_d", "class_e"],
    )

    assert summary["predicted_label"] == "class_a"
    assert summary["chunk_predictions"] == {
        "class_a": 12,
        "class_b": 8,
        "class_c": 5,
        "class_d": 2,
    }
    assert summary["margin_confidence"] == pytest.approx(4 / 27)
    assert summary["decision_status"] in {"agent_review", "human_review"}
