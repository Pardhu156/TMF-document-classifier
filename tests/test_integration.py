from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from src.config import PredictionConfig


client = TestClient(app)


def _model_artifacts_available() -> bool:
    config = PredictionConfig()
    return (
        config.model_dir.exists()
        and config.label_encoder_path.exists()
        and any(Path(config.model_dir).glob("*.safetensors"))
    )


def test_api_health_and_model_info_flow() -> None:
    health_response = client.get("/health")
    info_response = client.get("/model-info")

    assert health_response.status_code == 200
    assert info_response.status_code == 200
    assert "model_name" in info_response.json()


def test_api_prediction_flow_with_saved_model() -> None:
    if not _model_artifacts_available():
        pytest.skip("Saved model artifacts are not available.")

    response = client.post(
        "/predict",
        json={"text": "The protocol defines treatment procedures and study objectives."},
    )

    assert response.status_code == 200
    assert {
        "predicted_label",
        "confidence",
        "model_confidence",
        "vote_confidence",
        "margin_confidence",
        "requires_review",
        "decision_status",
    }.issubset(response.json())
