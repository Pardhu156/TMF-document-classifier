from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app
from src.config import PredictionConfig
from tests.auth_helpers import auth_headers, install_auth_override


client = TestClient(app)


def _model_artifacts_available() -> bool:
    config = PredictionConfig()
    return (
        config.model_dir.exists()
        and config.label_encoder_path.exists()
        and any(Path(config.model_dir).glob("*.safetensors"))
    )


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info_endpoint() -> None:
    response = client.get("/model-info")
    payload = response.json()

    assert response.status_code == 200
    assert "model_version" in payload
    assert "dataset_version" in payload
    assert "class_names" in payload


def test_predict_endpoint_valid_text() -> None:
    if not _model_artifacts_available():
        pytest.skip("Saved model artifacts are not available.")
    install_auth_override()

    response = client.post(
        "/predict",
        json={"text": "This document describes study objectives and inclusion criteria."},
        headers=auth_headers("User"),
    )
    payload = response.json()

    assert response.status_code == 200
    assert isinstance(payload["predicted_label"], str)
    assert 0.0 <= float(payload["confidence"]) <= 1.0
    assert 0.0 <= float(payload["model_confidence"]) <= 1.0
    assert payload["decision_status"] in {"auto_classify", "agent_review", "human_review"}


def test_predict_endpoint_empty_text() -> None:
    install_auth_override()
    response = client.post("/predict", json={"text": "   "}, headers=auth_headers("User"))

    assert response.status_code == 400
