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


def test_predict_file_valid_txt_upload(monkeypatch) -> None:
    if not _model_artifacts_available():
        pytest.skip("Saved model artifacts are not available.")
    install_auth_override()
    monkeypatch.setattr("src.rag.service.RAGIndexer.is_configured", classmethod(lambda cls: False))

    response = client.post(
        "/predict-file",
        files={
            "file": (
                "sample_protocol.txt",
                b"This protocol describes study objectives, inclusion criteria, treatment procedures, and visits.",
                "text/plain",
            )
        },
        headers=auth_headers("User"),
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["filename"] == "sample_protocol.txt"
    assert isinstance(payload["predicted_label"], str)
    assert 0.0 <= float(payload["confidence"]) <= 1.0
    assert 0.0 <= float(payload["model_confidence"]) <= 1.0
    assert 0.0 <= float(payload["vote_confidence"]) <= 1.0
    assert 0.0 <= float(payload["margin_confidence"]) <= 1.0
    assert isinstance(payload["requires_review"], bool)
    assert payload["decision_status"] in {"auto_classify", "agent_review", "human_review"}
    assert payload["num_chunks"] >= 1
    assert isinstance(payload["chunk_predictions"], dict)


def test_predict_file_rejects_unsupported_upload() -> None:
    install_auth_override()
    response = client.post(
        "/predict-file",
        files={"file": ("sample.csv", b"not,a,supported,document", "text/csv")},
        headers=auth_headers("User"),
    )

    assert response.status_code == 400


def test_predict_file_rejects_empty_upload() -> None:
    install_auth_override()
    response = client.post(
        "/predict-file",
        files={"file": ("empty.txt", b"", "text/plain")},
        headers=auth_headers("User"),
    )

    assert response.status_code == 400
