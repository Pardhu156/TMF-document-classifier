from pathlib import Path

import pytest

from src.config import PredictionConfig
from src.predict import predict_text


def _model_artifacts_available() -> bool:
    config = PredictionConfig()
    return (
        config.model_dir.exists()
        and config.label_encoder_path.exists()
        and any(Path(config.model_dir).glob("*.safetensors"))
    )


def test_predict_text_returns_label_and_confidence() -> None:
    if not _model_artifacts_available():
        pytest.skip("Saved model artifacts are not available.")

    result = predict_text("This document describes study objectives and inclusion criteria.")

    assert isinstance(result["predicted_label"], str)
    assert 0.0 <= float(result["confidence"]) <= 1.0
    assert 0.0 <= float(result["model_confidence"]) <= 1.0
    assert result["decision_status"] in {"auto_classify", "agent_review", "human_review"}


def test_predict_text_rejects_empty_input() -> None:
    with pytest.raises(Exception):
        predict_text("")
