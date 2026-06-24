"""Single-text inference using a locally saved TMF classifier."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib

from src.config import PredictionConfig
from src.exception import CustomException
from src.logger import logger


def _validate_model_encoder_compatibility(model: object, label_encoder: object) -> None:
    """Fail fast when a model and label encoder do not describe the same labels."""
    model_config = model.config
    classes = [str(label) for label in label_encoder.classes_]
    if model_config.num_labels != len(classes):
        raise ValueError("Model output count does not match the saved label encoder.")
    configured_labels = [str(model_config.id2label.get(index, model_config.id2label.get(str(index), ""))) for index in range(model_config.num_labels)]
    if configured_labels and configured_labels != classes:
        raise ValueError("Model label order does not match the saved label encoder.")


@lru_cache(maxsize=1)
def _load_inference_artifacts(model_dir: str, label_encoder_path: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(model_dir)
    encoder_path = Path(label_encoder_path)
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("Saved model and label encoder must both exist in artifacts/.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    label_encoder = joblib.load(encoder_path)
    _validate_model_encoder_compatibility(model, label_encoder)
    return tokenizer, model, label_encoder


def predict_text(text: str, config: PredictionConfig | None = None) -> dict[str, float | str]:
    """Return the label and softmax confidence for one non-empty text input."""
    config = config or PredictionConfig()
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        import torch

        tokenizer, model, label_encoder = _load_inference_artifacts(
            str(config.model_dir), str(config.label_encoder_path)
        )
        encoded = tokenizer(text, truncation=True, max_length=config.max_length, return_tensors="pt")
        with torch.no_grad():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        result = {
            "predicted_label": str(label_encoder.inverse_transform([predicted_index])[0]),
            "confidence": float(probabilities[predicted_index].item()),
        }
        logger.info("Predicted label '%s' with confidence %.4f", result["predicted_label"], result["confidence"])
        return result
    except Exception as error:
        logger.exception("Prediction failed")
        raise CustomException(error, sys.exc_info()) from error
