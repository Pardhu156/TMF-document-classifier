"""Single-text inference using a locally saved TMF classifier."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib

from src.config import PredictionConfig
from src.exception import CustomException
from src.logger import logger


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
    return tokenizer, model, joblib.load(encoder_path)


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
