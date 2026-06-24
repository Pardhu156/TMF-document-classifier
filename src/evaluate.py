"""Chunk-level and majority-vote document-level model evaluation."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm

from src.config import DataIngestionConfig, EvaluationConfig, ModelTrainingConfig
from src.exception import CustomException
from src.logger import logger
from src.predict import _load_inference_artifacts


def _metric_summary(y_true: list[str], y_pred: list[str], classes: list[str]) -> tuple[dict, np.ndarray]:
    matrix = confusion_matrix(y_true, y_pred, labels=classes)
    return (
        {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
            "classification_report": classification_report(
                y_true, y_pred, labels=classes, target_names=classes, zero_division=0, output_dict=True
            ),
        },
        matrix,
    )


def _predict_saved_model(test_df: pd.DataFrame, training_config: ModelTrainingConfig) -> np.ndarray:
    """Generate encoded chunk predictions from the model already in artifacts/."""
    import torch

    tokenizer, model, _ = _load_inference_artifacts(
        str(training_config.save_model_dir), str(training_config.label_encoder_path)
    )
    predicted_indices: list[int] = []
    for start in tqdm(
        range(0, len(test_df), 16),
        desc="Evaluating chunks",
        unit="batch",
    ):
        texts = test_df["chunk_text"].iloc[start : start + 16].astype(str).tolist()
        batch = tokenizer(texts, truncation=True, max_length=training_config.max_length, padding=True, return_tensors="pt")
        with torch.no_grad():
            predicted_indices.extend(torch.argmax(model(**batch).logits, dim=-1).tolist())
    return np.asarray(predicted_indices)


def run_evaluation(
    trainer=None,
    test_dataset=None,
    test_df: pd.DataFrame | None = None,
    label_encoder=None,
    data_config: DataIngestionConfig | None = None,
    training_config: ModelTrainingConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
) -> dict:
    """Evaluate chunk predictions and document-level majority-vote predictions.

    Pass the returned objects from ``run_training`` to avoid reloading the model, or
    call without them to evaluate the saved model against ``artifacts/test.csv``.
    """
    data_config = data_config or DataIngestionConfig()
    training_config = training_config or ModelTrainingConfig()
    evaluation_config = evaluation_config or EvaluationConfig()
    try:
        if test_df is None:
            if not data_config.test_data_path.exists():
                raise FileNotFoundError(f"Test dataset not found: {data_config.test_data_path}")
            test_df = pd.read_csv(data_config.test_data_path)
        required_columns = {"file_name", "chunk_text", "class"}
        missing_columns = required_columns.difference(test_df.columns)
        if missing_columns:
            raise ValueError(f"Test data is missing required columns: {sorted(missing_columns)}")
        test_df = test_df.dropna(subset=list(required_columns)).copy()
        test_df["class"] = test_df["class"].astype(str)

        if label_encoder is None:
            if not training_config.label_encoder_path.exists():
                raise FileNotFoundError(f"Label encoder not found: {training_config.label_encoder_path}")
            label_encoder = joblib.load(training_config.label_encoder_path)
        classes = [str(label) for label in label_encoder.classes_]

        if trainer is not None and test_dataset is not None:
            logger.info("Evaluating with the in-memory trainer.")
            predicted_indices = np.argmax(trainer.predict(test_dataset).predictions, axis=-1)
        else:
            logger.info("Evaluating the saved model from %s", training_config.save_model_dir)
            predicted_indices = _predict_saved_model(test_df, training_config)

        if len(predicted_indices) != len(test_df):
            raise RuntimeError("Prediction count does not match the number of test chunks.")
        test_df["predicted_class"] = label_encoder.inverse_transform(predicted_indices).astype(str)

        chunk_metrics, chunk_matrix = _metric_summary(
            test_df["class"].tolist(), test_df["predicted_class"].tolist(), classes
        )
        document_df = (
            test_df.groupby("file_name", as_index=False)
            .agg(
                actual_class=("class", lambda labels: Counter(labels).most_common(1)[0][0]),
                predicted_class=("predicted_class", lambda labels: Counter(labels).most_common(1)[0][0]),
            )
        )
        document_metrics, document_matrix = _metric_summary(
            document_df["actual_class"].tolist(), document_df["predicted_class"].tolist(), classes
        )

        metrics = {"chunk_level": chunk_metrics, "document_level": document_metrics}
        train_df = pd.read_csv(data_config.train_data_path) if data_config.train_data_path.exists() else pd.DataFrame()
        metadata = {
            "train_docs": int(train_df["file_name"].nunique()) if "file_name" in train_df else None,
            "test_docs": int(document_df["file_name"].nunique()),
            "train_chunks": int(len(train_df)) if not train_df.empty else None,
            "test_chunks": int(len(test_df)),
            "classes": classes,
            "model_name": training_config.model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_level_metrics": chunk_metrics,
            "document_level_metrics": document_metrics,
        }
        matrix_rows: list[dict[str, object]] = []
        for level, matrix in (("chunk_level", chunk_matrix), ("document_level", document_matrix)):
            for true_label, row in zip(classes, matrix):
                matrix_rows.append({"evaluation_level": level, "true_label": true_label, **dict(zip(classes, row.tolist()))})

        evaluation_config.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        pd.DataFrame(matrix_rows).to_csv(evaluation_config.confusion_matrix_path, index=False)
        evaluation_config.run_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info("Evaluation complete: chunk macro-F1 %.4f; document macro-F1 %.4f.", chunk_metrics["macro_f1"], document_metrics["macro_f1"])
        return {"metrics": metrics, "metadata": metadata, "document_predictions": document_df}
    except Exception as error:
        logger.exception("Evaluation failed")
        raise CustomException(error, sys.exc_info()) from error
