"""Writers for versioned, structured project metadata."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import (
    DataIngestionConfig,
    EvaluationConfig,
    MetadataConfig,
    MLOpsConfig,
    ModelTrainingConfig,
)
from src.utils import get_current_timestamp, get_git_commit_hash, load_json, save_json


def create_dataset_metadata(
    data_config: DataIngestionConfig,
    metadata_config: MetadataConfig,
    mlops_config: MLOpsConfig,
    preprocessed_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Save dataset lineage, class distribution, and split details."""
    class_names = sorted(preprocessed_df["class"].astype(str).unique().tolist())
    metadata = {
        "dataset_version": mlops_config.dataset_version,
        "raw_data_path": str(data_config.raw_data_dir),
        "preprocessed_data_path": str(data_config.preprocessed_data_path),
        "train_data_path": str(data_config.train_data_path),
        "validation_data_path": str(data_config.validation_data_path),
        "test_data_path": str(data_config.test_data_path),
        "created_at": get_current_timestamp(),
        "num_classes": len(class_names),
        "class_names": class_names,
        "total_documents": int(preprocessed_df["file_name"].nunique()),
        "total_chunks": int(len(preprocessed_df)),
        "train_documents": int(train_df["file_name"].nunique()),
        "validation_documents": int(validation_df["file_name"].nunique()) if validation_df is not None else None,
        "test_documents": int(test_df["file_name"].nunique()),
        "train_chunks": int(len(train_df)),
        "validation_chunks": int(len(validation_df)) if validation_df is not None else None,
        "test_chunks": int(len(test_df)),
        "docs_per_class": preprocessed_df.groupby("class")["file_name"].nunique().sort_index().to_dict(),
        "chunks_per_class": preprocessed_df["class"].value_counts().sort_index().to_dict(),
        "split_strategy": "document_level_split_by_file_name",
        "chunking_strategy": "overlapping_word_chunks",
        "chunk_size": data_config.chunk_size,
        "chunk_overlap": data_config.chunk_overlap,
        "balance_classes": data_config.balance_classes,
    }
    save_json(metadata, metadata_config.dataset_metadata_path)
    return metadata


def create_model_metadata(
    training_config: ModelTrainingConfig,
    metadata_config: MetadataConfig,
    mlops_config: MLOpsConfig,
    class_names: list[str],
) -> dict[str, Any]:
    """Save model location, label-space, and source-control details."""
    metadata = {
        "model_version": mlops_config.model_version,
        "model_name": training_config.model_name,
        "model_path": str(training_config.save_model_dir),
        "tokenizer_path": str(training_config.save_model_dir),
        "label_encoder_path": str(training_config.label_encoder_path),
        "num_labels": len(class_names),
        "class_names": class_names,
        "max_length": training_config.max_length,
        "created_at": get_current_timestamp(),
        "git_commit_hash": get_git_commit_hash(),
    }
    save_json(metadata, metadata_config.model_metadata_path)
    return metadata


def create_training_run_metadata(
    metadata_config: MetadataConfig,
    mlops_config: MLOpsConfig,
    training_config: ModelTrainingConfig,
    run_id: str,
    started_at: str,
    completed_at: str | None,
    training_status: str,
    device: str | None,
    mlflow_run_id: str | None = None,
) -> dict[str, Any]:
    """Save one structured record for a local or Colab training run."""
    metadata = {
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "model_version": mlops_config.model_version,
        "dataset_version": mlops_config.dataset_version,
        "started_at": started_at,
        "completed_at": completed_at,
        "epochs": training_config.num_epochs,
        "learning_rate": training_config.learning_rate,
        "train_batch_size": training_config.train_batch_size,
        "eval_batch_size": training_config.eval_batch_size,
        "max_length": training_config.max_length,
        "device": device,
        "training_status": training_status,
    }
    save_json(metadata, metadata_config.training_run_metadata_path)
    return metadata


def create_evaluation_metadata(
    metadata_config: MetadataConfig,
    mlops_config: MLOpsConfig,
    evaluation_config: EvaluationConfig,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Save a compact, query-friendly summary of evaluation results."""
    metadata = {
        "model_version": mlops_config.model_version,
        "dataset_version": mlops_config.dataset_version,
        "evaluated_at": get_current_timestamp(),
        "chunk_level_accuracy": metrics["chunk_level"]["accuracy"],
        "chunk_level_macro_f1": metrics["chunk_level"]["macro_f1"],
        "document_level_accuracy": metrics["document_level"]["accuracy"],
        "document_level_macro_f1": metrics["document_level"]["macro_f1"],
        "metrics_path": str(evaluation_config.metrics_path),
        "confusion_matrix_path": str(evaluation_config.confusion_matrix_path),
        "classification_report_available": bool(metrics["chunk_level"].get("classification_report")),
    }
    save_json(metadata, metadata_config.evaluation_metadata_path)
    return metadata


def update_version_history(
    metadata_config: MetadataConfig,
    mlops_config: MLOpsConfig,
    change_summary: str,
    metrics: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Append a version event while keeping prior history intact."""
    history: list[dict[str, Any]] = []
    if metadata_config.version_history_path.exists():
        existing = load_json(metadata_config.version_history_path)
        if isinstance(existing, list):
            history = existing
    metrics = metrics or {}
    history.append(
        {
            "version": mlops_config.model_version,
            "dataset_version": mlops_config.dataset_version,
            "model_version": mlops_config.model_version,
            "change_summary": change_summary,
            "created_at": get_current_timestamp(),
            "metrics": {
                "chunk_macro_f1": metrics.get("chunk_level", {}).get("macro_f1"),
                "document_macro_f1": metrics.get("document_level", {}).get("macro_f1"),
            },
        }
    )
    save_json(history, metadata_config.version_history_path)
    return history
