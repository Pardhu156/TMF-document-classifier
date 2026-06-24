"""Optional MLflow tracking that remains safe without DagsHub credentials."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # Keeps Stage 1 local execution safe before optional packages are installed.
    def load_dotenv(*args, **kwargs) -> bool:
        return False

from src.exception import CustomException
from src.logger import logger
from src.utils import flatten_dict


_mlflow_configured = False


def _get_mlflow():
    """Import MLflow only when tracking is requested."""
    import mlflow

    return mlflow


def setup_mlflow() -> bool:
    """Configure DagsHub-backed MLflow from .env, without exposing credentials."""
    global _mlflow_configured
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    required_values = {
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
        "MLFLOW_TRACKING_USERNAME": os.getenv("MLFLOW_TRACKING_USERNAME"),
        "MLFLOW_TRACKING_PASSWORD": os.getenv("MLFLOW_TRACKING_PASSWORD"),
        "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME"),
    }
    missing = [name for name, value in required_values.items() if not value]
    if missing:
        _mlflow_configured = False
        logger.warning("MLflow not configured. Skipping MLflow logging.")
        return False
    try:
        mlflow = _get_mlflow()
        os.environ["MLFLOW_TRACKING_USERNAME"] = required_values["MLFLOW_TRACKING_USERNAME"] or ""
        os.environ["MLFLOW_TRACKING_PASSWORD"] = required_values["MLFLOW_TRACKING_PASSWORD"] or ""
        mlflow.set_tracking_uri(required_values["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(required_values["MLFLOW_EXPERIMENT_NAME"])
        _mlflow_configured = True
        logger.info("MLflow tracking configured for experiment '%s'.", required_values["MLFLOW_EXPERIMENT_NAME"])
        return True
    except Exception as error:
        _mlflow_configured = False
        wrapped_error = CustomException(error, sys.exc_info())
        logger.warning("MLflow not configured. Skipping MLflow logging. Reason: %s", wrapped_error)
        return False


def start_mlflow_run(run_name: str | None = None):
    """Start and return an MLflow run, or None when tracking is unavailable."""
    try:
        if not setup_mlflow():
            return None
        mlflow = _get_mlflow()
        return mlflow.active_run() or mlflow.start_run(run_name=run_name)
    except Exception as error:
        logger.warning("MLflow not configured. Skipping MLflow logging. Reason: %s", error)
        return None


def log_params(params: dict[str, Any]) -> None:
    """Log parameter values when MLflow is configured."""
    if not _mlflow_configured:
        return
    try:
        _get_mlflow().log_params({str(key): str(value) for key, value in params.items() if value is not None})
    except Exception as error:
        logger.warning("Unable to log MLflow parameters: %s", error)


def log_metrics(metrics: dict[str, Any], prefix: str | None = None) -> None:
    """Flatten nested metrics and log only numeric values."""
    if not _mlflow_configured:
        return
    flattened = flatten_dict(metrics)
    numeric_metrics = {
        f"{prefix}_{key}" if prefix else key: float(value)
        for key, value in flattened.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not numeric_metrics:
        return
    try:
        _get_mlflow().log_metrics(numeric_metrics)
    except Exception as error:
        logger.warning("Unable to log MLflow metrics: %s", error)


def log_artifacts(artifact_paths: list[str]) -> None:
    """Log existing files as MLflow artifacts."""
    if not _mlflow_configured:
        return
    try:
        mlflow = _get_mlflow()
        for artifact_path in artifact_paths:
            path = Path(artifact_path)
            if path.is_file():
                mlflow.log_artifact(str(path))
    except Exception as error:
        logger.warning("Unable to log MLflow artifacts: %s", error)


def end_mlflow_run() -> None:
    """End the active MLflow run without affecting local pipeline completion."""
    if not _mlflow_configured:
        return
    try:
        mlflow = _get_mlflow()
        if mlflow.active_run() is not None:
            mlflow.end_run()
    except Exception as error:
        logger.warning("Unable to end MLflow run: %s", error)
