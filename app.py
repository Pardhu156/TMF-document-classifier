"""FastAPI application for serving the trained TMF classifier."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.config import DatabaseConfig, MetadataConfig, MLOpsConfig, ModelTrainingConfig
from src.exception import CustomException
from src.logger import logger
from src.pipeline.cloud_ingestion_pipeline import CloudIngestionPipeline
from src.pipeline.conditional_retraining_pipeline import ConditionalRetrainingPipeline
from src.predict import predict_text
from src.schemas import (
    DocumentVerificationRequest,
    DocumentVerificationResponse,
    FilePredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.utils import load_json


app = FastAPI(
    title="TMF Classifier",
    description="API service for classifying Trial Master File document text.",
    version="1.0.0",
)

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _safe_load_json(path: Path) -> dict[str, Any]:
    """Load a JSON metadata file, returning an empty dict when it is unavailable."""
    if not path.exists():
        return {}
    try:
        data = load_json(path)
        return data if isinstance(data, dict) else {}
    except Exception as error:
        logger.warning("Could not read metadata file %s: %s", path, error)
        return {}


def _get_model_info() -> dict[str, Any]:
    """Build model information from metadata, falling back to safe config defaults."""
    metadata_config = MetadataConfig()
    mlops_config = MLOpsConfig()
    training_config = ModelTrainingConfig()

    model_metadata = _safe_load_json(metadata_config.model_metadata_path)
    dataset_metadata = _safe_load_json(metadata_config.dataset_metadata_path)

    class_names = model_metadata.get("class_names") or dataset_metadata.get("class_names") or []
    return {
        "model_version": model_metadata.get("model_version", mlops_config.model_version),
        "dataset_version": dataset_metadata.get("dataset_version", mlops_config.dataset_version),
        "model_name": model_metadata.get("model_name", training_config.model_name),
        "number_of_classes": model_metadata.get("num_labels") or dataset_metadata.get("num_classes") or len(class_names),
        "class_names": class_names,
    }


@app.get("/")
def read_root() -> dict[str, str]:
    """Root endpoint used for a quick service check."""
    return {"project": "TMF Classifier", "status": "running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return API health status."""
    return {"status": "healthy"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Return model and dataset metadata when available."""
    return _get_model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict[str, float | str]:
    """Predict the TMF class for one text input."""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="text must be a non-empty string")
        return predict_text(request.text)
    except HTTPException:
        raise
    except CustomException as error:
        logger.exception("Prediction endpoint failed")
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        logger.exception("Prediction endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.post("/predict-file", response_model=FilePredictionResponse)
async def predict_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """Predict a complete uploaded TMF document using chunk-level aggregation."""
    safe_filename = Path(file.filename or "uploaded_document").name
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported formats are .pdf, .docx, and .txt.",
        )

    try:
        return await CloudIngestionPipeline().run(file)
    except HTTPException:
        raise
    except ValueError as error:
        logger.exception("File prediction validation failed for %s", safe_filename)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except CustomException as error:
        logger.exception("File prediction failed for %s", safe_filename)
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        logger.exception("File prediction failed for %s", safe_filename)
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.post("/retrain")
def retrain() -> dict[str, Any]:
    """Manually start the conditional retraining pipeline.

    This endpoint only checks for verified new training data and prepares the
    retraining metadata. It does not run an agentic workflow.
    """
    try:
        repository = None
        if DatabaseConfig().is_configured:
            from src.database.repository import TMFRepository

            repository = TMFRepository()
        return ConditionalRetrainingPipeline(repository=repository).run()
    except CustomException as error:
        logger.exception("Retraining endpoint failed")
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        logger.exception("Retraining endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.post("/documents/{doc_id}/verify", response_model=DocumentVerificationResponse)
def verify_document(doc_id: int, request: DocumentVerificationRequest) -> dict[str, Any]:
    """Manual admin review endpoint for verified retraining labels.

    This is intentionally simple for the current admin-only workflow. In a
    production app, replace this with RBAC/authenticated review permissions.
    """
    try:
        if not request.verified_label.strip():
            raise HTTPException(status_code=400, detail="verified_label must be a non-empty string")
        if not DatabaseConfig().is_configured:
            raise HTTPException(status_code=503, detail="PostgreSQL is not configured.")

        from src.database.repository import TMFRepository

        repository = TMFRepository()
        document = repository.verify_document(doc_id, request.verified_label.strip())
        if document is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found.")

        repository.save_audit_log(
            event_type="document_verified",
            entity_type="document",
            entity_id=str(doc_id),
            message="Document verified by admin review.",
            details={
                "verified_label": request.verified_label.strip(),
                "reviewer": request.reviewer,
                "notes": request.notes,
            },
        )
        logger.info("Document %s verified with label '%s'.", doc_id, request.verified_label.strip())
        return {
            "doc_id": document["doc_id"],
            "filename": document["filename"],
            "verified_label": document["verified_label"],
            "document_status": document["document_status"],
            "used_for_training": document["used_for_training"],
            "message": "Document verified. It is now eligible for conditional retraining.",
        }
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Document verification failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error
