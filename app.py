"""FastAPI application for serving the trained TMF classifier."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.config import DataIngestionConfig, MetadataConfig, MLOpsConfig, ModelTrainingConfig
from src.data_preprocessing import chunk_document_text, clean_document_text
from src.exception import CustomException
from src.file_utils import extract_text_from_docx, extract_text_from_pdf, extract_text_from_txt
from src.logger import logger
from src.predict import predict_text
from src.schemas import FilePredictionResponse, PredictionRequest, PredictionResponse
from src.utils import document_confidence_summary, load_json


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


def _extract_uploaded_file_text(file_path: Path) -> str:
    """Extract text from one supported uploaded document."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return extract_text_from_docx(file_path)
    if suffix == ".txt":
        return extract_text_from_txt(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _predict_document_chunks(filename: str, text: str) -> dict[str, Any]:
    """Chunk full document text, predict each chunk, and aggregate by majority vote."""
    data_config = DataIngestionConfig()
    cleaned_text = clean_document_text(text)
    if not cleaned_text.strip():
        raise ValueError("Uploaded file does not contain extractable text.")

    chunks = chunk_document_text(
        cleaned_text,
        chunk_size=data_config.chunk_size,
        chunk_overlap=data_config.chunk_overlap,
    )
    if not chunks:
        raise ValueError("Uploaded file did not produce any prediction chunks.")

    chunk_results = [predict_text(chunk) for chunk in chunks]
    class_order = _get_model_info().get("class_names") or sorted(
        {str(result["predicted_label"]) for result in chunk_results}
    )
    confidence_summary = document_confidence_summary(
        chunk_results,
        [str(label) for label in class_order],
    )

    logger.info(
        "File prediction complete for %s: total_chunks=%d, winning_votes=%d, "
        "second_best_votes=%d, final_confidence=%.4f, decision_status=%s",
        filename,
        confidence_summary["num_chunks"],
        confidence_summary["winning_votes"],
        confidence_summary["second_best_votes"],
        confidence_summary["confidence"],
        confidence_summary["decision_status"],
    )
    return {
        "filename": filename,
        "predicted_label": confidence_summary["predicted_label"],
        "confidence": confidence_summary["confidence"],
        "model_confidence": confidence_summary["model_confidence"],
        "vote_confidence": confidence_summary["vote_confidence"],
        "margin_confidence": confidence_summary["margin_confidence"],
        "requires_review": confidence_summary["requires_review"],
        "decision_status": confidence_summary["decision_status"],
        "num_chunks": confidence_summary["num_chunks"],
        "chunk_predictions": confidence_summary["chunk_predictions"],
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
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with tempfile.TemporaryDirectory(prefix="tmf_upload_") as temp_dir:
            temp_path = Path(temp_dir) / safe_filename
            temp_path.write_bytes(contents)
            extracted_text = _extract_uploaded_file_text(temp_path)

        logger.info(
            "Uploaded file %s produced %d extracted characters.",
            safe_filename,
            len(extracted_text),
        )
        return _predict_document_chunks(safe_filename, extracted_text)
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
