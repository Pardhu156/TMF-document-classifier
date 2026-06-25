"""Pydantic schemas for the TMF Classifier API."""

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Request body for single-text prediction."""

    text: str


class PredictionResponse(BaseModel):
    """Response body returned by the prediction endpoint."""

    predicted_label: str
    confidence: float
    model_confidence: float
    vote_confidence: float
    margin_confidence: float
    requires_review: bool
    decision_status: str


class FilePredictionResponse(BaseModel):
    """Response body returned by the document-upload prediction endpoint."""

    filename: str
    predicted_label: str
    confidence: float
    model_confidence: float
    vote_confidence: float
    margin_confidence: float
    requires_review: bool
    decision_status: str
    num_chunks: int
    chunk_predictions: dict[str, int]
