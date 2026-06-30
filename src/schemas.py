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
    duplicate: bool = False
    document_status: str | None = None
    persistence_enabled: bool = False
    doc_id: int | None = None
    agentic_action: str | None = None
    final_class: str | None = None
    confidence_gap: float | None = None
    decision_confidence: float | None = None
    top_k_predictions: list[dict] | None = None
    metadata_path: str | None = None
    rag_ingestion_status: str | None = None


class DocumentVerificationRequest(BaseModel):
    """Manual admin review payload for marking a document as verified."""

    verified_label: str
    reviewer: str | None = "admin"
    notes: str | None = None


class DocumentVerificationResponse(BaseModel):
    """Response returned after a manual document verification."""

    doc_id: int
    filename: str
    verified_label: str
    document_status: str
    used_for_training: bool
    message: str


class ManualReviewCorrectionRequest(BaseModel):
    """Payload for Stage 6 manual correction and filing."""

    corrected_class: str
    reviewer_id: str | None = "admin"
    notes: str | None = None


class TrainingApprovalRequest(BaseModel):
    """Payload for approving/rejecting a document for future training."""

    reviewer_id: str | None = "admin"
    notes: str | None = None
