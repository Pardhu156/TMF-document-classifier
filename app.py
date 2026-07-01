"""FastAPI application for serving the trained TMF classifier."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from src.auth import (
    ROLE_ADMIN,
    ROLE_MANAGER,
    ROLE_USER,
    authenticate_user,
    create_access_token,
    get_auth_repository,
    get_current_user,
    hash_password,
    normalize_role,
    public_user,
    require_min_role,
    require_roles,
)
from src.config import AuthConfig, DatabaseConfig, MetadataConfig, MLOpsConfig, ModelTrainingConfig
from src.database.repository import TMFRepository
from src.exception import CustomException
from src.logger import logger
from src.agentic_filing.pipeline import AgenticTMFFilingPipeline
from src.pipeline.conditional_retraining_pipeline import ConditionalRetrainingPipeline
from src.predict import predict_text
from src.schemas import (
    DocumentVerificationRequest,
    DocumentVerificationResponse,
    FilePredictionResponse,
    AuthResponse,
    LoginRequest,
    ManualReviewCorrectionRequest,
    PredictionRequest,
    PredictionResponse,
    TrainingApprovalRequest,
    UserCreateRequest,
    UserResponse,
    UserUpdateRequest,
)
from src.rag.metrics import log_rag_metrics_to_mlflow, rag_metrics as rag_metrics_tracker
from src.rag.master_data_ingestion import MasterDataIngestionPipeline
from src.rag.schemas import (
    RAGAskRequest,
    RAGAskResponse,
    RAGDocumentResponse,
    RAGIndexMasterDataResponse,
    RAGMetricsResponse,
    RAGStatusResponse,
)
from src.rag.service import RAGService
from src.utils import load_json


app = FastAPI(
    title="TMF Classifier",
    description="API service for classifying Trial Master File document text.",
    version="1.0.0",
)

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}
AnyAuthenticatedUser = Annotated[dict, Depends(require_roles([ROLE_USER, ROLE_MANAGER, ROLE_ADMIN]))]
ManagerUser = Annotated[dict, Depends(require_min_role(ROLE_MANAGER))]
AdminUser = Annotated[dict, Depends(require_roles([ROLE_ADMIN]))]


def _dashboard_for_role(role: str) -> str:
    return {
        ROLE_USER: "User Dashboard",
        ROLE_MANAGER: "Manager Dashboard",
        ROLE_ADMIN: "Admin Dashboard",
    }[role]


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


@app.post("/auth/login", response_model=AuthResponse)
def login(request: LoginRequest, repository: Annotated[TMFRepository, Depends(get_auth_repository)]) -> dict[str, Any]:
    """Authenticate with email/password and return a bearer token."""
    user = authenticate_user(request.email, request.password, repository)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    config = AuthConfig()
    return {
        "access_token": create_access_token(user, config),
        "token_type": "bearer",
        "expires_in_minutes": config.access_token_expire_minutes,
        "user": public_user(user),
        "dashboard": _dashboard_for_role(user["role"]),
    }


@app.post("/auth/logout")
def logout(current_user: Annotated[dict, Depends(get_current_user)]) -> dict[str, str]:
    """Stateless logout hook for clients to discard their bearer token."""
    return {"message": f"Logged out {current_user['email']}. Discard the bearer token client-side."}


@app.get("/auth/me", response_model=UserResponse)
def auth_me(current_user: Annotated[dict, Depends(get_current_user)]) -> dict[str, Any]:
    """Return the currently authenticated user."""
    return public_user(current_user)


@app.get("/users", response_model=list[UserResponse])
def list_users(
    current_user: AdminUser,
    repository: Annotated[TMFRepository, Depends(get_auth_repository)],
) -> list[dict[str, Any]]:
    """List users. Admin-only."""
    return [public_user(user) for user in repository.list_users()]


@app.post("/users", response_model=UserResponse)
def create_user(
    request: UserCreateRequest,
    current_user: AdminUser,
    repository: Annotated[TMFRepository, Depends(get_auth_repository)],
) -> dict[str, Any]:
    """Create a user. Admin-only."""
    try:
        user = repository.create_user(
            {
                "name": request.name.strip(),
                "email": request.email.strip().lower(),
                "hashed_password": hash_password(request.password),
                "role": normalize_role(request.role),
                "is_active": request.is_active,
            }
        )
        return public_user(user)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Create user endpoint failed")
        raise HTTPException(status_code=400, detail="Could not create user.") from error


@app.patch("/users/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    request: UserUpdateRequest,
    current_user: AdminUser,
    repository: Annotated[TMFRepository, Depends(get_auth_repository)],
) -> dict[str, Any]:
    """Update a user profile, status, password, or role. Admin-only."""
    try:
        values = request.dict(exclude_unset=True)
        if "role" in values and values["role"] is not None:
            values["role"] = normalize_role(values["role"])
        if "password" in values:
            values["hashed_password"] = hash_password(values.pop("password"))
        user = repository.update_user(user_id, values)
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found.")
        return public_user(user)
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, current_user: AnyAuthenticatedUser) -> dict[str, float | str]:
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
async def predict_file(current_user: AnyAuthenticatedUser, file: UploadFile = File(...)) -> dict[str, Any]:
    """Predict a complete uploaded TMF document using chunk-level aggregation."""
    safe_filename = Path(file.filename or "uploaded_document").name
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Supported formats are .pdf, .docx, and .txt.",
        )

    try:
        return await AgenticTMFFilingPipeline().run(file, uploaded_by=current_user["email"])
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


@app.get("/documents/my-uploads")
def my_uploads(
    current_user: AnyAuthenticatedUser,
    repository: Annotated[TMFRepository, Depends(get_auth_repository)],
) -> dict[str, Any]:
    """Return document upload/status rows for the current user."""
    return {"items": repository.list_documents_by_uploader(current_user["email"])}


@app.get("/agentic/reviews")
def list_manual_reviews(current_user: ManagerUser) -> dict[str, Any]:
    """List pending Stage 6 manual-review items."""
    try:
        return {"items": AgenticTMFFilingPipeline().list_pending_reviews()}
    except Exception as error:
        logger.exception("Manual review list endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.post("/agentic/reviews/{doc_id}/submit")
def submit_manual_review(doc_id: int, request: ManualReviewCorrectionRequest, current_user: ManagerUser) -> dict[str, Any]:
    """Submit the corrected TMF class for a low-confidence document."""
    try:
        return AgenticTMFFilingPipeline().submit_manual_review(
            doc_id=doc_id,
            corrected_class=request.corrected_class,
            reviewer_id=request.reviewer_id,
            notes=request.notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Manual review submit endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.post("/agentic/training/{doc_id}/approve")
def approve_training(doc_id: int, request: TrainingApprovalRequest, current_user: AdminUser) -> dict[str, Any]:
    """Approve a finalized document for future training dataset export/retraining."""
    try:
        return AgenticTMFFilingPipeline().approve_for_training(
            doc_id=doc_id,
            approved=True,
            reviewer_id=request.reviewer_id,
            notes=request.notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Training approval endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.post("/agentic/training/{doc_id}/reject")
def reject_training(doc_id: int, request: TrainingApprovalRequest, current_user: AdminUser) -> dict[str, Any]:
    """Reject a finalized document from future training inclusion."""
    try:
        return AgenticTMFFilingPipeline().approve_for_training(
            doc_id=doc_id,
            approved=False,
            reviewer_id=request.reviewer_id,
            notes=request.notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Training rejection endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.post("/agentic/documents/{doc_id}/correct")
def correct_auto_filed_document(doc_id: int, request: ManualReviewCorrectionRequest, current_user: ManagerUser) -> dict[str, Any]:
    """Correct an already auto-filed document."""
    try:
        return AgenticTMFFilingPipeline().correct_auto_filed(
            doc_id=doc_id,
            corrected_class=request.corrected_class,
            reviewer_id=request.reviewer_id,
            notes=request.notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Auto-file correction endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.get("/agentic/metrics")
def agentic_metrics(current_user: AdminUser) -> dict[str, Any]:
    """Return Stage 6 filing/review/training-feedback metrics."""
    try:
        return {"metrics": AgenticTMFFilingPipeline().metrics()}
    except Exception as error:
        logger.exception("Agentic metrics endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.post("/retrain")
def retrain(current_user: AdminUser) -> dict[str, Any]:
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
def verify_document(doc_id: int, request: DocumentVerificationRequest, current_user: AdminUser) -> dict[str, Any]:
    """Manual admin review endpoint for verified retraining labels."""
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

        try:
            from src.rag.vector_store import PgVectorStore

            PgVectorStore().update_document_metadata(
                document_id=str(doc_id),
                predicted_class=request.verified_label.strip(),
                verification_status="verified",
            )
        except Exception as error:
            logger.warning("Document %s verified, but RAG metadata sync was skipped: %s", doc_id, error)

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


@app.post("/rag/ask", response_model=RAGAskResponse)
def rag_ask(request: RAGAskRequest, current_user: AnyAuthenticatedUser) -> dict[str, Any]:
    """Ask questions over already indexed uploaded documents."""
    try:
        if not RAGService.is_configured():
            raise HTTPException(status_code=503, detail="RAG is not configured. Set PostgreSQL and GEMINI_API_KEY.")
        return RAGService().ask(
            question=request.question,
            document_id=request.document_id,
            predicted_class=request.predicted_class,
            file_name=request.file_name,
            source_type=request.source_type,
            verification_status=request.verification_status,
            scope=request.scope,
        )
    except HTTPException:
        raise
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("RAG ask endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.get("/rag/documents", response_model=list[RAGDocumentResponse])
def rag_documents(current_user: AnyAuthenticatedUser) -> list[dict[str, Any]]:
    """Return all indexed RAG documents."""
    try:
        if not RAGService.is_configured():
            raise HTTPException(status_code=503, detail="RAG is not configured. Set PostgreSQL and GEMINI_API_KEY.")
        return RAGService().list_documents()
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("RAG documents endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.get("/rag/status/{document_id}", response_model=RAGStatusResponse)
def rag_status(document_id: str, current_user: AnyAuthenticatedUser) -> dict[str, str]:
    """Return indexing status for a document."""
    try:
        if not RAGService.is_configured():
            raise HTTPException(status_code=503, detail="RAG is not configured. Set PostgreSQL and GEMINI_API_KEY.")
        return {"document_id": document_id, "status": RAGService().status(document_id)}
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("RAG status endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.get("/rag/metrics", response_model=RAGMetricsResponse)
def rag_metrics(current_user: AdminUser) -> dict[str, Any]:
    """Return basic in-process RAG metrics."""
    try:
        metrics = rag_metrics_tracker.snapshot()
        log_rag_metrics_to_mlflow(
            params={"component": "rag", "source": "api_metrics_endpoint"},
            metrics=metrics,
        )
        return {"metrics": metrics}
    except Exception as error:
        logger.exception("RAG metrics endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.post("/rag/index-master-data", response_model=RAGIndexMasterDataResponse)
def rag_index_master_data(current_user: AdminUser) -> dict[str, Any]:
    """Manually index trusted MASTER_DATA files into pgvector."""
    try:
        if not MasterDataIngestionPipeline.is_configured():
            raise HTTPException(
                status_code=503,
                detail="Master-data indexing requires PostgreSQL and GEMINI_API_KEY.",
            )
        return MasterDataIngestionPipeline().run()
    except HTTPException:
        raise
    except CustomException as error:
        logger.exception("MASTER_DATA indexing endpoint failed")
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:
        logger.exception("MASTER_DATA indexing endpoint failed")
        raise HTTPException(status_code=500, detail=str(CustomException(error, sys.exc_info()))) from error


@app.get("/audit-logs")
def audit_logs(
    current_user: AdminUser,
    repository: Annotated[TMFRepository, Depends(get_auth_repository)],
    limit: int = 100,
) -> dict[str, Any]:
    """Return audit logs. Admin-only."""
    return {"items": repository.list_audit_logs(limit=max(1, min(limit, 500)))}
