"""Repository layer hiding SQLAlchemy details from pipelines."""

from __future__ import annotations

from typing import Iterable

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.database.db_connection import create_session_factory, session_scope
from src.database.models import AuditLog, Chunk, ChunkPrediction, Document, DocumentMetadata, ModelVersion, Prediction


def _as_dict(model) -> dict | None:
    """Serialize an ORM row to a plain dict."""
    if model is None:
        return None
    return {column.name: getattr(model, column.name) for column in model.__table__.columns}


class TMFRepository:
    """Database operations used by ingestion and retraining pipelines."""

    def __init__(self, session_factory: sessionmaker | None = None) -> None:
        self.session_factory = session_factory or create_session_factory()

    def _scope(self):
        return session_scope(self.session_factory)

    def save_document(self, document_data: dict) -> dict:
        with self._scope() as session:
            document = Document(**document_data)
            session.add(document)
            try:
                session.flush()
            except IntegrityError:
                session.rollback()
                if document_data.get("file_hash"):
                    existing = session.execute(
                        select(Document).where(Document.file_hash == document_data["file_hash"])
                    ).scalar_one_or_none()
                    if existing is not None:
                        return _as_dict(existing)
                raise
            return _as_dict(document)

    def get_document_by_hash(self, file_hash: str) -> dict | None:
        with self._scope() as session:
            document = session.execute(select(Document).where(Document.file_hash == file_hash)).scalar_one_or_none()
            return _as_dict(document)

    def get_document_by_id(self, doc_id: int) -> dict | None:
        with self._scope() as session:
            document = session.get(Document, doc_id)
            return _as_dict(document)

    def update_document_status(
        self,
        doc_id: int,
        document_status: str,
        verified_label: str | None = None,
        used_for_training: bool | None = None,
    ) -> dict | None:
        with self._scope() as session:
            document = session.get(Document, doc_id)
            if document is None:
                return None
            document.document_status = document_status
            if verified_label is not None:
                document.verified_label = verified_label
            if used_for_training is not None:
                document.used_for_training = used_for_training
            session.flush()
            return _as_dict(document)

    def verify_document(self, doc_id: int, verified_label: str) -> dict | None:
        """Set the human/admin verified label used by future retraining."""
        with self._scope() as session:
            document = session.get(Document, doc_id)
            if document is None:
                return None
            document.verified_label = verified_label
            document.document_status = "verified"
            document.used_for_training = False
            session.flush()
            return _as_dict(document)

    def save_chunks(self, doc_id: int, chunks: Iterable[dict]) -> list[dict]:
        with self._scope() as session:
            rows: list[dict] = []
            for chunk_data in chunks:
                chunk = Chunk(doc_id=doc_id, **chunk_data)
                session.add(chunk)
                session.flush()
                rows.append(_as_dict(chunk))
            return rows

    def save_prediction(self, doc_id: int, prediction_data: dict) -> dict:
        with self._scope() as session:
            prediction = Prediction(doc_id=doc_id, **prediction_data)
            session.add(prediction)
            session.flush()
            return _as_dict(prediction)

    def save_document_metadata(self, metadata_data: dict) -> dict:
        with self._scope() as session:
            metadata = DocumentMetadata(**metadata_data)
            session.add(metadata)
            session.flush()
            return _as_dict(metadata)

    def get_latest_document_metadata(self, doc_id: int) -> dict | None:
        with self._scope() as session:
            metadata = session.execute(
                select(DocumentMetadata)
                .where(DocumentMetadata.doc_id == doc_id)
                .order_by(DocumentMetadata.created_at.desc())
            ).scalars().first()
            return _as_dict(metadata)

    def update_latest_document_metadata(self, doc_id: int, values: dict) -> dict | None:
        with self._scope() as session:
            metadata = session.execute(
                select(DocumentMetadata)
                .where(DocumentMetadata.doc_id == doc_id)
                .order_by(DocumentMetadata.created_at.desc())
            ).scalars().first()
            if metadata is None:
                return None
            for key, value in values.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            session.flush()
            return _as_dict(metadata)

    def save_chunk_predictions(self, doc_id: int, chunk_predictions: Iterable[dict]) -> list[dict]:
        with self._scope() as session:
            rows: list[dict] = []
            for prediction_data in chunk_predictions:
                prediction = ChunkPrediction(doc_id=doc_id, **prediction_data)
                session.add(prediction)
                session.flush()
                rows.append(_as_dict(prediction))
            return rows

    def get_existing_prediction_by_doc_id(self, doc_id: int) -> dict | None:
        with self._scope() as session:
            prediction = session.execute(
                select(Prediction)
                .where(Prediction.doc_id == doc_id)
                .order_by(Prediction.created_at.desc())
            ).scalars().first()
            return _as_dict(prediction)

    def list_documents_by_status(self, statuses: list[str]) -> list[dict]:
        with self._scope() as session:
            documents = session.execute(
                select(Document)
                .where(Document.document_status.in_(statuses))
                .order_by(Document.upload_timestamp.desc())
            ).scalars().all()
            return [_as_dict(document) for document in documents]

    def get_new_verified_documents(self, limit: int | None = None) -> list[dict]:
        with self._scope() as session:
            query = (
                select(Document)
                .where(Document.verified_label.is_not(None))
                .where(Document.used_for_training.is_(False))
                .order_by(Document.upload_timestamp.asc())
            )
            if limit:
                query = query.limit(limit)
            return [_as_dict(document) for document in session.execute(query).scalars().all()]

    def mark_documents_used_for_training(self, doc_ids: list[int], dataset_version: str) -> int:
        if not doc_ids:
            return 0
        with self._scope() as session:
            result = session.execute(
                update(Document)
                .where(Document.doc_id.in_(doc_ids))
                .values(used_for_training=True, dataset_version=dataset_version)
            )
            return int(result.rowcount or 0)

    def save_model_version(self, model_version_data: dict) -> dict:
        with self._scope() as session:
            if model_version_data.get("is_active"):
                session.execute(update(ModelVersion).values(is_active=False))
            model_version = ModelVersion(**model_version_data)
            session.add(model_version)
            session.flush()
            return _as_dict(model_version)

    def get_active_model_version(self) -> dict | None:
        with self._scope() as session:
            model_version = session.execute(
                select(ModelVersion)
                .where(ModelVersion.is_active.is_(True))
                .order_by(ModelVersion.created_at.desc())
            ).scalar_one_or_none()
            return _as_dict(model_version)

    def save_audit_log(self, event_type: str, entity_type: str, entity_id: str | None, message: str, details: dict | None = None) -> dict:
        with self._scope() as session:
            audit_log = AuditLog(
                event_type=event_type,
                entity_type=entity_type,
                entity_id=entity_id,
                message=message,
                details=details,
            )
            session.add(audit_log)
            session.flush()
            return _as_dict(audit_log)

    def agentic_metrics(self) -> dict:
        with self._scope() as session:
            total_uploaded = int(session.execute(select(func.count(Document.doc_id))).scalar() or 0)
            status_rows = session.execute(
                select(Document.document_status, func.count(Document.doc_id)).group_by(Document.document_status)
            ).all()
            status_counts = {str(status): int(count) for status, count in status_rows}
            avg_confidence = session.execute(select(func.avg(Prediction.confidence))).scalar()
            correction_count = int(
                session.execute(
                    select(func.count(AuditLog.audit_id)).where(AuditLog.event_type == "auto_file_corrected")
                ).scalar()
                or 0
            )
            duplicate_count = int(
                session.execute(
                    select(func.count(AuditLog.audit_id)).where(AuditLog.event_type == "duplicate_detected")
                ).scalar()
                or 0
            )
            rag_ingested_count = int(
                session.execute(
                    select(func.count(DocumentMetadata.metadata_id)).where(DocumentMetadata.rag_ingested.is_(True))
                ).scalar()
                or 0
            )
            return {
                "total_uploaded_documents": total_uploaded,
                "auto_file_rate": status_counts.get("auto_filed", 0) / max(total_uploaded, 1),
                "manual_review_rate": status_counts.get("pending_review", 0) / max(total_uploaded, 1),
                "human_correction_rate": status_counts.get("human_corrected", 0) / max(total_uploaded, 1),
                "duplicate_detection_count": duplicate_count,
                "average_confidence": float(avg_confidence or 0.0),
                "documents_added_to_rag": rag_ingested_count,
                "pending_training_approval": status_counts.get("pending_training_approval", 0),
                "approved_for_training": status_counts.get("approved_for_training", 0),
                "rejected_for_training": status_counts.get("rejected_for_training", 0),
                "wrong_auto_file_correction_count": correction_count,
                "status_distribution": status_counts,
            }
