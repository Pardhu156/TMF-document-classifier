"""Repository layer hiding SQLAlchemy details from pipelines."""

from __future__ import annotations

from typing import Iterable

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from src.database.db_connection import create_session_factory, session_scope
from src.database.models import AuditLog, Chunk, ChunkPrediction, Document, ModelVersion, Prediction


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
