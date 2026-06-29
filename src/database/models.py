"""SQLAlchemy ORM models for Stage 4 cloud persistence."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp for DB rows."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base class for ORM models."""


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    uploaded_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    raw_document_s3_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_text_s3_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    document_status: Mapped[str] = mapped_column(String(64), default="new", nullable=False)
    verified_label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    used_for_training: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    dataset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("doc_id", "chunk_hash", name="uq_chunks_doc_hash"),)

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.doc_id"), index=True, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    chunk_text_s3_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")
    chunk_predictions: Mapped[list["ChunkPrediction"]] = relationship(back_populates="chunk", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.doc_id"), index=True, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    vote_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    margin_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    requires_review: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    decision_status: Mapped[str] = mapped_column(String(64), nullable=False)
    num_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_predictions: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="predictions")


class ChunkPrediction(Base):
    __tablename__ = "chunk_predictions"

    chunk_prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.chunk_id"), index=True, nullable=False)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.doc_id"), index=True, nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)

    chunk: Mapped["Chunk"] = relationship(back_populates="chunk_predictions")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    model_version_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_artifact_s3_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    audit_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
