"""Pydantic schemas for RAG endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RAGAskRequest(BaseModel):
    question: str
    document_id: str | None = None
    predicted_class: str | None = None
    file_name: str | None = None
    source_type: str | None = None
    verification_status: str | None = None
    scope: str = "master"


class RAGSource(BaseModel):
    document_id: str
    file_name: str | None = None
    page_no: int | None = None
    chunk_id: str
    chunk_index: int | None = None
    score: float | None = None


class RAGAskResponse(BaseModel):
    answer: str
    sources: list[RAGSource]
    cache_hit: bool
    cache_type: str
    retrieved_chunks: list[dict[str, Any]]
    latency_ms: int
    retrieval_scope: str | None = None
    matched_file_name: str | None = None
    clarification_required: bool = False
    candidate_files: list[str] = []


class RAGDocumentResponse(BaseModel):
    document_id: str
    file_name: str | None = None
    predicted_class: str | None = None
    source_type: str | None = None
    verification_status: str | None = None
    uploaded_by: str | None = None
    created_at: str | None = None


class RAGIndexMasterDataResponse(BaseModel):
    indexed_documents: int
    skipped_documents: int
    indexed_chunks: int
    source_type: str
    master_data_dir: str
    message: str


class RAGStatusResponse(BaseModel):
    document_id: str
    status: str


class RAGMetricsResponse(BaseModel):
    metrics: dict[str, Any]
