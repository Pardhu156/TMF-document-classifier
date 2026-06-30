"""Metadata scoping and filename matching for safer RAG retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
from typing import Any


VALID_SCOPES = {"document", "master", "verified", "all", "class"}
MASTER_SOURCE_TYPE = "MASTER_DATA"
TRAINING_SOURCE_TYPE = "TRAINING_DATA"
PREDICT_UPLOAD_SOURCE_TYPE = "PREDICT_UPLOAD"
RAG_UPLOAD_SOURCE_TYPE = "RAG_UPLOAD"


@dataclass
class RetrievalPlan:
    """Resolved retrieval filters before vector/FTS search."""

    filters: dict[str, Any]
    retrieval_scope: str
    matched_file_name: str | None = None
    clarification_required: bool = False
    candidate_files: list[str] = field(default_factory=list)


def normalize_scope(scope: str | None) -> str:
    normalized = (scope or "master").strip().lower()
    return normalized if normalized in VALID_SCOPES else "master"


def _normalize_text(value: str) -> str:
    stem = re.sub(r"\.[a-z0-9]+$", "", value.lower())
    return re.sub(r"[^a-z0-9]+", " ", stem).strip()


def _meaningful_tokens(value: str) -> list[str]:
    stopwords = {
        "file",
        "document",
        "doc",
        "pdf",
        "docx",
        "txt",
        "v",
        "version",
        "template",
    }
    return [
        token
        for token in _normalize_text(value).split()
        if len(token) > 1 and not token.isdigit() and token not in stopwords and not re.fullmatch(r"v\d+", token)
    ]


def fuzzy_file_matches(question: str, documents: list[dict[str, Any]], threshold: float = 0.72) -> tuple[str | None, list[str]]:
    """Return a high-confidence filename match or ambiguity candidates."""
    question_norm = _normalize_text(question)
    if not question_norm:
        return None, []

    scored: list[tuple[float, str]] = []
    for document in documents:
        file_name = str(document.get("file_name") or "")
        if not file_name:
            continue
        name_norm = _normalize_text(file_name)
        if not name_norm:
            continue
        score = SequenceMatcher(None, question_norm, name_norm).ratio()
        if name_norm in question_norm:
            score = max(score, 0.95)
        else:
            name_tokens = _meaningful_tokens(file_name)
            token_hits = sum(1 for token in name_tokens if token in question_norm)
            if token_hits:
                score = max(score, token_hits / max(len(name_tokens), 1))
        scored.append((score, file_name))

    scored.sort(reverse=True)
    if not scored or scored[0][0] < threshold:
        return None, []

    best_score, best_name = scored[0]
    close_matches = [name for score, name in scored if score >= threshold and best_score - score <= 0.08]
    if len(close_matches) > 1:
        return None, close_matches[:5]
    return best_name, []


def build_retrieval_plan(
    question: str,
    documents: list[dict[str, Any]],
    document_id: str | None = None,
    predicted_class: str | None = None,
    file_name: str | None = None,
    source_type: str | None = None,
    verification_status: str | None = None,
    scope: str | None = "master",
) -> RetrievalPlan:
    """Resolve safe metadata filters before retrieval touches vectors or FTS."""
    normalized_scope = normalize_scope(scope)
    filters: dict[str, Any] = {}

    if document_id:
        filters["document_id"] = str(document_id)
        return RetrievalPlan(filters=filters, retrieval_scope="document")

    if file_name:
        filters["file_name"] = file_name
        return RetrievalPlan(filters=filters, retrieval_scope="file", matched_file_name=file_name)

    matched_file_name, candidates = fuzzy_file_matches(question, documents)
    if candidates:
        return RetrievalPlan(
            filters={},
            retrieval_scope="clarification_required",
            clarification_required=True,
            candidate_files=candidates,
        )
    if matched_file_name:
        filters["file_name"] = matched_file_name
        return RetrievalPlan(filters=filters, retrieval_scope="file", matched_file_name=matched_file_name)

    if normalized_scope == "all":
        pass
    elif normalized_scope == "verified":
        filters["verification_status"] = "verified"
    elif normalized_scope == "class":
        if predicted_class:
            filters["predicted_class"] = predicted_class
        else:
            filters["source_type"] = MASTER_SOURCE_TYPE
            normalized_scope = "master"
    elif normalized_scope == "document":
        filters["source_type"] = MASTER_SOURCE_TYPE
        normalized_scope = "master"
    else:
        filters["source_type"] = MASTER_SOURCE_TYPE
        normalized_scope = "master"

    if predicted_class and normalized_scope != "class":
        filters["predicted_class"] = predicted_class
    if source_type:
        filters["source_type"] = source_type
    if verification_status:
        filters["verification_status"] = verification_status

    return RetrievalPlan(filters=filters, retrieval_scope=normalized_scope)
