"""Redis-backed semantic cache for RAG question-answer pairs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from math import sqrt
from typing import Any
from uuid import uuid4

from src.config import RAGConfig
from src.logger import logger
from src.utils.hashing import calculate_text_hash


def document_scope(document_id: str | None = None, predicted_class: str | None = None) -> str:
    if document_id:
        return f"document:{document_id}"
    if predicted_class:
        return f"class:{predicted_class}"
    return "all_documents"


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


class RedisSemanticCache:
    """Semantic cache using Redis storage and scoped cosine matching."""

    def __init__(self, config: RAGConfig | None = None, client=None) -> None:
        self.config = config or RAGConfig()
        self.client = client or self._create_client()

    def _create_client(self):
        if not self.config.redis_configured:
            return None
        import redis

        return redis.Redis.from_url(self.config.redis_url, decode_responses=True)

    @property
    def is_enabled(self) -> bool:
        return self.config.semantic_cache_enabled and self.client is not None

    def get_exact(self, question: str, scope: str) -> dict[str, Any] | None:
        if not self.is_enabled:
            return None
        key = f"rag:answer:{scope}:{calculate_text_hash(question)}"
        payload = self.client.get(key)
        return json.loads(payload) if payload else None

    def search_semantic(self, question_embedding: list[float], scope: str) -> dict[str, Any] | None:
        if not self.is_enabled:
            return None
        best_payload = None
        best_score = 0.0
        for key in self.client.scan_iter("rag:cache:*"):
            payload_json = self.client.get(key)
            if not payload_json:
                continue
            payload = json.loads(payload_json)
            if payload.get("document_scope") != scope:
                continue
            score = cosine_similarity(question_embedding, payload.get("question_embedding", []))
            if score > best_score:
                best_score = score
                best_payload = payload
        if best_payload and best_score >= self.config.semantic_cache_threshold:
            best_payload["semantic_cache_score"] = best_score
            return best_payload
        return None

    def set(
        self,
        question: str,
        question_embedding: list[float],
        answer: str,
        scope: str,
        sources: list[dict[str, Any]],
        document_id: str | None = None,
        predicted_class: str | None = None,
    ) -> None:
        if not self.is_enabled:
            return
        cache_id = str(uuid4())
        payload = {
            "cache_id": cache_id,
            "question": question,
            "question_embedding": question_embedding,
            "answer": answer,
            "document_scope": scope,
            "document_id": document_id,
            "predicted_class": predicted_class,
            "source_chunks": sources,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ttl": self.config.semantic_cache_ttl_seconds,
            "model_name": self.config.gemini_generation_model,
            "embedding_model": self.config.gemini_embedding_model,
        }
        cache_key = f"rag:cache:{cache_id}"
        exact_key = f"rag:answer:{scope}:{calculate_text_hash(question)}"
        payload_json = json.dumps(payload, default=str)
        self.client.setex(cache_key, self.config.semantic_cache_ttl_seconds, payload_json)
        self.client.setex(exact_key, self.config.semantic_cache_ttl_seconds, payload_json)

    def set_document_status(self, document_id: str, status: str) -> None:
        if self.client is not None:
            try:
                self.client.set(f"doc:status:{document_id}", status)
            except Exception as error:
                logger.warning("Redis status update skipped for document_id=%s: %s", document_id, error)

    def get_document_status(self, document_id: str) -> str | None:
        if self.client is None:
            return None
        try:
            return self.client.get(f"doc:status:{document_id}")
        except Exception as error:
            logger.warning("Redis status read skipped for document_id=%s: %s", document_id, error)
            return None
