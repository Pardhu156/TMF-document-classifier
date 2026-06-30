"""Redis-backed manual review queue for Stage 6 filing decisions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

from src.config import AgenticFilingConfig, RAGConfig
from src.logger import logger


class ManualReviewQueue:
    """Queue pending manual reviews in Redis with an in-memory fallback for tests."""

    def __init__(self, config: AgenticFilingConfig | None = None, redis_url: str | None = None, client=None) -> None:
        self.config = config or AgenticFilingConfig()
        self.redis_url = redis_url or RAGConfig().redis_url
        self.client = client if client is not None else self._create_client()
        self._memory_items: dict[str, dict[str, Any]] = {}

    def _create_client(self):
        if not self.redis_url:
            return None
        try:
            import redis

            return redis.Redis.from_url(self.redis_url, decode_responses=True)
        except Exception as error:
            logger.warning("Manual review Redis queue unavailable; using in-memory fallback: %s", error)
            return None

    def _item_key(self, document_id: str | int) -> str:
        return f"manual_review:item:{document_id}"

    def push(self, item: dict[str, Any]) -> dict[str, Any]:
        payload = {
            **item,
            "status": item.get("status", "pending_review"),
            "timestamp": item.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        }
        document_id = str(payload["document_id"])
        if self.client is None:
            self._memory_items[document_id] = payload
            return payload

        payload_json = json.dumps(payload, default=str)
        self.client.set(self._item_key(document_id), payload_json)
        queued_ids = self.client.lrange(self.config.manual_review_queue_name, 0, -1)
        if document_id not in queued_ids:
            self.client.rpush(self.config.manual_review_queue_name, document_id)
        return payload

    def list_pending(self) -> list[dict[str, Any]]:
        if self.client is None:
            return list(self._memory_items.values())

        pending: list[dict[str, Any]] = []
        for document_id in self.client.lrange(self.config.manual_review_queue_name, 0, -1):
            payload_json = self.client.get(self._item_key(document_id))
            if payload_json:
                pending.append(json.loads(payload_json))
        return pending

    def remove(self, document_id: str | int) -> None:
        document_id = str(document_id)
        if self.client is None:
            self._memory_items.pop(document_id, None)
            return
        self.client.delete(self._item_key(document_id))
        self.client.lrem(self.config.manual_review_queue_name, 0, document_id)
