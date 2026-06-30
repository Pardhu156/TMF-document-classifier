"""Reranking interface for RAG retrieval results."""

from __future__ import annotations

from typing import Any

from src.config import RAGConfig


class Reranker:
    """Lightweight reranker facade.

    Gemini/cross-encoder reranking can be plugged in later. For now, when
    reranking is disabled or unavailable, hybrid score ranking is used.
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()

    def rerank(self, question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.config.reranker_enabled:
            return chunks
        # Placeholder for future Gemini/cross-encoder reranking. Keeping this
        # explicit avoids silently pretending a true reranker is active.
        return chunks
