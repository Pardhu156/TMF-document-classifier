"""Hybrid semantic + keyword retrieval for RAG."""

from __future__ import annotations

from typing import Any

from src.config import RAGConfig
from src.rag.embeddings import GeminiEmbeddingClient
from src.rag.reranker import Reranker
from src.rag.vector_store import PgVectorStore


class HybridRetriever:
    """Retrieve chunks with pgvector semantic search and PostgreSQL FTS."""

    def __init__(
        self,
        vector_store: PgVectorStore | None = None,
        embedding_client: GeminiEmbeddingClient | None = None,
        reranker: Reranker | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.vector_store = vector_store or PgVectorStore(rag_config=self.config)
        self.embedding_client = embedding_client or GeminiEmbeddingClient(self.config)
        self.reranker = reranker or Reranker(self.config)

    def retrieve(
        self,
        question: str,
        document_id: str | None = None,
        predicted_class: str | None = None,
        query_embedding: list[float] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[float]]:
        query_embedding = query_embedding or self.embedding_client.embed_query(question)
        semantic_results = self.vector_store.semantic_search(
            query_embedding,
            top_k=self.config.semantic_top_k,
            document_id=document_id,
            predicted_class=predicted_class,
            filters=filters,
        )
        keyword_results = self.vector_store.keyword_search(
            question,
            top_k=self.config.keyword_top_k,
            document_id=document_id,
            predicted_class=predicted_class,
            filters=filters,
        )
        merged = self._merge_results(semantic_results, keyword_results)
        reranked = self.reranker.rerank(question, merged)
        return reranked[: self.config.final_top_k], query_embedding

    def _merge_results(self, semantic_results: list[dict[str, Any]], keyword_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_chunk_id: dict[str, dict[str, Any]] = {}
        for result in semantic_results:
            chunk_id = str(result["chunk_id"])
            semantic_score = float(result.get("semantic_score") or 0.0)
            by_chunk_id[chunk_id] = {
                **result,
                "semantic_score": semantic_score,
                "keyword_score": 0.0,
                "hybrid_score": semantic_score,
            }
        for result in keyword_results:
            chunk_id = str(result["chunk_id"])
            keyword_score = float(result.get("keyword_score") or 0.0)
            if chunk_id in by_chunk_id:
                by_chunk_id[chunk_id]["keyword_score"] = keyword_score
                by_chunk_id[chunk_id]["hybrid_score"] = float(by_chunk_id[chunk_id].get("semantic_score") or 0.0) + keyword_score
            else:
                by_chunk_id[chunk_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "keyword_score": keyword_score,
                    "hybrid_score": keyword_score,
                }
        return sorted(by_chunk_id.values(), key=lambda item: float(item.get("hybrid_score") or 0.0), reverse=True)
