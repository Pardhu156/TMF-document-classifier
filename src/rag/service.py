"""RAG orchestration service used by FastAPI endpoints and ingestion."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from src.config import DatabaseConfig, RAGConfig
from src.logger import logger
from src.rag.access_control import ACCESS_PERMISSION_MESSAGE, allowed_access_levels_for_role, apply_access_filters, normalize_access_level
from src.rag.embeddings import GeminiEmbeddingClient
from src.rag.generator import GeminiAnswerGenerator, NOT_FOUND_ANSWER
from src.rag.metrics import log_rag_metrics_to_mlflow, rag_metrics
from src.rag.retrieval_policy import build_retrieval_plan
from src.rag.retriever import HybridRetriever
from src.rag.semantic_cache import RedisSemanticCache, document_scope
from src.rag.vector_store import PgVectorStore


def _source_from_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": str(chunk.get("document_id")),
        "file_name": chunk.get("file_name"),
        "page_no": chunk.get("page_no"),
        "chunk_id": str(chunk.get("chunk_id")),
        "chunk_index": chunk.get("chunk_index"),
        "score": chunk.get("hybrid_score") or chunk.get("semantic_score") or chunk.get("keyword_score"),
    }


def _document_matches_filters(document: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Return True when a document satisfies exact/list metadata filters."""
    for key, value in filters.items():
        if value in (None, ""):
            continue
        document_value = document.get(key)
        if isinstance(value, (list, tuple, set)):
            if str(document_value) not in {str(item) for item in value}:
                return False
        elif str(document_value) != str(value):
            return False
    return True


class RAGService:
    """Question-answering service combining retrieval, cache, and generation."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        vector_store: PgVectorStore | None = None,
        embedding_client: GeminiEmbeddingClient | None = None,
        cache: RedisSemanticCache | None = None,
        generator: GeminiAnswerGenerator | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.vector_store = vector_store or PgVectorStore(rag_config=self.config)
        self.embedding_client = embedding_client or GeminiEmbeddingClient(self.config)
        self.cache = cache or RedisSemanticCache(self.config)
        self.generator = generator or GeminiAnswerGenerator(self.config)
        self.retriever = HybridRetriever(self.vector_store, self.embedding_client, config=self.config)

    @classmethod
    def is_configured(cls) -> bool:
        return DatabaseConfig().is_configured and RAGConfig().gemini_configured

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        predicted_class: str | None = None,
        file_name: str | None = None,
        source_type: str | None = None,
        verification_status: str | None = None,
        scope: str = "master",
        current_user: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        start = perf_counter()
        if not question.strip():
            raise ValueError("question must be a non-empty string")
        current_user = current_user or {"id": None, "role": "Admin"}
        user_id = current_user.get("id")
        role = normalize_access_level(str(current_user.get("role") or "User"))
        allowed_access_levels = allowed_access_levels_for_role(role)
        all_documents = self.list_documents()
        documents = [
            document
            for document in all_documents
            if normalize_access_level(document.get("access_level")) in allowed_access_levels
        ]
        filtered_document_ids = [
            str(document.get("document_id"))
            for document in all_documents
            if normalize_access_level(document.get("access_level")) not in allowed_access_levels
        ]
        logger.info(
            "RAG RBAC context: user_id=%s role=%s allowed_access_levels=%s authorized_document_ids=%s filtered_document_ids=%s",
            user_id,
            role,
            allowed_access_levels,
            [str(document.get("document_id")) for document in documents],
            filtered_document_ids,
        )
        retrieval_plan = build_retrieval_plan(
            question=question,
            documents=documents,
            document_id=document_id,
            predicted_class=predicted_class,
            file_name=file_name,
            source_type=source_type,
            verification_status=verification_status,
            scope=scope,
        )
        access_filters = apply_access_filters(retrieval_plan.filters, role)
        requested_documents = [
            document for document in all_documents if _document_matches_filters(document, retrieval_plan.filters)
        ]
        authorized_requested_documents = [
            document for document in requested_documents if _document_matches_filters(document, access_filters)
        ]
        if not documents or (requested_documents and not authorized_requested_documents):
            latency_ms = int((perf_counter() - start) * 1000)
            denied_ids = [str(document.get("document_id")) for document in requested_documents] or filtered_document_ids
            logger.warning(
                "RAG access denied: user_id=%s role=%s denied_document_ids=%s filters=%s",
                user_id,
                role,
                denied_ids,
                retrieval_plan.filters,
            )
            rag_metrics.record(latency_ms, "none")
            return {
                "answer": ACCESS_PERMISSION_MESSAGE,
                "sources": [],
                "cache_hit": False,
                "cache_type": "none",
                "retrieved_chunks": [],
                "latency_ms": latency_ms,
                "retrieval_scope": retrieval_plan.retrieval_scope,
                "matched_file_name": retrieval_plan.matched_file_name,
                "clarification_required": False,
                "candidate_files": [],
            }
        if retrieval_plan.clarification_required:
            latency_ms = int((perf_counter() - start) * 1000)
            rag_metrics.record(latency_ms, "none")
            return {
                "answer": "Multiple documents look relevant. Please specify the exact document_id or file_name.",
                "sources": [],
                "cache_hit": False,
                "cache_type": "none",
                "retrieved_chunks": [],
                "latency_ms": latency_ms,
                "retrieval_scope": retrieval_plan.retrieval_scope,
                "matched_file_name": None,
                "clarification_required": True,
                "candidate_files": retrieval_plan.candidate_files,
            }

        cache_scope = document_scope(
            access_filters.get("document_id"),
            access_filters.get("predicted_class") or predicted_class,
        )
        cache_scope = f"{cache_scope}|scope:{retrieval_plan.retrieval_scope}|file:{retrieval_plan.matched_file_name or access_filters.get('file_name') or ''}|source:{access_filters.get('source_type') or ''}|verification:{access_filters.get('verification_status') or ''}|access:{','.join(allowed_access_levels)}|role:{role}"
        query_embedding = self.embedding_client.embed_query(question)

        exact_cache = self.cache.get_exact(question, cache_scope)
        if exact_cache:
            latency_ms = int((perf_counter() - start) * 1000)
            rag_metrics.record(latency_ms, "exact")
            return {
                "answer": exact_cache["answer"],
                "sources": exact_cache.get("source_chunks", []),
                "cache_hit": True,
                "cache_type": "exact",
                "retrieved_chunks": [],
                "latency_ms": latency_ms,
                "retrieval_scope": retrieval_plan.retrieval_scope,
                "matched_file_name": retrieval_plan.matched_file_name,
                "clarification_required": False,
                "candidate_files": [],
            }

        semantic_cache = self.cache.search_semantic(query_embedding, cache_scope)
        if semantic_cache:
            latency_ms = int((perf_counter() - start) * 1000)
            rag_metrics.record(latency_ms, "semantic")
            return {
                "answer": semantic_cache["answer"],
                "sources": semantic_cache.get("source_chunks", []),
                "cache_hit": True,
                "cache_type": "semantic",
                "retrieved_chunks": [],
                "latency_ms": latency_ms,
                "retrieval_scope": retrieval_plan.retrieval_scope,
                "matched_file_name": retrieval_plan.matched_file_name,
                "clarification_required": False,
                "candidate_files": [],
            }

        retrieval_start = perf_counter()
        chunks, _ = self.retriever.retrieve(
            question,
            document_id=document_id,
            predicted_class=predicted_class,
            query_embedding=query_embedding,
            filters=access_filters,
        )
        retrieval_latency_ms = int((perf_counter() - retrieval_start) * 1000)
        generation_start = perf_counter()
        answer = self.generator.generate(question, chunks) if chunks else NOT_FOUND_ANSWER
        generation_latency_ms = int((perf_counter() - generation_start) * 1000)
        sources = [_source_from_chunk(chunk) for chunk in chunks]
        self.cache.set(
            question=question,
            question_embedding=query_embedding,
            answer=answer,
            scope=cache_scope,
            sources=sources,
            document_id=retrieval_plan.filters.get("document_id") or document_id,
            predicted_class=retrieval_plan.filters.get("predicted_class") or predicted_class,
        )
        latency_ms = int((perf_counter() - start) * 1000)
        rag_metrics.record(
            latency_ms,
            "none",
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
        )
        logger.info(
            "RAG answer generated: scope=%s, filters=%s, chunks=%d, latency_ms=%d",
            retrieval_plan.retrieval_scope,
            access_filters,
            len(chunks),
            latency_ms,
        )
        return {
            "answer": answer,
            "sources": sources,
            "cache_hit": False,
            "cache_type": "none",
            "retrieved_chunks": chunks,
            "latency_ms": latency_ms,
            "retrieval_scope": retrieval_plan.retrieval_scope,
            "matched_file_name": retrieval_plan.matched_file_name,
            "clarification_required": False,
            "candidate_files": [],
        }

    def list_documents(self, current_user: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        documents = self.vector_store.list_documents()
        if current_user is not None:
            allowed_levels = set(allowed_access_levels_for_role(str(current_user.get("role") or "User")))
            documents = [
                document
                for document in documents
                if normalize_access_level(document.get("access_level")) in allowed_levels
            ]
        for document in documents:
            created_at = document.get("created_at")
            if hasattr(created_at, "isoformat"):
                document["created_at"] = created_at.isoformat()
        return documents

    def status(self, document_id: str) -> str:
        return self.cache.get_document_status(document_id) or self.vector_store.document_status(document_id)

    def metrics(self) -> dict[str, Any]:
        metrics = rag_metrics.snapshot()
        log_rag_metrics_to_mlflow(
            params={"component": "rag", "source": "rag_service"},
            metrics=metrics,
        )
        return metrics


class RAGIndexer:
    """Indexes uploaded document chunks into pgvector."""

    def __init__(
        self,
        config: RAGConfig | None = None,
        vector_store: PgVectorStore | None = None,
        embedding_client: GeminiEmbeddingClient | None = None,
        cache: RedisSemanticCache | None = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.vector_store = vector_store or PgVectorStore(rag_config=self.config)
        self.embedding_client = embedding_client or GeminiEmbeddingClient(self.config)
        self.cache = cache or RedisSemanticCache(self.config)

    @classmethod
    def is_configured(cls) -> bool:
        return DatabaseConfig().is_configured and RAGConfig().gemini_configured

    def index_document(
        self,
        document_id: str,
        file_name: str,
        predicted_class: str | None,
        chunk_texts: list[str],
        uploaded_by: str | None = None,
        source_type: str = "PREDICT_UPLOAD",
        verification_status: str = "unverified",
        file_hash: str | None = None,
        access_level: str = "User",
        owner_id: str | None = None,
    ) -> int:
        try:
            self.cache.set_document_status(document_id, "uploaded")
            self.cache.set_document_status(document_id, "extracted")
            self.cache.set_document_status(document_id, "chunked")
            embeddings = self.embedding_client.embed_texts(chunk_texts)
            self.cache.set_document_status(document_id, "embedded")
            self.vector_store.upsert_document(
                document_id=document_id,
                file_name=file_name,
                predicted_class=predicted_class,
                uploaded_by=uploaded_by,
                source_type=source_type,
                verification_status=verification_status,
                file_hash=file_hash,
                access_level=normalize_access_level(access_level),
                owner_id=owner_id,
            )
            chunks = [
                {
                    "document_id": str(document_id),
                    "file_name": file_name,
                    "predicted_class": predicted_class,
                    "source_type": source_type,
                    "verification_status": verification_status,
                    "file_hash": file_hash,
                    "access_level": normalize_access_level(access_level),
                    "owner_id": owner_id,
                    "chunk_id": f"{document_id}:{index}",
                    "page_no": None,
                    "chunk_index": index,
                    "chunk_text": chunk_text,
                    "embedding": embedding,
                    "uploaded_by": uploaded_by,
                }
                for index, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings))
            ]
            indexed = self.vector_store.add_chunks(chunks)
            self.cache.set_document_status(document_id, "indexed")
            logger.info("Indexed %d RAG chunks for document_id=%s", indexed, document_id)
            return indexed
        except Exception:
            self.cache.set_document_status(document_id, "failed")
            raise
