"""PostgreSQL pgvector + full-text store for RAG chunks."""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine, text

from src.config import DatabaseConfig, RAGConfig
from src.logger import logger


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in embedding) + "]"


def _clean_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (filters or {}).items() if value not in (None, "")}


class PgVectorStore:
    """RAG vector store backed by PostgreSQL pgvector and FTS."""

    def __init__(self, database_config: DatabaseConfig | None = None, rag_config: RAGConfig | None = None, engine=None) -> None:
        self.database_config = database_config or DatabaseConfig()
        self.rag_config = rag_config or RAGConfig()
        self.engine = engine or self._create_engine()

    def _create_engine(self):
        if not self.database_config.sqlalchemy_url:
            raise ValueError("PostgreSQL is not configured for RAG vector store.")
        return create_engine(self.database_config.sqlalchemy_url, pool_pre_ping=True)

    def ensure_schema(self) -> None:
        """Create pgvector schema objects. Requires PostgreSQL with pgvector installed."""
        dimension = int(self.rag_config.embedding_dimension)
        with self.engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS rag_documents (
                        document_id TEXT PRIMARY KEY,
                        file_name TEXT,
                        predicted_class TEXT,
                        source_type TEXT DEFAULT 'PREDICT_UPLOAD',
                        verification_status TEXT DEFAULT 'unverified',
                        file_hash TEXT,
                        access_level TEXT DEFAULT 'User',
                        owner_id TEXT,
                        uploaded_by TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
            )
            connection.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        rag_chunk_pk BIGSERIAL PRIMARY KEY,
                        document_id TEXT REFERENCES rag_documents(document_id) ON DELETE CASCADE,
                        file_name TEXT,
                        predicted_class TEXT,
                        source_type TEXT DEFAULT 'PREDICT_UPLOAD',
                        verification_status TEXT DEFAULT 'unverified',
                        file_hash TEXT,
                        access_level TEXT DEFAULT 'User',
                        owner_id TEXT,
                        chunk_id TEXT UNIQUE,
                        page_no INTEGER,
                        chunk_index INTEGER,
                        chunk_text TEXT NOT NULL,
                        embedding VECTOR({dimension}) NOT NULL,
                        uploaded_by TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
            )
            for statement in (
                "ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'PREDICT_UPLOAD'",
                "ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS verification_status TEXT DEFAULT 'unverified'",
                "ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS file_hash TEXT",
                "ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS access_level TEXT DEFAULT 'User'",
                "ALTER TABLE rag_documents ADD COLUMN IF NOT EXISTS owner_id TEXT",
                "ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'PREDICT_UPLOAD'",
                "ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS verification_status TEXT DEFAULT 'unverified'",
                "ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS file_hash TEXT",
                "ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS access_level TEXT DEFAULT 'User'",
                "ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS owner_id TEXT",
            ):
                connection.execute(text(statement))
            connection.execute(text("UPDATE rag_documents SET access_level = 'User' WHERE access_level IS NULL"))
            connection.execute(text("UPDATE rag_chunks SET access_level = 'User' WHERE access_level IS NULL"))
            connection.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_rag_chunks_fts
                    ON rag_chunks USING GIN (to_tsvector('english', chunk_text));
                    """
                )
            )
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_documents_file_hash ON rag_documents (file_hash)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id ON rag_chunks (document_id)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_chunks_file_name ON rag_chunks (file_name)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_chunks_predicted_class ON rag_chunks (predicted_class)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_chunks_source_type ON rag_chunks (source_type)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_chunks_verification_status ON rag_chunks (verification_status)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_documents_access_level ON rag_documents (access_level)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_chunks_access_level ON rag_chunks (access_level)"))
            if dimension <= 2000:
                connection.execute(
                    text(
                        """
                        CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding
                        ON rag_chunks USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100);
                        """
                    )
                )
            else:
                logger.info(
                    "Skipping ivfflat index because pgvector supports vector indexes up to 2000 dimensions; current dimension=%d.",
                    dimension,
                )

    def upsert_document(
        self,
        document_id: str,
        file_name: str | None,
        predicted_class: str | None,
        uploaded_by: str | None,
        source_type: str = "PREDICT_UPLOAD",
        verification_status: str = "unverified",
        file_hash: str | None = None,
        access_level: str = "User",
        owner_id: str | None = None,
    ) -> None:
        self.ensure_schema()
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO rag_documents (
                        document_id, file_name, predicted_class, source_type,
                        verification_status, file_hash, access_level, owner_id, uploaded_by
                    )
                    VALUES (
                        :document_id, :file_name, :predicted_class, :source_type,
                        :verification_status, :file_hash, :access_level, :owner_id, :uploaded_by
                    )
                    ON CONFLICT (document_id) DO UPDATE SET
                        file_name = EXCLUDED.file_name,
                        predicted_class = EXCLUDED.predicted_class,
                        source_type = EXCLUDED.source_type,
                        verification_status = EXCLUDED.verification_status,
                        file_hash = EXCLUDED.file_hash,
                        access_level = EXCLUDED.access_level,
                        owner_id = EXCLUDED.owner_id,
                        uploaded_by = EXCLUDED.uploaded_by
                    """
                ),
                {
                    "document_id": str(document_id),
                    "file_name": file_name,
                    "predicted_class": predicted_class,
                    "source_type": source_type,
                    "verification_status": verification_status,
                    "file_hash": file_hash,
                    "access_level": access_level,
                    "owner_id": owner_id,
                    "uploaded_by": uploaded_by,
                },
            )

    def add_chunks(self, chunks: list[dict[str, Any]]) -> int:
        self.ensure_schema()
        if not chunks:
            return 0
        with self.engine.begin() as connection:
            for chunk in chunks:
                connection.execute(
                    text(
                        """
                        INSERT INTO rag_chunks (
                            document_id, file_name, predicted_class, source_type,
                            verification_status, file_hash, access_level, owner_id, chunk_id, page_no,
                            chunk_index, chunk_text, embedding, uploaded_by
                        )
                        VALUES (
                            :document_id, :file_name, :predicted_class, :source_type,
                            :verification_status, :file_hash, :access_level, :owner_id, :chunk_id, :page_no,
                            :chunk_index, :chunk_text, CAST(:embedding AS vector), :uploaded_by
                        )
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text,
                            embedding = EXCLUDED.embedding,
                            predicted_class = EXCLUDED.predicted_class,
                            source_type = EXCLUDED.source_type,
                            verification_status = EXCLUDED.verification_status,
                            file_hash = EXCLUDED.file_hash,
                            access_level = EXCLUDED.access_level,
                            owner_id = EXCLUDED.owner_id
                        """
                    ),
                    {
                        "source_type": "PREDICT_UPLOAD",
                        "verification_status": "unverified",
                        "file_hash": None,
                        "access_level": "User",
                        "owner_id": None,
                        **chunk,
                        "embedding": _vector_literal(chunk["embedding"]),
                    },
                )
        return len(chunks)

    def _build_metadata_where(
        self,
        filters: dict[str, Any] | None,
        params: dict[str, Any],
        extra_filters: list[str] | None = None,
    ) -> str:
        clauses = list(extra_filters or [])
        for key, value in _clean_filters(filters).items():
            if key not in {"document_id", "predicted_class", "file_name", "source_type", "verification_status", "access_level", "owner_id", "uploaded_by"}:
                continue
            if isinstance(value, (list, tuple, set)):
                values = [str(item) for item in value if item not in (None, "")]
                if not values:
                    continue
                param_names = []
                for index, item in enumerate(values):
                    param_name = f"{key}_{index}"
                    param_names.append(f":{param_name}")
                    params[param_name] = item
                clauses.append(f"{key} IN ({', '.join(param_names)})")
            else:
                clauses.append(f"{key} = :{key}")
                params[key] = str(value)
        return "WHERE " + " AND ".join(clauses) if clauses else ""

    def semantic_search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_id: str | None = None,
        predicted_class: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        params: dict[str, Any] = {"embedding": _vector_literal(query_embedding), "top_k": top_k}
        metadata_filters = _clean_filters(filters)
        if document_id:
            metadata_filters["document_id"] = str(document_id)
        if predicted_class:
            metadata_filters["predicted_class"] = predicted_class
        where_clause = self._build_metadata_where(metadata_filters, params)
        query = text(
            f"""
            SELECT document_id, file_name, predicted_class, source_type, verification_status,
                   access_level, owner_id,
                   chunk_id, page_no, chunk_index,
                   chunk_text, 1 - (embedding <=> CAST(:embedding AS vector)) AS semantic_score
            FROM rag_chunks
            {where_clause}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
            """
        )
        with self.engine.begin() as connection:
            return [dict(row._mapping) for row in connection.execute(query, params)]

    def keyword_search(
        self,
        question: str,
        top_k: int,
        document_id: str | None = None,
        predicted_class: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_schema()
        where_filters = ["to_tsvector('english', chunk_text) @@ plainto_tsquery('english', :question)"]
        params: dict[str, Any] = {"question": question, "top_k": top_k}
        metadata_filters = _clean_filters(filters)
        if document_id:
            metadata_filters["document_id"] = str(document_id)
        if predicted_class:
            metadata_filters["predicted_class"] = predicted_class
        where_clause = self._build_metadata_where(metadata_filters, params, extra_filters=where_filters)
        query = text(
            f"""
            SELECT document_id, file_name, predicted_class, source_type, verification_status,
                   access_level, owner_id,
                   chunk_id, page_no, chunk_index,
                   chunk_text,
                   ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', :question)) AS keyword_score
            FROM rag_chunks
            {where_clause}
            ORDER BY keyword_score DESC
            LIMIT :top_k
            """
        )
        with self.engine.begin() as connection:
            return [dict(row._mapping) for row in connection.execute(query, params)]

    def list_documents(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.ensure_schema()
        params: dict[str, Any] = {}
        where_clause = self._build_metadata_where(_clean_filters(filters), params)
        with self.engine.begin() as connection:
            return [
                dict(row._mapping)
                for row in connection.execute(
                    text(
                        f"""
                        SELECT document_id, file_name, predicted_class, source_type,
                               verification_status, access_level, owner_id, uploaded_by, created_at
                        FROM rag_documents
                        {where_clause}
                        ORDER BY created_at DESC
                        """
                    ),
                    params,
                )
            ]

    def get_document_by_file_hash(self, file_hash: str, source_type: str | None = None) -> dict[str, Any] | None:
        self.ensure_schema()
        params: dict[str, Any] = {"file_hash": file_hash}
        extra_filters = ["file_hash = :file_hash"]
        metadata_filters = {"source_type": source_type} if source_type else {}
        where_clause = self._build_metadata_where(metadata_filters, params, extra_filters=extra_filters)
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    f"""
                    SELECT document_id, file_name, predicted_class, source_type,
                           verification_status, file_hash, access_level, owner_id, uploaded_by, created_at
                    FROM rag_documents
                    {where_clause}
                    LIMIT 1
                    """
                ),
                params,
            ).first()
            return dict(row._mapping) if row else None

    def update_document_metadata(
        self,
        document_id: str,
        predicted_class: str | None = None,
        verification_status: str | None = None,
        access_level: str | None = None,
        owner_id: str | None = None,
    ) -> None:
        self.ensure_schema()
        updates = []
        params: dict[str, Any] = {"document_id": str(document_id)}
        if predicted_class:
            updates.append("predicted_class = :predicted_class")
            params["predicted_class"] = predicted_class
        if verification_status:
            updates.append("verification_status = :verification_status")
            params["verification_status"] = verification_status
        if access_level:
            updates.append("access_level = :access_level")
            params["access_level"] = access_level
        if owner_id:
            updates.append("owner_id = :owner_id")
            params["owner_id"] = owner_id
        if not updates:
            return
        update_clause = ", ".join(updates)
        with self.engine.begin() as connection:
            connection.execute(
                text(f"UPDATE rag_documents SET {update_clause} WHERE document_id = :document_id"),
                params,
            )
            connection.execute(
                text(f"UPDATE rag_chunks SET {update_clause} WHERE document_id = :document_id"),
                params,
            )

    def document_status(self, document_id: str) -> str:
        self.ensure_schema()
        with self.engine.begin() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM rag_documents WHERE document_id = :document_id"),
                {"document_id": str(document_id)},
            ).scalar_one_or_none()
            return "indexed" if exists else "not_found"
