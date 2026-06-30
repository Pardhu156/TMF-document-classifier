"""Drop and recreate RAG pgvector tables after an embedding dimension change."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text

from src.rag.vector_store import PgVectorStore


def main() -> None:
    store = PgVectorStore()
    with store.engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS rag_chunks"))
        connection.execute(text("DROP TABLE IF EXISTS rag_documents"))
    store.ensure_schema()
    print("RAG vector store reset complete. Re-index MASTER_DATA before asking RAG questions.")


if __name__ == "__main__":
    main()
