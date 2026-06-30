"""Seed PostgreSQL/pgvector with DVC-managed MASTER_DATA when RAG tables are empty.

This script is designed for Docker Compose startup. It is idempotent:
- if rag_chunks already contains rows, it exits without ingesting;
- if rag_chunks is empty, it runs the existing MasterDataIngestionPipeline;
- it never deletes or rewrites existing vector data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from sqlalchemy import text
from sqlalchemy.exc import OperationalError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAGConfig
from src.logger import logger
from src.rag.master_data_ingestion import MasterDataIngestionPipeline, SUPPORTED_MASTER_EXTENSIONS
from src.rag.vector_store import PgVectorStore


def _wait_for_postgres(vector_store: PgVectorStore, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with vector_store.engine.begin() as connection:
                connection.execute(text("SELECT 1"))
            return
        except OperationalError as error:
            last_error = error
            logger.info("Waiting for PostgreSQL/pgvector to become ready...")
            time.sleep(2)
    raise TimeoutError(f"PostgreSQL/pgvector was not ready after {timeout_seconds}s: {last_error}")


def _count_rag_chunks(vector_store: PgVectorStore) -> int:
    vector_store.ensure_schema()
    with vector_store.engine.begin() as connection:
        return int(connection.execute(text("SELECT COUNT(*) FROM rag_chunks")).scalar() or 0)


def _count_supported_documents(master_data_dir: Path) -> int:
    return sum(
        1
        for path in master_data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_MASTER_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Idempotently seed RAG pgvector tables from MASTER_DATA.")
    parser.add_argument("--master-data-dir", type=Path, default=None, help="Directory restored by DVC.")
    parser.add_argument("--timeout-seconds", type=int, default=90, help="PostgreSQL readiness timeout.")
    args = parser.parse_args()

    rag_config = RAGConfig()
    master_data_dir = Path(args.master_data_dir or rag_config.master_data_dir)
    if not master_data_dir.exists():
        raise FileNotFoundError(
            f"MASTER_DATA directory was not found at {master_data_dir}. "
            "Run `dvc pull` before `docker compose up --build`."
        )
    supported_document_count = _count_supported_documents(master_data_dir)
    if supported_document_count == 0:
        raise FileNotFoundError(
            f"No supported documents were found in {master_data_dir}. "
            "Run `dvc pull` and confirm MASTER_DATA contains .pdf, .docx, or .txt files."
        )

    vector_store = PgVectorStore(rag_config=rag_config)
    _wait_for_postgres(vector_store, timeout_seconds=args.timeout_seconds)

    existing_chunks = _count_rag_chunks(vector_store)
    if existing_chunks > 0:
        logger.info("RAG database already contains %d chunks. Skipping seed ingestion.", existing_chunks)
        print({"status": "skipped", "reason": "rag_chunks_not_empty", "existing_chunks": existing_chunks})
        return

    logger.info(
        "RAG database is empty. Starting MASTER_DATA seed ingestion from %s with %d supported documents.",
        master_data_dir,
        supported_document_count,
    )
    result = MasterDataIngestionPipeline(rag_config=rag_config, vector_store=vector_store).run(
        master_data_dir=master_data_dir
    )
    print({"status": "seeded", **result})


if __name__ == "__main__":
    main()
