"""Manual ingestion pipeline for the trusted MASTER_DATA knowledge base."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.config import DataIngestionConfig, RAGConfig
from src.data_preprocessing import chunk_document_text, clean_document_text
from src.exception import CustomException
from src.file_utils import extract_text_from_docx, extract_text_from_pdf, extract_text_from_txt
from src.logger import logger
from src.rag.access_control import demo_access_level_for_index
from src.rag.retrieval_policy import MASTER_SOURCE_TYPE
from src.rag.service import RAGIndexer
from src.rag.vector_store import PgVectorStore
from src.utils.hashing import calculate_file_hash


SUPPORTED_MASTER_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return extract_text_from_docx(file_path)
    if suffix == ".txt":
        return extract_text_from_txt(file_path)
    raise ValueError(f"Unsupported master-data file type: {suffix}")


class MasterDataIngestionPipeline:
    """Indexes local MASTER_DATA files into PostgreSQL pgvector.

    This pipeline is intentionally manual. It does not train the classifier and
    it does not run automatically unless explicitly called by the developer.
    """

    def __init__(
        self,
        rag_config: RAGConfig | None = None,
        data_config: DataIngestionConfig | None = None,
        vector_store: PgVectorStore | None = None,
        indexer: RAGIndexer | None = None,
    ) -> None:
        self.rag_config = rag_config or RAGConfig()
        self.data_config = data_config or DataIngestionConfig()
        self.vector_store = vector_store
        self.indexer = indexer

    @classmethod
    def is_configured(cls) -> bool:
        return RAGIndexer.is_configured()

    def run(self, master_data_dir: Path | str | None = None) -> dict[str, Any]:
        try:
            if not self.is_configured():
                raise ValueError("Master RAG indexing requires PostgreSQL and GEMINI_API_KEY.")

            root = Path(master_data_dir or self.rag_config.master_data_dir)
            if not root.exists():
                raise FileNotFoundError(f"MASTER_DATA_DIR does not exist: {root}")

            files = sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_MASTER_EXTENSIONS
            )
            indexed_documents = 0
            skipped_documents = 0
            indexed_chunks = 0
            vector_store = self.vector_store or PgVectorStore(rag_config=self.rag_config)
            indexer = self.indexer or RAGIndexer(
                config=self.rag_config,
                vector_store=vector_store,
            )

            for file_index, file_path in enumerate(tqdm(files, desc="Indexing MASTER_DATA", unit="file")):
                file_hash = calculate_file_hash(file_path)
                existing = vector_store.get_document_by_file_hash(file_hash, source_type=MASTER_SOURCE_TYPE)
                if existing:
                    skipped_documents += 1
                    logger.info("Skipping already indexed MASTER_DATA file: %s", file_path)
                    continue

                text = clean_document_text(_extract_text(file_path))
                if not text.strip():
                    logger.warning("Skipping MASTER_DATA file with no extractable text: %s", file_path)
                    skipped_documents += 1
                    continue

                chunk_texts = chunk_document_text(
                    text,
                    chunk_size=self.data_config.chunk_size,
                    chunk_overlap=self.data_config.chunk_overlap,
                )
                if not chunk_texts:
                    logger.warning("Skipping MASTER_DATA file with no chunks: %s", file_path)
                    skipped_documents += 1
                    continue

                relative_name = file_path.relative_to(root).as_posix()
                document_id = f"master_{file_hash[:12]}"
                access_level = demo_access_level_for_index(file_index)
                indexed_count = indexer.index_document(
                    document_id=document_id,
                    file_name=relative_name,
                    predicted_class=None,
                    chunk_texts=chunk_texts,
                    uploaded_by="master_data_ingestion",
                    source_type=MASTER_SOURCE_TYPE,
                    verification_status="verified",
                    file_hash=file_hash,
                    access_level=access_level,
                    owner_id="master_data_ingestion",
                )
                indexed_documents += 1
                indexed_chunks += indexed_count

            result = {
                "indexed_documents": indexed_documents,
                "skipped_documents": skipped_documents,
                "indexed_chunks": indexed_chunks,
                "source_type": MASTER_SOURCE_TYPE,
                "master_data_dir": str(root),
                "message": "MASTER_DATA indexing complete.",
            }
            logger.info("MASTER_DATA indexing result: %s", result)
            return result
        except Exception as error:
            logger.exception("MASTER_DATA ingestion failed")
            raise CustomException(error, sys.exc_info()) from error
