"""Manual ingestion pipeline for the labeled local data/ folder."""

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
from src.rag.retrieval_policy import TRAINING_SOURCE_TYPE
from src.rag.service import RAGIndexer
from src.rag.vector_store import PgVectorStore
from src.utils.hashing import calculate_file_hash


SUPPORTED_TRAINING_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".docx":
        return extract_text_from_docx(file_path)
    if suffix == ".txt":
        return extract_text_from_txt(file_path)
    raise ValueError(f"Unsupported training-data file type: {suffix}")


class TrainingDataIngestionPipeline:
    """Indexes labeled local training documents into PostgreSQL pgvector.

    Expected structure:

    data/
      protocol/
      safety_report/
      statistical_analysis_plan/

    The class folder name is stored as `predicted_class` metadata. This is a
    manual RAG indexing utility only; it does not retrain the classifier.
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

    def run(self, data_dir: Path | str | None = None) -> dict[str, Any]:
        try:
            if not self.is_configured():
                raise ValueError("Training-data RAG indexing requires PostgreSQL and GEMINI_API_KEY.")

            root = Path(data_dir or self.data_config.raw_data_dir)
            if not root.exists():
                raise FileNotFoundError(f"data directory does not exist: {root}")

            files = sorted(
                path
                for path in root.rglob("*")
                if path.is_file() and not path.name.startswith(".") and path.suffix.lower() in SUPPORTED_TRAINING_EXTENSIONS
            )
            indexed_documents = 0
            skipped_documents = 0
            indexed_chunks = 0
            classes_seen: set[str] = set()
            vector_store = self.vector_store or PgVectorStore(rag_config=self.rag_config)
            indexer = self.indexer or RAGIndexer(
                config=self.rag_config,
                vector_store=vector_store,
            )

            for file_path in tqdm(files, desc="Indexing data/ into pgvector", unit="file"):
                relative_path = file_path.relative_to(root)
                if len(relative_path.parts) < 2:
                    logger.warning("Skipping file outside a class folder: %s", file_path)
                    skipped_documents += 1
                    continue

                class_name = relative_path.parts[0]
                classes_seen.add(class_name)
                file_hash = calculate_file_hash(file_path)
                existing = vector_store.get_document_by_file_hash(file_hash, source_type=TRAINING_SOURCE_TYPE)
                if existing:
                    skipped_documents += 1
                    logger.info("Skipping already indexed training data file: %s", file_path)
                    continue

                text = clean_document_text(_extract_text(file_path))
                if not text.strip():
                    logger.warning("Skipping training data file with no extractable text: %s", file_path)
                    skipped_documents += 1
                    continue

                chunk_texts = chunk_document_text(
                    text,
                    chunk_size=self.data_config.chunk_size,
                    chunk_overlap=self.data_config.chunk_overlap,
                )
                if not chunk_texts:
                    logger.warning("Skipping training data file with no chunks: %s", file_path)
                    skipped_documents += 1
                    continue

                document_id = f"training_{file_hash[:12]}"
                indexed_count = indexer.index_document(
                    document_id=document_id,
                    file_name=relative_path.as_posix(),
                    predicted_class=class_name,
                    chunk_texts=chunk_texts,
                    uploaded_by="training_data_ingestion",
                    source_type=TRAINING_SOURCE_TYPE,
                    verification_status="verified",
                    file_hash=file_hash,
                    access_level="User",
                    owner_id="training_data_ingestion",
                )
                indexed_documents += 1
                indexed_chunks += indexed_count

            result = {
                "indexed_documents": indexed_documents,
                "skipped_documents": skipped_documents,
                "indexed_chunks": indexed_chunks,
                "source_type": TRAINING_SOURCE_TYPE,
                "data_dir": str(root),
                "classes_seen": sorted(classes_seen),
                "message": "Training data indexing complete.",
            }
            logger.info("Training data indexing result: %s", result)
            return result
        except Exception as error:
            logger.exception("Training data ingestion failed")
            raise CustomException(error, sys.exc_info()) from error
