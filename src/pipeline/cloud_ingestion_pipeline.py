"""Cloud upload, duplicate detection, extraction, prediction, and persistence pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from src.cloud.s3_manager import S3Manager
from src.config import CloudConfig, DataIngestionConfig, DatabaseConfig, MLOpsConfig
from src.data_preprocessing import chunk_document_text, clean_document_text
from src.database.repository import TMFRepository
from src.exception import CustomException
from src.file_utils import extract_text_from_docx, extract_text_from_pdf, extract_text_from_txt
from src.logger import logger
from src.predict import predict_text
from src.utils import document_confidence_summary
from src.utils.hashing import calculate_file_hash, calculate_text_hash


SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".txt"}


class CloudIngestionPipeline:
    """Idempotent cloud ingestion pipeline for uploaded TMF documents.

    If S3/PostgreSQL are not configured, the pipeline still performs local
    extraction and inference for backward-compatible API use.
    """

    def __init__(
        self,
        cloud_config: CloudConfig | None = None,
        data_config: DataIngestionConfig | None = None,
        mlops_config: MLOpsConfig | None = None,
        repository: TMFRepository | None = None,
        s3_manager: S3Manager | None = None,
    ) -> None:
        self.cloud_config = cloud_config or CloudConfig()
        self.data_config = data_config or DataIngestionConfig()
        self.mlops_config = mlops_config or MLOpsConfig()
        self.repository = repository or self._build_repository_if_configured()
        self.s3_manager = s3_manager or (S3Manager(self.cloud_config) if self.cloud_config.is_configured else None)

    def _build_repository_if_configured(self) -> TMFRepository | None:
        if not DatabaseConfig().is_configured:
            return None
        try:
            return TMFRepository()
        except Exception as error:
            logger.warning("PostgreSQL persistence is unavailable. Running upload prediction without DB persistence: %s", error)
            return None

    @property
    def persistence_enabled(self) -> bool:
        return self.repository is not None and self.s3_manager is not None

    async def run(self, upload_file, uploaded_by: str | None = None) -> dict[str, Any]:
        """Run the idempotent ingestion pipeline for a FastAPI UploadFile-like object."""
        safe_filename = Path(upload_file.filename or "uploaded_document").name
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            raise ValueError("Unsupported file type. Supported formats are .pdf, .docx, and .txt.")

        contents = await upload_file.read()
        if not contents:
            raise ValueError("Uploaded file is empty.")

        try:
            with tempfile.TemporaryDirectory(prefix="tmf_cloud_ingest_") as temp_dir:
                temp_path = Path(temp_dir) / safe_filename
                temp_path.write_bytes(contents)
                file_hash = calculate_file_hash(temp_path)

                duplicate_result = self._handle_duplicate(file_hash)
                if duplicate_result:
                    return duplicate_result

                raw_s3_uri = self._upload_raw_document(temp_path, safe_filename, file_hash) if self.s3_manager else None
                extracted_text = self._extract_text(temp_path)
                cleaned_text = clean_document_text(extracted_text)
                if not cleaned_text.strip():
                    raise ValueError("Uploaded file does not contain extractable text.")

                extracted_text_s3_uri = (
                    self._upload_extracted_text(temp_dir, safe_filename, file_hash, cleaned_text)
                    if self.s3_manager
                    else None
                )
                chunk_texts = chunk_document_text(
                    cleaned_text,
                    chunk_size=self.data_config.chunk_size,
                    chunk_overlap=self.data_config.chunk_overlap,
                )
                if not chunk_texts:
                    raise ValueError("Uploaded file did not produce any prediction chunks.")

                chunk_results = [predict_text(chunk_text) for chunk_text in chunk_texts]
                prediction = document_confidence_summary(
                    chunk_results,
                    class_order=None,
                )

                response = {
                    "filename": safe_filename,
                    **{key: prediction[key] for key in (
                        "predicted_label",
                        "confidence",
                        "model_confidence",
                        "vote_confidence",
                        "margin_confidence",
                        "requires_review",
                        "decision_status",
                        "num_chunks",
                        "chunk_predictions",
                    )},
                    "duplicate": False,
                    "document_status": "predicted_unverified",
                    "persistence_enabled": self.persistence_enabled,
                }

                if self.repository:
                    document = self._persist_document(
                        safe_filename=safe_filename,
                        uploaded_by=uploaded_by,
                        file_hash=file_hash,
                        raw_s3_uri=raw_s3_uri,
                        extracted_text_s3_uri=extracted_text_s3_uri,
                    )
                    chunk_rows = self._persist_chunks(document["doc_id"], chunk_texts)
                    self._persist_prediction(document["doc_id"], prediction)
                    self._persist_chunk_predictions(document["doc_id"], chunk_rows, chunk_results)
                    self._index_rag_document(
                        document["doc_id"],
                        safe_filename,
                        response["predicted_label"],
                        chunk_texts,
                        uploaded_by,
                        file_hash,
                    )
                    self.repository.save_audit_log(
                        event_type="document_ingested",
                        entity_type="document",
                        entity_id=str(document["doc_id"]),
                        message="Document uploaded, predicted, and marked unverified.",
                        details={"filename": safe_filename, "file_hash": file_hash},
                    )
                    response["doc_id"] = document["doc_id"]

                logger.info(
                    "Cloud ingestion complete: filename=%s, chars=%d, chunks=%d, prediction=%s, confidence=%.4f",
                    safe_filename,
                    len(cleaned_text),
                    len(chunk_texts),
                    response["predicted_label"],
                    response["confidence"],
                )
                return response
        except Exception as error:
            logger.exception("Cloud ingestion pipeline failed for %s", safe_filename)
            raise CustomException(error) from error

    def _handle_duplicate(self, file_hash: str) -> dict[str, Any] | None:
        if not self.repository or self.cloud_config.allow_duplicate_documents:
            return None
        existing_document = self.repository.get_document_by_hash(file_hash)
        if not existing_document:
            return None
        existing_prediction = self.repository.get_existing_prediction_by_doc_id(existing_document["doc_id"])
        if not existing_prediction:
            logger.info("Duplicate document metadata found, but no prediction exists yet. Reprocessing hash %s.", file_hash)
            return None
        logger.info("Duplicate document detected for hash %s. Skipping upload/extraction/prediction.", file_hash)
        self.repository.save_audit_log(
            event_type="duplicate_document_skipped",
            entity_type="document",
            entity_id=str(existing_document["doc_id"]),
            message="Duplicate document upload skipped.",
            details={"file_hash": file_hash},
        )
        return {
            "filename": existing_document["filename"],
            "doc_id": existing_document["doc_id"],
            "duplicate": True,
            "document_status": existing_document["document_status"],
            "persistence_enabled": self.persistence_enabled,
            **(existing_prediction or {}),
        }

    def _extract_text(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return extract_text_from_pdf(file_path)
        if suffix == ".docx":
            return extract_text_from_docx(file_path)
        if suffix == ".txt":
            return extract_text_from_txt(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _upload_raw_document(self, local_path: Path, filename: str, file_hash: str) -> str:
        key = f"{self.cloud_config.raw_documents_prefix}{file_hash}_{filename}"
        return self.s3_manager.upload_file(local_path, key)

    def _upload_extracted_text(self, temp_dir: str, filename: str, file_hash: str, text: str) -> str:
        text_path = Path(temp_dir) / f"{Path(filename).stem}_{file_hash}.txt"
        text_path.write_text(text, encoding="utf-8")
        key = f"{self.cloud_config.processed_documents_prefix}{file_hash}.txt"
        return self.s3_manager.upload_file(text_path, key, content_type="text/plain")

    def _persist_document(
        self,
        safe_filename: str,
        uploaded_by: str | None,
        file_hash: str,
        raw_s3_uri: str | None,
        extracted_text_s3_uri: str | None,
    ) -> dict:
        return self.repository.save_document(
            {
                "filename": safe_filename,
                "uploaded_by": uploaded_by,
                "raw_document_s3_uri": raw_s3_uri,
                "extracted_text_s3_uri": extracted_text_s3_uri,
                "file_hash": file_hash,
                "document_status": "predicted_unverified",
                "used_for_training": False,
                "dataset_version": self.mlops_config.dataset_version,
            }
        )

    def _persist_chunks(self, doc_id: int, chunk_texts: list[str]) -> list[dict]:
        return self.repository.save_chunks(
            doc_id,
            [
                {
                    "chunk_index": index,
                    "chunk_hash": calculate_text_hash(chunk_text),
                    "chunk_text_s3_uri": None,
                    "chunk_word_count": len(chunk_text.split()),
                }
                for index, chunk_text in enumerate(chunk_texts)
            ],
        )

    def _persist_prediction(self, doc_id: int, prediction: dict[str, Any]) -> dict:
        return self.repository.save_prediction(
            doc_id,
            {
                "predicted_label": prediction["predicted_label"],
                "confidence": prediction["confidence"],
                "model_confidence": prediction["model_confidence"],
                "vote_confidence": prediction["vote_confidence"],
                "margin_confidence": prediction["margin_confidence"],
                "requires_review": prediction["requires_review"],
                "decision_status": prediction["decision_status"],
                "num_chunks": prediction["num_chunks"],
                "chunk_predictions": prediction["chunk_predictions"],
                "model_version": self.mlops_config.model_version,
                "dataset_version": self.mlops_config.dataset_version,
            },
        )

    def _persist_chunk_predictions(self, doc_id: int, chunk_rows: list[dict], chunk_results: list[dict]) -> list[dict]:
        return self.repository.save_chunk_predictions(
            doc_id,
            [
                {
                    "chunk_id": chunk_row["chunk_id"],
                    "predicted_label": str(chunk_result["predicted_label"]),
                    "confidence": float(chunk_result["confidence"]),
                    "model_version": self.mlops_config.model_version,
                }
                for chunk_row, chunk_result in zip(chunk_rows, chunk_results)
            ],
        )

    def _index_rag_document(
        self,
        doc_id: int,
        filename: str,
        predicted_class: str,
        chunk_texts: list[str],
        uploaded_by: str | None,
        file_hash: str | None,
    ) -> None:
        try:
            from src.rag.service import RAGIndexer

            if not RAGIndexer.is_configured():
                logger.info("RAG indexing skipped for doc_id=%s because Gemini/PostgreSQL is not configured.", doc_id)
                return
            RAGIndexer().index_document(
                document_id=str(doc_id),
                file_name=filename,
                predicted_class=predicted_class,
                chunk_texts=chunk_texts,
                uploaded_by=uploaded_by,
                source_type="PREDICT_UPLOAD",
                verification_status="unverified",
                file_hash=file_hash,
                access_level="User",
                owner_id=uploaded_by,
            )
        except Exception as error:
            logger.warning("RAG indexing failed for doc_id=%s without blocking classification upload: %s", doc_id, error)
