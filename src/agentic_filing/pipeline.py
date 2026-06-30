"""Stage 6 agentic TMF filing pipeline.

The pipeline wraps existing extraction, classifier prediction, S3 persistence,
PostgreSQL metadata, Redis manual review, and RAG indexing. It does not retrain
the model and it does not replace the existing RAG implementation.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

from src.agentic_filing.review_queue import ManualReviewQueue
from src.cloud.s3_manager import S3Manager
from src.config import AgenticFilingConfig, CloudConfig, DataIngestionConfig, DatabaseConfig, MLOpsConfig
from src.data_preprocessing import chunk_document_text, clean_document_text
from src.database.repository import TMFRepository
from src.exception import CustomException
from src.file_utils import extract_text_from_docx, extract_text_from_pdf, extract_text_from_txt
from src.logger import logger
from src.predict import predict_text
from src.utils import document_confidence_summary
from src.utils.hashing import calculate_file_hash, calculate_text_hash


SUPPORTED_AGENTIC_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _s3_key_from_uri(s3_uri: str | None) -> str | None:
    if not s3_uri or not s3_uri.startswith("s3://"):
        return None
    parts = s3_uri.replace("s3://", "", 1).split("/", 1)
    return parts[1] if len(parts) == 2 else None


def _local_storage_key(storage_path: str | None) -> str | None:
    """Extract an agentic workspace key from S3 or local paths."""
    if not storage_path:
        return None
    s3_key = _s3_key_from_uri(storage_path)
    if s3_key:
        return s3_key
    new_marker = "agentic_tmf_workspace/"
    if new_marker in storage_path:
        return storage_path[storage_path.index(new_marker) :]
    marker = "cloud/"
    if marker in storage_path:
        return storage_path[storage_path.index(marker) :]
    return None


def _safe_folder(value: str | None, fallback: str = "unclassified") -> str:
    clean = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "").strip().lower())
    return clean or fallback


class AgenticTMFFilingPipeline:
    """Confidence-based auto-filing and manual-review workflow."""

    def __init__(
        self,
        agentic_config: AgenticFilingConfig | None = None,
        cloud_config: CloudConfig | None = None,
        data_config: DataIngestionConfig | None = None,
        mlops_config: MLOpsConfig | None = None,
        repository: TMFRepository | None = None,
        s3_manager: S3Manager | None = None,
        review_queue: ManualReviewQueue | None = None,
        predictor: Callable[[str], dict[str, Any]] | None = None,
        rag_indexer: Any | None = None,
    ) -> None:
        self.agentic_config = agentic_config or AgenticFilingConfig()
        self.cloud_config = cloud_config or CloudConfig()
        self.data_config = data_config or DataIngestionConfig()
        self.mlops_config = mlops_config or MLOpsConfig()
        self.repository = repository or self._build_repository_if_configured()
        self.s3_manager = s3_manager or (S3Manager(self.cloud_config) if self.cloud_config.is_configured else None)
        self.review_queue = review_queue or ManualReviewQueue(config=self.agentic_config)
        self.predictor = predictor or predict_text
        self.rag_indexer = rag_indexer

    def _build_repository_if_configured(self) -> TMFRepository | None:
        if not DatabaseConfig().is_configured:
            return None
        try:
            return TMFRepository()
        except Exception as error:
            logger.warning("PostgreSQL unavailable for Stage 6 filing; persistence disabled: %s", error)
            return None

    @property
    def persistence_enabled(self) -> bool:
        return self.repository is not None and self.s3_manager is not None

    async def run(self, upload_file, uploaded_by: str | None = None) -> dict[str, Any]:
        """Upload, classify, then auto-file or queue for manual review."""
        safe_filename = Path(upload_file.filename or "uploaded_document").name
        suffix = Path(safe_filename).suffix.lower()
        if suffix not in SUPPORTED_AGENTIC_EXTENSIONS:
            raise ValueError("Unsupported file type. Supported formats are .pdf, .docx, and .txt.")

        contents = await upload_file.read()
        if not contents:
            raise ValueError("Uploaded file is empty.")

        try:
            with tempfile.TemporaryDirectory(prefix="tmf_agentic_filing_") as temp_dir:
                logger.info("Stage 6 filing started: filename=%s, bytes=%d", safe_filename, len(contents))
                temp_path = Path(temp_dir) / safe_filename
                temp_path.write_bytes(contents)
                file_hash = calculate_file_hash(temp_path)
                logger.info("Stage 6 progress [1/7]: calculated file hash for %s", safe_filename)

                duplicate = self._handle_duplicate(file_hash)
                if duplicate:
                    return duplicate

                extracted_text = self._extract_text(temp_path)
                cleaned_text = clean_document_text(extracted_text)
                if not cleaned_text.strip():
                    raise ValueError("Uploaded file does not contain extractable text.")
                logger.info(
                    "Stage 6 progress [2/7]: extracted and cleaned text for %s (%d chars)",
                    safe_filename,
                    len(cleaned_text),
                )

                chunk_texts = chunk_document_text(
                    cleaned_text,
                    chunk_size=self.data_config.chunk_size,
                    chunk_overlap=self.data_config.chunk_overlap,
                )
                if not chunk_texts:
                    raise ValueError("Uploaded file did not produce any prediction chunks.")
                logger.info("Stage 6 progress [3/7]: created %d chunks for %s", len(chunk_texts), safe_filename)

                logger.info("Stage 6 progress [4/7]: running classifier on chunks for %s", safe_filename)
                chunk_results = [self.predictor(chunk_text) for chunk_text in chunk_texts]
                prediction = document_confidence_summary(chunk_results, class_order=None)
                top_k_predictions = self._aggregate_top_k_predictions(chunk_results, prediction)
                confidence_gap = self._confidence_gap(top_k_predictions)
                decision_confidence = float(prediction["confidence"])
                predicted_class = str(prediction["predicted_label"])
                should_auto_file = self._should_auto_file(decision_confidence, confidence_gap)
                logger.info(
                    "Stage 6 progress [5/7]: decision=%s predicted=%s confidence=%.4f gap=%.4f threshold=%.4f",
                    "auto_file" if should_auto_file else "manual_review",
                    predicted_class,
                    decision_confidence,
                    confidence_gap,
                    self.agentic_config.auto_approval_threshold,
                )

                if should_auto_file:
                    return self._auto_file(
                        temp_dir=Path(temp_dir),
                        temp_path=temp_path,
                        safe_filename=safe_filename,
                        uploaded_by=uploaded_by,
                        file_hash=file_hash,
                        cleaned_text=cleaned_text,
                        chunk_texts=chunk_texts,
                        chunk_results=chunk_results,
                        prediction=prediction,
                        top_k_predictions=top_k_predictions,
                        confidence_gap=confidence_gap,
                        decision_confidence=decision_confidence,
                        final_class=predicted_class,
                    )
                return self._queue_manual_review(
                    temp_dir=Path(temp_dir),
                    temp_path=temp_path,
                    safe_filename=safe_filename,
                    uploaded_by=uploaded_by,
                    file_hash=file_hash,
                    cleaned_text=cleaned_text,
                    prediction=prediction,
                    top_k_predictions=top_k_predictions,
                    confidence_gap=confidence_gap,
                    decision_confidence=decision_confidence,
                    predicted_class=predicted_class,
                )
        except Exception as error:
            logger.exception("Stage 6 agentic filing failed for %s", safe_filename)
            raise CustomException(error, sys.exc_info()) from error

    def _handle_duplicate(self, file_hash: str) -> dict[str, Any] | None:
        if not self.repository or self.cloud_config.allow_duplicate_documents:
            return None
        existing_document = self.repository.get_document_by_hash(file_hash)
        if not existing_document:
            return None
        existing_prediction = self.repository.get_existing_prediction_by_doc_id(existing_document["doc_id"])
        if not existing_prediction:
            logger.info("Duplicate document metadata found without prediction. Reprocessing hash %s.", file_hash)
            return None
        self.repository.save_audit_log(
            event_type="duplicate_detected",
            entity_type="document",
            entity_id=str(existing_document["doc_id"]),
            message="Duplicate document upload detected and skipped.",
            details={"file_hash": file_hash},
        )
        return {
            "filename": existing_document["filename"],
            "doc_id": existing_document["doc_id"],
            "duplicate": True,
            "document_status": "duplicate_detected",
            "persistence_enabled": self.persistence_enabled,
            "agentic_action": "duplicate_detected",
            "final_class": existing_document.get("verified_label"),
            **self._sanitize_prediction_response(existing_prediction),
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

    def _aggregate_top_k_predictions(self, chunk_results: list[dict[str, Any]], prediction: dict[str, Any]) -> list[dict[str, Any]]:
        scores: dict[str, float] = defaultdict(float)
        for result in chunk_results:
            top_predictions = result.get("top_predictions")
            if isinstance(top_predictions, list):
                for item in top_predictions:
                    if isinstance(item, dict) and item.get("label") is not None:
                        scores[str(item["label"])] += float(item.get("confidence", 0.0))
                continue
            if isinstance(top_predictions, dict):
                for label, confidence in top_predictions.items():
                    scores[str(label)] += float(confidence)
                continue
            scores[str(result["predicted_label"])] += float(result["confidence"])

        total_chunks = max(len(chunk_results), 1)
        for label in prediction.get("chunk_predictions", {}):
            scores.setdefault(str(label), 0.0)
        ranked = sorted(
            ({"label": label, "confidence": confidence / total_chunks} for label, confidence in scores.items()),
            key=lambda item: item["confidence"],
            reverse=True,
        )
        return ranked[:5]

    def _confidence_gap(self, top_k_predictions: list[dict[str, Any]]) -> float:
        if len(top_k_predictions) < 2:
            return 1.0
        return float(top_k_predictions[0]["confidence"] - top_k_predictions[1]["confidence"])

    def _should_auto_file(self, confidence: float, confidence_gap: float) -> bool:
        return (
            confidence >= self.agentic_config.auto_approval_threshold
            and confidence_gap >= self.agentic_config.min_confidence_gap
        )

    def _local_path_for_key(self, key: str) -> Path:
        """Return local mirror path for an agentic workspace object key."""
        normalized_key = key.lstrip("/")
        if normalized_key.startswith("agentic_tmf_workspace/"):
            normalized_key = normalized_key[len("agentic_tmf_workspace/") :]
        if normalized_key.startswith("cloud/"):
            normalized_key = normalized_key[len("cloud/") :]
        return self.cloud_config.local_cloud_root / normalized_key

    def _mirror_file_local(self, local_path: Path, key: str) -> str:
        destination = self._local_path_for_key(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, destination)
        return str(destination)

    def _mirror_text_local(self, text: str, key: str) -> str:
        destination = self._local_path_for_key(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        return str(destination)

    def _mirror_copy_local(self, source_key: str | None, destination_key: str) -> str | None:
        if not source_key:
            return None
        source = self._local_path_for_key(source_key)
        if not source.exists():
            return None
        destination = self._local_path_for_key(destination_key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return str(destination)

    def _remove_local_key(self, key: str | None) -> None:
        if not key:
            return
        path = self._local_path_for_key(key)
        if path.exists():
            path.unlink()
            logger.info("Removed local pending/staging file: %s", path)

    def _delete_storage_path(self, storage_path: str | None) -> None:
        """Delete a local/S3 staging object after it has been safely copied elsewhere."""
        key = _local_storage_key(storage_path)
        if not key:
            return
        self._remove_local_key(key)
        if self.s3_manager and storage_path and storage_path.startswith("s3://") and hasattr(self.s3_manager, "delete_object"):
            self.s3_manager.delete_object(key)

    def _read_text_from_storage(self, storage_path: str | None) -> str:
        key = _local_storage_key(storage_path)
        if self.s3_manager and storage_path and storage_path.startswith("s3://") and key:
            return self.s3_manager.read_text(key)
        if key:
            local_path = self._local_path_for_key(key)
            if local_path.exists():
                return local_path.read_text(encoding="utf-8")
        if storage_path and Path(storage_path).exists():
            return Path(storage_path).read_text(encoding="utf-8")
        raise ValueError("Stored text object is unavailable locally and in cloud storage.")

    def _upload_raw(self, local_path: Path, filename: str, file_hash: str, prefix: str, final_class: str | None = None) -> str | None:
        class_part = f"{final_class.strip().lower()}/" if final_class else ""
        key = f"{prefix.rstrip('/')}/{class_part}{file_hash}_{filename}"
        local_mirror_path = self._mirror_file_local(local_path, key)
        if not self.s3_manager:
            return local_mirror_path
        return self.s3_manager.upload_file(local_path, key)

    def _upload_text(self, text: str, file_hash: str, prefix: str, filename: str | None = None, final_class: str | None = None) -> str | None:
        class_part = f"{final_class.strip().lower()}/" if final_class else ""
        stem = Path(filename or file_hash).stem
        key = f"{prefix.rstrip('/')}/{class_part}{file_hash}_{stem}.txt"
        local_mirror_path = self._mirror_text_local(text, key)
        if not self.s3_manager:
            return local_mirror_path
        return self.s3_manager.upload_text(text, key)

    def _save_document(self, filename: str, uploaded_by: str | None, file_hash: str, raw_uri: str | None, text_uri: str | None, status: str) -> dict | None:
        if not self.repository:
            return None
        return self.repository.save_document(
            {
                "filename": filename,
                "uploaded_by": uploaded_by,
                "raw_document_s3_uri": raw_uri,
                "extracted_text_s3_uri": text_uri,
                "file_hash": file_hash,
                "document_status": status,
                "used_for_training": False,
                "dataset_version": self.mlops_config.dataset_version,
            }
        )

    def _save_prediction(self, doc_id: int, prediction: dict[str, Any], top_k_predictions: list[dict[str, Any]]) -> None:
        if not self.repository:
            return
        self.repository.save_prediction(
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

    def _sanitize_prediction_response(self, prediction: dict[str, Any]) -> dict[str, Any]:
        """Keep API response chunk_predictions as class-count integers only."""
        sanitized = dict(prediction)
        chunk_predictions = sanitized.get("chunk_predictions") or {}
        if isinstance(chunk_predictions, dict):
            top_k_predictions = chunk_predictions.get("_top_k_predictions")
            sanitized["chunk_predictions"] = {
                str(label): int(count)
                for label, count in chunk_predictions.items()
                if label != "_top_k_predictions" and isinstance(count, (int, float))
            }
            if top_k_predictions and "top_k_predictions" not in sanitized:
                sanitized["top_k_predictions"] = top_k_predictions
        return sanitized

    def _save_chunks_and_predictions(self, doc_id: int, chunk_texts: list[str], chunk_results: list[dict[str, Any]]) -> None:
        if not self.repository:
            return
        chunk_rows = self.repository.save_chunks(
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
        self.repository.save_chunk_predictions(
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

    def _metadata_payload(
        self,
        doc_id: int | None,
        filename: str,
        file_hash: str,
        predicted_class: str,
        confidence_score: float,
        final_class: str | None,
        cloud_storage_path: str | None,
        metadata_path: str | None,
        status: str,
        approval_status: str,
        rag_ingested: bool,
        top_k_predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "document_id": str(doc_id or file_hash),
            "file_name": filename,
            "file_hash": file_hash,
            "document_type": final_class,
            "predicted_class": predicted_class,
            "confidence_score": confidence_score,
            "final_class": final_class,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "source": "api_upload",
            "cloud_storage_path": cloud_storage_path,
            "metadata_path": metadata_path,
            "version": self.mlops_config.model_version,
            "status": status,
            "approval_status": approval_status,
            "rag_ingested": rag_ingested,
            "top_k_predictions": top_k_predictions,
        }

    def _persist_metadata(self, doc_id: int | None, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.repository or doc_id is None:
            return None
        return self.repository.save_document_metadata(
            {
                "doc_id": doc_id,
                "document_id": str(doc_id),
                "file_name": payload["file_name"],
                "file_hash": payload["file_hash"],
                "document_type": payload["document_type"],
                "predicted_class": payload["predicted_class"],
                "confidence_score": payload["confidence_score"],
                "final_class": payload["final_class"],
                "source": payload["source"],
                "cloud_storage_path": payload["cloud_storage_path"],
                "metadata_path": payload["metadata_path"],
                "version": payload["version"],
                "status": payload["status"],
                "approval_status": payload["approval_status"],
                "rag_ingested": payload["rag_ingested"],
                "details": payload,
            }
        )

    def _upload_metadata(self, payload: dict[str, Any], file_hash: str, doc_id: int | None) -> str | None:
        status_folder = _safe_folder(payload.get("status") or payload.get("approval_status"), "unknown_status")
        class_folder = _safe_folder(payload.get("final_class") or payload.get("predicted_class"), "unclassified")
        key = f"{self.cloud_config.metadata_prefix.rstrip('/')}/{status_folder}/{class_folder}/{doc_id or file_hash}.json"
        metadata_path = (
            self.s3_manager.generate_s3_uri(key)
            if self.s3_manager and hasattr(self.s3_manager, "generate_s3_uri")
            else str(self._local_path_for_key(key))
        )
        payload["metadata_path"] = metadata_path
        payload_json = json.dumps(payload, indent=2, default=str)
        local_metadata_path = self._mirror_text_local(payload_json, key)
        if not self.s3_manager:
            return local_metadata_path
        logger.info("Uploading structured metadata for doc_id=%s to %s", doc_id, key)
        return self.s3_manager.upload_text(payload_json, key, content_type="application/json")

    def _index_rag(self, doc_id: int, filename: str, final_class: str, chunk_texts: list[str], uploaded_by: str | None, file_hash: str) -> tuple[bool, str]:
        try:
            if self.rag_indexer is not None:
                self.rag_indexer.index_document(
                    document_id=str(doc_id),
                    file_name=filename,
                    predicted_class=final_class,
                    chunk_texts=chunk_texts,
                    uploaded_by=uploaded_by,
                    source_type="PREDICT_UPLOAD",
                    verification_status="verified",
                    file_hash=file_hash,
                )
                return True, "rag_ingested"
            from src.rag.service import RAGIndexer

            if not RAGIndexer.is_configured():
                logger.info("RAG indexing skipped for doc_id=%s because RAG is not configured.", doc_id)
                return False, "skipped"
            RAGIndexer().index_document(
                document_id=str(doc_id),
                file_name=filename,
                predicted_class=final_class,
                chunk_texts=chunk_texts,
                uploaded_by=uploaded_by,
                source_type="PREDICT_UPLOAD",
                verification_status="verified",
                file_hash=file_hash,
            )
            return True, "rag_ingested"
        except Exception as error:
            logger.warning("RAG ingestion failed for doc_id=%s: %s", doc_id, error)
            return False, "failed"

    def _register_pending_training(self, temp_path: Path | None, filename: str, file_hash: str, final_class: str, source_uri: str | None = None) -> str | None:
        if not self.s3_manager:
            if temp_path:
                return self._upload_raw(temp_path, filename, file_hash, self.cloud_config.pending_training_prefix, final_class)
            return self._mirror_copy_local(
                _s3_key_from_uri(source_uri) or source_uri,
                f"{self.cloud_config.pending_training_prefix.rstrip('/')}/{final_class}/{file_hash}_{filename}",
            )
        destination_key = f"{self.cloud_config.pending_training_prefix.rstrip('/')}/{final_class}/{file_hash}_{filename}"
        source_key = _s3_key_from_uri(source_uri)
        self._mirror_copy_local(source_key, destination_key)
        if source_key:
            return self.s3_manager.copy_object(source_key, destination_key)
        if temp_path:
            local_path = self._mirror_file_local(temp_path, destination_key)
            return self.s3_manager.upload_file(temp_path, destination_key) or local_path
        return None

    def _audit(self, doc_id: int | None, action_type: str, message: str, details: dict[str, Any]) -> None:
        if not self.repository:
            return
        self.repository.save_audit_log(
            event_type=action_type,
            entity_type="document",
            entity_id=str(doc_id) if doc_id is not None else None,
            message=message,
            details=details,
        )

    def _base_response(
        self,
        filename: str,
        prediction: dict[str, Any],
        top_k_predictions: list[dict[str, Any]],
        duplicate: bool,
        status: str,
        doc_id: int | None,
        agentic_action: str,
        final_class: str | None,
        confidence_gap: float,
        decision_confidence: float | None = None,
        metadata_path: str | None = None,
        rag_ingestion_status: str | None = None,
    ) -> dict[str, Any]:
        return {
            "filename": filename,
            "predicted_label": prediction["predicted_label"],
            "confidence": prediction["confidence"],
            "model_confidence": prediction["model_confidence"],
            "vote_confidence": prediction["vote_confidence"],
            "margin_confidence": prediction["margin_confidence"],
            "requires_review": agentic_action == "manual_review_required",
            "decision_status": prediction["decision_status"],
            "num_chunks": prediction["num_chunks"],
            "chunk_predictions": prediction["chunk_predictions"],
            "duplicate": duplicate,
            "document_status": status,
            "persistence_enabled": self.persistence_enabled,
            "doc_id": doc_id,
            "agentic_action": agentic_action,
            "final_class": final_class,
            "confidence_gap": confidence_gap,
            "decision_confidence": decision_confidence,
            "top_k_predictions": top_k_predictions,
            "metadata_path": metadata_path,
            "rag_ingestion_status": rag_ingestion_status,
        }

    def _auto_file(self, **kwargs) -> dict[str, Any]:
        final_class = kwargs["final_class"]
        filename = kwargs["safe_filename"]
        file_hash = kwargs["file_hash"]
        prediction = kwargs["prediction"]
        logger.info("Stage 6 progress [6/7]: auto-filing %s into class %s", filename, final_class)
        raw_uri = self._upload_raw(
            kwargs["temp_path"],
            filename,
            file_hash,
            self.cloud_config.tmf_prefix,
            final_class=final_class,
        )
        text_uri = self._upload_text(
            kwargs["cleaned_text"],
            file_hash,
            self.cloud_config.processed_documents_prefix,
            filename=filename,
            final_class=final_class,
        )
        document = self._save_document(
            filename,
            kwargs["uploaded_by"],
            file_hash,
            raw_uri,
            text_uri,
            status="auto_filed",
        )
        doc_id = document["doc_id"] if document else None
        if doc_id:
            logger.info("Stage 6 progress: saving predictions/chunks for doc_id=%s", doc_id)
            self._save_prediction(doc_id, prediction, kwargs["top_k_predictions"])
            self._save_chunks_and_predictions(doc_id, kwargs["chunk_texts"], kwargs["chunk_results"])

        rag_ingested = False
        rag_status = "skipped"
        if doc_id:
            logger.info("Stage 6 progress: starting RAG ingestion for doc_id=%s", doc_id)
            rag_ingested, rag_status = self._index_rag(
                doc_id,
                filename,
                final_class,
                kwargs["chunk_texts"],
                kwargs["uploaded_by"],
                file_hash,
            )
            self.repository.update_document_status(
                doc_id,
                "pending_training_approval" if rag_ingested else "auto_filed",
            )
        training_uri = self._register_pending_training(kwargs["temp_path"], filename, file_hash, final_class, raw_uri)
        logger.info("Stage 6 progress: registered pending training copy for doc_id=%s", doc_id)
        metadata_payload = self._metadata_payload(
            doc_id,
            filename,
            file_hash,
            predicted_class=str(prediction["predicted_label"]),
            confidence_score=kwargs["decision_confidence"],
            final_class=final_class,
            cloud_storage_path=raw_uri,
            metadata_path=None,
            status="pending_training_approval" if rag_ingested else "auto_filed",
            approval_status="pending_training_approval",
            rag_ingested=rag_ingested,
            top_k_predictions=kwargs["top_k_predictions"],
        )
        metadata_payload["pending_training_path"] = training_uri
        metadata_path = self._upload_metadata(metadata_payload, file_hash, doc_id)
        metadata_payload["metadata_path"] = metadata_path
        self._persist_metadata(doc_id, metadata_payload)
        logger.info("Stage 6 progress [7/7]: metadata/audit saved for auto-filed doc_id=%s", doc_id)
        self._audit(
            doc_id,
            "auto_filed",
            "High-confidence document auto-filed and registered for training approval.",
            {
                **metadata_payload,
                "rag_ingestion_status": rag_status,
                "decision_confidence": kwargs["decision_confidence"],
            },
        )
        logger.info(
            "Document auto-filed: doc_id=%s, final_class=%s, decision_confidence=%.4f",
            doc_id,
            final_class,
            kwargs["decision_confidence"],
        )
        return self._base_response(
            filename,
            prediction,
            kwargs["top_k_predictions"],
            duplicate=False,
            status="pending_training_approval" if rag_ingested else "auto_filed",
            doc_id=doc_id,
            agentic_action="auto_filed",
            final_class=final_class,
            confidence_gap=kwargs["confidence_gap"],
            decision_confidence=kwargs["decision_confidence"],
            metadata_path=metadata_path,
            rag_ingestion_status=rag_status,
        )

    def _queue_manual_review(self, **kwargs) -> dict[str, Any]:
        filename = kwargs["safe_filename"]
        file_hash = kwargs["file_hash"]
        prediction = kwargs["prediction"]
        logger.info("Stage 6 progress [6/7]: storing %s in pending_review", filename)
        raw_uri = self._upload_raw(kwargs["temp_path"], filename, file_hash, self.cloud_config.pending_review_prefix)
        text_uri = self._upload_text(kwargs["cleaned_text"], file_hash, self.cloud_config.pending_review_prefix, filename=filename)
        document = self._save_document(
            filename,
            kwargs["uploaded_by"],
            file_hash,
            raw_uri,
            text_uri,
            status="pending_review",
        )
        doc_id = document["doc_id"] if document else None
        if doc_id:
            self._save_prediction(doc_id, prediction, kwargs["top_k_predictions"])
        metadata_payload = self._metadata_payload(
            doc_id,
            filename,
            file_hash,
            predicted_class=kwargs["predicted_class"],
            confidence_score=kwargs["decision_confidence"],
            final_class=None,
            cloud_storage_path=raw_uri,
            metadata_path=None,
            status="pending_review",
            approval_status="pending_review",
            rag_ingested=False,
            top_k_predictions=kwargs["top_k_predictions"],
        )
        metadata_path = self._upload_metadata(metadata_payload, file_hash, doc_id)
        metadata_payload["metadata_path"] = metadata_path
        self._persist_metadata(doc_id, metadata_payload)
        logger.info("Stage 6 progress: metadata saved for pending-review doc_id=%s", doc_id)
        review_item = self.review_queue.push(
            {
                "document_id": doc_id or file_hash,
                "original_file_name": filename,
                "predicted_class": kwargs["predicted_class"],
                "confidence_score": kwargs["decision_confidence"],
                "top_k_predictions": kwargs["top_k_predictions"],
                "temporary_cloud_path": raw_uri,
                "extracted_text_s3_uri": text_uri,
                "metadata_path": metadata_path,
                "status": "pending_review",
            }
        )
        self._audit(
            doc_id,
            "manual_review_required",
            "Low-confidence document queued for manual review.",
            {**metadata_payload, "review_item": review_item},
        )
        logger.info(
            "Stage 6 progress [7/7]: document queued for manual review: doc_id=%s, decision_confidence=%.4f",
            doc_id,
            kwargs["decision_confidence"],
        )
        return self._base_response(
            filename,
            prediction,
            kwargs["top_k_predictions"],
            duplicate=False,
            status="pending_review",
            doc_id=doc_id,
            agentic_action="manual_review_required",
            final_class=None,
            confidence_gap=kwargs["confidence_gap"],
            decision_confidence=kwargs["decision_confidence"],
            metadata_path=metadata_path,
            rag_ingestion_status="not_started",
        )

    def list_pending_reviews(self) -> list[dict[str, Any]]:
        return self.review_queue.list_pending()

    def submit_manual_review(self, doc_id: int, corrected_class: str, reviewer_id: str | None = "admin", notes: str | None = None) -> dict[str, Any]:
        if not corrected_class.strip():
            raise ValueError("corrected_class must be a non-empty string")
        if not self.repository:
            raise ValueError("PostgreSQL persistence is required for manual review submission.")
        document = self.repository.get_document_by_id(doc_id)
        if not document:
            raise ValueError(f"Document {doc_id} not found.")
        metadata = self.repository.get_latest_document_metadata(doc_id) or {}
        text_uri = document.get("extracted_text_s3_uri")
        text_key = _local_storage_key(text_uri)
        cleaned_text = clean_document_text(self._read_text_from_storage(text_uri))
        chunk_texts = chunk_document_text(
            cleaned_text,
            chunk_size=self.data_config.chunk_size,
            chunk_overlap=self.data_config.chunk_overlap,
        )
        raw_key = _s3_key_from_uri(document.get("raw_document_s3_uri"))
        final_class = corrected_class.strip()
        filed_uri = None
        filed_key = f"{self.cloud_config.tmf_prefix.rstrip('/')}/{final_class}/{document['file_hash']}_{document['filename']}"
        self._mirror_copy_local(_local_storage_key(document.get("raw_document_s3_uri")), filed_key)
        if raw_key:
            if self.s3_manager:
                filed_uri = self.s3_manager.copy_object(raw_key, filed_key)
                if hasattr(self.s3_manager, "download_file"):
                    self.s3_manager.download_file(filed_key, self._local_path_for_key(filed_key))
        filed_uri = filed_uri or str(self._local_path_for_key(filed_key))
        training_uri = self._register_pending_training(
            None,
            document["filename"],
            document["file_hash"],
            final_class,
            filed_uri or document.get("raw_document_s3_uri"),
        )
        rag_ingested, rag_status = self._index_rag(
            doc_id,
            document["filename"],
            final_class,
            chunk_texts,
            document.get("uploaded_by"),
            document["file_hash"],
        )
        self.repository.update_document_status(doc_id, "pending_training_approval" if rag_ingested else "human_corrected")
        details = {
            **(metadata.get("details") or {}),
            "final_class": final_class,
            "reviewer_id": reviewer_id,
            "review_notes": notes,
            "pending_training_path": training_uri,
            "rag_ingestion_status": rag_status,
        }
        metadata_path = self._upload_metadata(details, document["file_hash"], doc_id)
        details["metadata_path"] = metadata_path
        update_values = {
            "document_type": final_class,
            "final_class": final_class,
            "cloud_storage_path": filed_uri or document.get("raw_document_s3_uri"),
            "status": "pending_training_approval" if rag_ingested else "human_corrected",
            "approval_status": "pending_training_approval",
            "rag_ingested": rag_ingested,
            "metadata_path": metadata_path,
            "details": details,
        }
        self.repository.update_latest_document_metadata(doc_id, update_values)
        self.review_queue.remove(doc_id)
        self._delete_storage_path(document.get("raw_document_s3_uri"))
        self._delete_storage_path(text_uri)
        self._audit(
            doc_id,
            "human_corrected",
            "Manual review completed and document filed.",
            {
                "document_id": doc_id,
                "original_file_name": document["filename"],
                "predicted_class": metadata.get("predicted_class"),
                "confidence_score": metadata.get("confidence_score"),
                "final_class": final_class,
                "reviewer_id": reviewer_id,
                "cloud_path": filed_uri,
                "rag_ingestion_status": rag_status,
                "training_approval_status": "pending_training_approval",
                "notes": notes,
            },
        )
        return {
            "doc_id": doc_id,
            "filename": document["filename"],
            "final_class": final_class,
            "document_status": "pending_training_approval" if rag_ingested else "human_corrected",
            "rag_ingestion_status": rag_status,
            "pending_training_path": training_uri,
            "message": "Manual review submitted. Document is filed and pending training approval.",
        }

    def approve_for_training(self, doc_id: int, approved: bool, reviewer_id: str | None = "admin", notes: str | None = None) -> dict[str, Any]:
        if not self.repository:
            raise ValueError("PostgreSQL persistence is required for training approval.")
        document = self.repository.get_document_by_id(doc_id)
        if not document:
            raise ValueError(f"Document {doc_id} not found.")
        metadata = self.repository.get_latest_document_metadata(doc_id) or {}
        final_class = metadata.get("final_class") or document.get("verified_label")
        if approved and not final_class:
            raise ValueError("Document cannot be approved for training before final_class is set.")
        status = "approved_for_training" if approved else "rejected_for_training"
        training_folder_uri = None
        details_before = metadata.get("details") or {}
        pending_training_path = details_before.get("pending_training_path")
        source_storage_path = pending_training_path or metadata.get("cloud_storage_path") or document.get("raw_document_s3_uri")
        source_key = _local_storage_key(source_storage_path)
        if source_key and final_class:
            prefix = self.cloud_config.approved_training_prefix if approved else self.cloud_config.rejected_training_prefix
            destination_key = f"{prefix.rstrip('/')}/{final_class}/{document['file_hash']}_{document['filename']}"
            local_training_path = self._mirror_copy_local(source_key, destination_key)
            if self.s3_manager and str(source_storage_path or "").startswith("s3://"):
                training_folder_uri = self.s3_manager.copy_object(source_key, destination_key)
                if hasattr(self.s3_manager, "download_file"):
                    self.s3_manager.download_file(destination_key, self._local_path_for_key(destination_key))
            training_folder_uri = training_folder_uri or local_training_path
            if pending_training_path:
                self._delete_storage_path(pending_training_path)
        self.repository.update_document_status(
            doc_id,
            status,
            verified_label=str(final_class) if approved else None,
            used_for_training=False,
        )
        details = {
            **details_before,
            "status": status,
            "approval_status": status,
            "training_reviewer_id": reviewer_id,
            "training_review_notes": notes,
            "training_folder_uri": training_folder_uri,
        }
        metadata_path = self._upload_metadata(details, document["file_hash"], doc_id)
        details["metadata_path"] = metadata_path
        self.repository.update_latest_document_metadata(
            doc_id,
            {"status": status, "approval_status": status, "metadata_path": metadata_path, "details": details},
        )
        self._audit(
            doc_id,
            "approved_for_training" if approved else "rejected_for_training",
            "Training inclusion decision recorded.",
            {
                "document_id": doc_id,
                "final_class": final_class,
                "reviewer_id": reviewer_id,
                "training_approval_status": status,
                "cloud_path": training_folder_uri,
                "notes": notes,
            },
        )
        return {"doc_id": doc_id, "document_status": status, "final_class": final_class, "cloud_path": training_folder_uri}

    def correct_auto_filed(self, doc_id: int, corrected_class: str, reviewer_id: str | None = "admin", notes: str | None = None) -> dict[str, Any]:
        result = self.submit_manual_review(doc_id, corrected_class, reviewer_id=reviewer_id, notes=notes)
        self._audit(
            doc_id,
            "auto_file_corrected",
            "Previously auto-filed document corrected by admin.",
            {"corrected_class": corrected_class, "reviewer_id": reviewer_id, "notes": notes},
        )
        return result

    def metrics(self) -> dict[str, Any]:
        if not self.repository:
            return {
                "total_uploaded_documents": 0,
                "auto_file_rate": 0.0,
                "manual_review_rate": 0.0,
                "human_correction_rate": 0.0,
                "duplicate_detection_count": 0,
                "average_confidence": 0.0,
                "documents_added_to_rag": 0,
                "pending_training_approval": 0,
                "approved_for_training": 0,
                "rejected_for_training": 0,
                "wrong_auto_file_correction_count": 0,
                "status_distribution": {},
            }
        return self.repository.agentic_metrics()
