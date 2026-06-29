"""Conditional retraining coordinator for verified cloud data."""

from __future__ import annotations

from datetime import datetime, timezone
import re

from src.config import MLOpsConfig, RetrainingConfig
from src.database.repository import TMFRepository
from src.exception import CustomException
from src.logger import logger


class ConditionalRetrainingPipeline:
    """Start retraining only when new verified documents are available.

    This stage intentionally coordinates retraining readiness and metadata.
    The expensive fine-tuning job can be connected to a GPU worker later without
    changing the API contract.
    """

    def __init__(
        self,
        repository: TMFRepository | None = None,
        retraining_config: RetrainingConfig | None = None,
        mlops_config: MLOpsConfig | None = None,
    ) -> None:
        self.repository = repository
        self.retraining_config = retraining_config or RetrainingConfig()
        self.mlops_config = mlops_config or MLOpsConfig()

    def run(self) -> dict:
        """Run conditional retraining checks and return the action taken."""
        try:
            if self.repository is None:
                logger.warning("Retraining skipped because PostgreSQL repository is not configured.")
                return {
                    "status": "skipped",
                    "message": "PostgreSQL is not configured. Retraining skipped.",
                    "active_model_version": self.mlops_config.model_version,
                }

            active_model = self.repository.get_active_model_version()
            verified_documents = self.repository.get_new_verified_documents()
            if len(verified_documents) < self.retraining_config.retrain_min_new_documents:
                logger.info("No new training data found. Retraining skipped.")
                self.repository.save_audit_log(
                    event_type="retraining_skipped",
                    entity_type="model",
                    entity_id=active_model.get("model_version") if active_model else self.mlops_config.model_version,
                    message="No new verified data found. Retraining skipped.",
                    details={"new_verified_documents": len(verified_documents)},
                )
                return {
                    "status": "skipped",
                    "message": "No new verified data found. Retraining skipped.",
                    "active_model_version": active_model.get("model_version") if active_model else self.mlops_config.model_version,
                    "new_verified_documents": len(verified_documents),
                }

            next_model_version = self._next_model_version(active_model)
            dataset_version = f"{self.mlops_config.dataset_version}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            model_record = self.repository.save_model_version(
                {
                    "model_version": next_model_version,
                    "model_name": "emilyalsentzer/Bio_ClinicalBERT",
                    "model_artifact_s3_uri": None,
                    "dataset_version": dataset_version,
                    "metrics": {"status": "pending_gpu_fine_tuning"},
                    "is_active": False,
                }
            )
            self.repository.save_audit_log(
                event_type="retraining_started",
                entity_type="model",
                entity_id=next_model_version,
                message="Verified data found. Retraining pipeline prepared a new model version record.",
                details={
                    "new_verified_documents": len(verified_documents),
                    "dataset_version": dataset_version,
                    "note": "GPU fine-tuning should continue from the active checkpoint and save a new version.",
                },
            )
            logger.info(
                "Conditional retraining started: %d verified documents, next model version %s.",
                len(verified_documents),
                next_model_version,
            )
            return {
                "status": "started",
                "message": "New verified data found. Retraining pipeline started.",
                "active_model_version": active_model.get("model_version") if active_model else self.mlops_config.model_version,
                "candidate_model_version": next_model_version,
                "new_verified_documents": len(verified_documents),
                "dataset_version": dataset_version,
                "model_record": model_record,
            }
        except Exception as error:
            logger.exception("Conditional retraining pipeline failed")
            raise CustomException(error) from error

    def _next_model_version(self, active_model: dict | None) -> str:
        current = (active_model or {}).get("model_version") or self.mlops_config.model_version or "model_v1"
        model_match = re.fullmatch(r"model_v(\d+)", str(current))
        if model_match:
            return f"model_v{int(model_match.group(1)) + 1}"
        semantic_match = re.fullmatch(r"v(\d+)(?:\.\d+)*", str(current))
        if semantic_match:
            return f"model_v{int(semantic_match.group(1)) + 1}"
        return f"{current}_next"
