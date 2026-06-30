"""Manual cloud bootstrap uploads for existing training/model assets."""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any

from src.cloud.s3_manager import S3Manager
from src.config import CloudConfig, DataIngestionConfig, EvaluationConfig, MetadataConfig, ModelTrainingConfig, RAGConfig
from src.exception import CustomException
from src.logger import logger


DEFAULT_STAGE4_BUCKET = os.getenv("AWS_S3_BUCKET_NAME", "")


class CloudBootstrapPipeline:
    """Upload existing local project assets that are not created by /predict-file.

    This is intentionally separate from the prediction ingestion pipeline:

    - this pipeline uploads existing training/model/report assets to S3
    - /predict-file handles new prediction documents and PostgreSQL persistence
    """

    def __init__(
        self,
        cloud_config: CloudConfig | None = None,
        s3_manager: S3Manager | None = None,
        bucket_name: str | None = None,
    ) -> None:
        bucket_name = bucket_name or DEFAULT_STAGE4_BUCKET
        self.cloud_config = cloud_config or CloudConfig(aws_s3_bucket_name=bucket_name)
        if not self.cloud_config.aws_s3_bucket_name:
            self.cloud_config.aws_s3_bucket_name = bucket_name
        self.s3_manager = s3_manager or S3Manager(self.cloud_config)
        self.data_config = DataIngestionConfig()
        self.training_config = ModelTrainingConfig()
        self.evaluation_config = EvaluationConfig()
        self.metadata_config = MetadataConfig()
        self.rag_config = RAGConfig()

    def run(self, skip_existing: bool = True) -> dict[str, Any]:
        """Upload existing local assets to S3 and return a manifest."""
        try:
            manifest = {
                "bucket": self.cloud_config.aws_s3_bucket_name,
                "skip_existing": skip_existing,
                "uploads": {
                    "raw_training_data": [],
                    "model_artifacts": [],
                    "reports": [],
                    "rag_artifact_prefixes": [],
                },
            }

            manifest["uploads"]["raw_training_data"] = self.upload_training_data(skip_existing=skip_existing)
            manifest["uploads"]["model_artifacts"] = self.upload_model_artifacts(skip_existing=skip_existing)
            manifest["uploads"]["reports"] = self.upload_reports(skip_existing=skip_existing)
            manifest["uploads"]["rag_artifact_prefixes"] = self.ensure_rag_artifact_prefixes()

            logger.info("Cloud bootstrap upload complete: %s", manifest)
            return manifest
        except Exception as error:
            logger.exception("Cloud bootstrap upload failed")
            raise CustomException(error) from error

    def upload_training_data(self, skip_existing: bool = True) -> list[str]:
        """Upload local data/ folder as raw training data backup."""
        if not self.data_config.raw_data_dir.exists():
            logger.warning("Skipping training data upload because %s does not exist.", self.data_config.raw_data_dir)
            return []
        return self.s3_manager.upload_directory(
            self.data_config.raw_data_dir,
            "raw_training_data/data",
            skip_existing=skip_existing,
        )

    def upload_model_artifacts(self, skip_existing: bool = True) -> list[str]:
        """Upload current inference model folder and label encoder."""
        uploaded: list[str] = []
        if self.training_config.save_model_dir.exists():
            uploaded.extend(
                self.s3_manager.upload_directory(
                    self.training_config.save_model_dir,
                    "model_artifacts/model_v1/saved_bioclinicalbert_tmf_3class",
                    skip_existing=skip_existing,
                )
            )
        else:
            logger.warning("Skipping model directory upload because %s does not exist.", self.training_config.save_model_dir)

        if self.training_config.label_encoder_path.exists():
            uploaded.append(
                self._upload_file_if_needed(
                    self.training_config.label_encoder_path,
                    "model_artifacts/model_v1/label_encoder.pkl",
                    skip_existing=skip_existing,
                )
            )
        else:
            logger.warning("Skipping label encoder upload because %s does not exist.", self.training_config.label_encoder_path)
        return uploaded

    def upload_reports(self, skip_existing: bool = True) -> list[str]:
        """Upload metadata and evaluation report files."""
        uploaded: list[str] = []
        if self.metadata_config.metadata_dir.exists():
            uploaded.extend(
                self.s3_manager.upload_directory(
                    self.metadata_config.metadata_dir,
                    "reports/metadata",
                    skip_existing=skip_existing,
                )
            )

        report_paths = (
            self.evaluation_config.metrics_path,
            self.evaluation_config.confusion_matrix_path,
            self.evaluation_config.run_metadata_path,
        )
        for report_path in report_paths:
            if report_path.exists():
                uploaded.append(
                    self._upload_file_if_needed(
                        report_path,
                        f"reports/{report_path.name}",
                        skip_existing=skip_existing,
                    )
                )
            else:
                logger.warning("Skipping report upload because %s does not exist.", report_path)
        return uploaded

    def ensure_rag_artifact_prefixes(self) -> list[str]:
        """Ensure optional RAG artifact prefixes exist in the existing Stage 4 bucket."""
        if not hasattr(self.s3_manager, "ensure_prefix"):
            logger.info("S3 manager does not support ensure_prefix; skipping RAG prefix initialization.")
            return []
        prefixes = (
            self.rag_config.rag_artifacts_s3_prefix,
            self.rag_config.model_backup_s3_prefix,
            self.rag_config.rag_ingestion_reports_s3_prefix,
            self.rag_config.rag_evaluation_s3_prefix,
            self.rag_config.rag_failed_ingestions_s3_prefix,
        )
        return [self.s3_manager.ensure_prefix(prefix) for prefix in prefixes]

    def _upload_file_if_needed(self, local_path: Path, key: str, skip_existing: bool = True) -> str:
        if skip_existing and self.s3_manager.file_exists(key):
            s3_uri = self.s3_manager.generate_s3_uri(key)
            logger.info("Skipping existing S3 object: %s", s3_uri)
            return s3_uri
        return self.s3_manager.upload_file(local_path, key)
