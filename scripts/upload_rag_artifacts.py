"""Upload optional RAG artifacts and local embedding model backup to existing S3 bucket."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloud.s3_manager import S3Manager
from src.config import CloudConfig, RAGConfig


def _safe_upload_directory(s3: S3Manager, local_dir: Path, prefix: str) -> list[str]:
    if not local_dir.exists() or not local_dir.is_dir():
        return []
    return s3.upload_directory(local_dir, prefix, skip_existing=True)


def main() -> None:
    cloud_config = CloudConfig()
    rag_config = RAGConfig()
    if not cloud_config.aws_s3_bucket_name:
        print("AWS_S3_BUCKET_NAME is not configured. Skipping RAG artifact upload.")
        return

    s3 = S3Manager(cloud_config)
    uploads: dict[str, list[str]] = {
        "embedding_model_backup": [],
        "ingestion_reports": [],
        "rag_evaluation": [],
        "failed_ingestions": [],
    }

    uploads["embedding_model_backup"] = _safe_upload_directory(
        s3,
        Path(rag_config.local_model_dir),
        rag_config.model_backup_s3_prefix,
    )
    uploads["ingestion_reports"] = _safe_upload_directory(
        s3,
        PROJECT_ROOT / "artifacts" / "rag" / "ingestion-reports",
        rag_config.rag_ingestion_reports_s3_prefix,
    )
    uploads["rag_evaluation"] = _safe_upload_directory(
        s3,
        PROJECT_ROOT / "artifacts" / "rag" / "rag-evaluation",
        rag_config.rag_evaluation_s3_prefix,
    )
    uploads["failed_ingestions"] = _safe_upload_directory(
        s3,
        PROJECT_ROOT / "artifacts" / "rag" / "failed-ingestions",
        rag_config.rag_failed_ingestions_s3_prefix,
    )

    print(json.dumps({"bucket": cloud_config.aws_s3_bucket_name, "uploads": uploads}, indent=2))


if __name__ == "__main__":
    main()
