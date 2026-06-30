"""Small AWS S3 manager used by Stage 4 ingestion and model pipelines."""

from __future__ import annotations

from pathlib import Path

from src.config import CloudConfig
from src.exception import CustomException
from src.logger import logger


class S3Manager:
    """Wrapper around boto3 S3 operations with test-friendly methods."""

    def __init__(self, config: CloudConfig | None = None, client=None) -> None:
        self.config = config or CloudConfig()
        self.client = client or self._create_client()

    def _create_client(self):
        if not self.config.is_configured:
            return None
        import boto3

        kwargs = {"region_name": self.config.aws_region}
        if self.config.aws_access_key_id and self.config.aws_secret_access_key:
            kwargs["aws_access_key_id"] = self.config.aws_access_key_id
            kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
        return boto3.client("s3", **kwargs)

    def _require_client(self):
        if self.client is None or not self.config.aws_s3_bucket_name:
            raise ValueError("AWS S3 is not configured. Set AWS_S3_BUCKET_NAME and AWS credentials if required.")
        return self.client

    def generate_s3_uri(self, key: str) -> str:
        """Return an s3:// URI for a bucket key."""
        if not self.config.aws_s3_bucket_name:
            raise ValueError("AWS_S3_BUCKET_NAME is required to generate S3 URIs.")
        return f"s3://{self.config.aws_s3_bucket_name}/{key}"

    def upload_file(self, local_path: Path | str, key: str, content_type: str | None = None) -> str:
        """Upload a local file to S3 and return its s3:// URI."""
        try:
            client = self._require_client()
            extra_args = {"ContentType": content_type} if content_type else None
            if extra_args:
                client.upload_file(str(local_path), self.config.aws_s3_bucket_name, key, ExtraArgs=extra_args)
            else:
                client.upload_file(str(local_path), self.config.aws_s3_bucket_name, key)
            s3_uri = self.generate_s3_uri(key)
            logger.info("Uploaded %s to %s", local_path, s3_uri)
            return s3_uri
        except Exception as error:
            raise CustomException(error) from error

    def upload_directory(
        self,
        local_dir: Path | str,
        prefix: str,
        skip_existing: bool = True,
    ) -> list[str]:
        """Upload every file from a local directory under an S3 prefix."""
        directory = Path(local_dir)
        if not directory.exists() or not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        uploaded_uris: list[str] = []
        for file_path in sorted(path for path in directory.rglob("*") if path.is_file()):
            relative_key = file_path.relative_to(directory).as_posix()
            key = f"{prefix.rstrip('/')}/{relative_key}"
            if skip_existing and self.file_exists(key):
                logger.info("Skipping existing S3 object: %s", self.generate_s3_uri(key))
                uploaded_uris.append(self.generate_s3_uri(key))
                continue
            uploaded_uris.append(self.upload_file(file_path, key))
        return uploaded_uris

    def download_file(self, key: str, local_path: Path | str) -> Path:
        """Download a file from S3 to a local path."""
        try:
            client = self._require_client()
            destination = Path(local_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(self.config.aws_s3_bucket_name, key, str(destination))
            logger.info("Downloaded s3://%s/%s to %s", self.config.aws_s3_bucket_name, key, destination)
            return destination
        except Exception as error:
            raise CustomException(error) from error

    def ensure_prefix(self, prefix: str) -> str:
        """Create a harmless placeholder object for an S3 prefix if missing."""
        normalized_prefix = prefix.rstrip("/") + "/"
        placeholder_key = normalized_prefix + ".keep"
        if not self.file_exists(placeholder_key):
            client = self._require_client()
            client.put_object(Bucket=self.config.aws_s3_bucket_name, Key=placeholder_key, Body=b"")
            logger.info("Created S3 prefix placeholder: %s", self.generate_s3_uri(placeholder_key))
        return self.generate_s3_uri(normalized_prefix)

    def file_exists(self, key: str) -> bool:
        """Return True if an S3 object exists."""
        try:
            client = self._require_client()
            client.head_object(Bucket=self.config.aws_s3_bucket_name, Key=key)
            return True
        except Exception as error:
            response = getattr(error, "response", {}) or {}
            error_code = str(response.get("Error", {}).get("Code", ""))
            http_status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if error_code in {"404", "NoSuchKey", "NotFound"} or http_status == 404:
                return False
            raise CustomException(error) from error
