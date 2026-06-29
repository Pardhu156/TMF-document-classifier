from pathlib import Path

from src.cloud.s3_manager import S3Manager
from src.config import CloudConfig


class FakeS3Client:
    def __init__(self) -> None:
        self.uploads = []
        self.downloads = []
        self.objects = set()

    def upload_file(self, local_path, bucket, key, ExtraArgs=None):
        self.uploads.append((local_path, bucket, key, ExtraArgs))
        self.objects.add(key)

    def download_file(self, bucket, key, local_path):
        self.downloads.append((bucket, key, local_path))
        Path(local_path).write_text("downloaded", encoding="utf-8")

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            error = Exception("not found")
            error.response = {"Error": {"Code": "404"}}
            raise error
        return {}


class AccessDeniedS3Client(FakeS3Client):
    def head_object(self, Bucket, Key):
        error = Exception("access denied")
        error.response = {"Error": {"Code": "403"}, "ResponseMetadata": {"HTTPStatusCode": 403}}
        raise error


def test_s3_upload_file_returns_uri(tmp_path) -> None:
    local_file = tmp_path / "doc.txt"
    local_file.write_text("hello", encoding="utf-8")
    client = FakeS3Client()
    manager = S3Manager(
        CloudConfig(aws_s3_bucket_name="tmf-test-bucket"),
        client=client,
    )

    uri = manager.upload_file(local_file, "raw_documents/doc.txt", content_type="text/plain")

    assert uri == "s3://tmf-test-bucket/raw_documents/doc.txt"
    assert client.uploads[0][1] == "tmf-test-bucket"


def test_s3_file_exists_uses_head_object(tmp_path) -> None:
    client = FakeS3Client()
    client.objects.add("raw_documents/doc.txt")
    manager = S3Manager(CloudConfig(aws_s3_bucket_name="tmf-test-bucket"), client=client)

    assert manager.file_exists("raw_documents/doc.txt") is True
    assert manager.file_exists("missing.txt") is False


def test_s3_file_exists_does_not_hide_permission_errors() -> None:
    manager = S3Manager(CloudConfig(aws_s3_bucket_name="tmf-test-bucket"), client=AccessDeniedS3Client())

    try:
        manager.file_exists("private.txt")
    except Exception as error:
        assert "access denied" in str(error)
    else:
        raise AssertionError("Expected permission errors to be raised, not treated as missing files.")
