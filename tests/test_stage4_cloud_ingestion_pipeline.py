import asyncio

from src.config import CloudConfig
from src.pipeline.cloud_ingestion_pipeline import CloudIngestionPipeline


class FakeUploadFile:
    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._content = content

    async def read(self) -> bytes:
        return self._content


class FakeRepository:
    def __init__(self, duplicate: bool = False) -> None:
        self.duplicate = duplicate
        self.saved_documents = []
        self.saved_chunks = []
        self.saved_predictions = []
        self.saved_chunk_predictions = []
        self.audit_logs = []

    def get_document_by_hash(self, file_hash):
        if self.duplicate:
            return {"doc_id": 7, "filename": "existing.txt", "document_status": "predicted_unverified"}
        return None

    def get_existing_prediction_by_doc_id(self, doc_id):
        return {
            "predicted_label": "protocol",
            "confidence": 0.95,
            "model_confidence": 0.95,
            "vote_confidence": 1.0,
            "margin_confidence": 1.0,
            "requires_review": False,
            "decision_status": "auto_classify",
            "num_chunks": 1,
            "chunk_predictions": {"protocol": 1},
        }

    def save_document(self, data):
        record = {"doc_id": 1, **data}
        self.saved_documents.append(record)
        return record

    def save_chunks(self, doc_id, chunks):
        rows = [{"chunk_id": index + 1, "doc_id": doc_id, **chunk} for index, chunk in enumerate(chunks)]
        self.saved_chunks.extend(rows)
        return rows

    def save_prediction(self, doc_id, prediction):
        self.saved_predictions.append({"doc_id": doc_id, **prediction})

    def save_chunk_predictions(self, doc_id, predictions):
        self.saved_chunk_predictions.extend(predictions)

    def save_audit_log(self, **kwargs):
        self.audit_logs.append(kwargs)


class FakeS3Manager:
    def __init__(self) -> None:
        self.uploaded = []

    def upload_file(self, local_path, key, content_type=None):
        self.uploaded.append((str(local_path), key, content_type))
        return f"s3://bucket/{key}"


def test_cloud_ingestion_pipeline_duplicate_skip() -> None:
    repo = FakeRepository(duplicate=True)
    s3 = FakeS3Manager()
    pipeline = CloudIngestionPipeline(
        cloud_config=CloudConfig(aws_s3_bucket_name="bucket", allow_duplicate_documents=False),
        repository=repo,
        s3_manager=s3,
    )

    result = asyncio.run(pipeline.run(FakeUploadFile("duplicate.txt", b"same file")))

    assert result["duplicate"] is True
    assert result["predicted_label"] == "protocol"
    assert s3.uploaded == []


def test_cloud_ingestion_pipeline_new_file_persists(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.pipeline.cloud_ingestion_pipeline.predict_text",
        lambda text: {
            "predicted_label": "protocol",
            "confidence": 0.9,
            "model_confidence": 0.9,
            "vote_confidence": 1.0,
            "margin_confidence": 1.0,
            "requires_review": False,
            "decision_status": "auto_classify",
        },
    )
    monkeypatch.setattr("src.rag.service.RAGIndexer.is_configured", classmethod(lambda cls: False))
    repo = FakeRepository(duplicate=False)
    s3 = FakeS3Manager()
    pipeline = CloudIngestionPipeline(
        cloud_config=CloudConfig(aws_s3_bucket_name="bucket"),
        repository=repo,
        s3_manager=s3,
    )

    result = asyncio.run(pipeline.run(FakeUploadFile("new.txt", b"This is a protocol document with study objectives.")))

    assert result["duplicate"] is False
    assert result["predicted_label"] == "protocol"
    assert repo.saved_documents
    assert repo.saved_chunks
    assert repo.saved_predictions
    assert len(s3.uploaded) == 2
