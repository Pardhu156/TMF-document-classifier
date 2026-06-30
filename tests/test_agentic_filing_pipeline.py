import asyncio

from src.agentic_filing.pipeline import AgenticTMFFilingPipeline
from src.agentic_filing.review_queue import ManualReviewQueue
from src.config import AgenticFilingConfig, CloudConfig


class FakeUpload:
    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._content = content

    async def read(self) -> bytes:
        return self._content


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, str | bytes] = {}
        self.config = CloudConfig(aws_s3_bucket_name="fake-bucket")

    def upload_file(self, local_path, key, content_type=None):
        self.objects[key] = bytes(open(local_path, "rb").read())
        return f"s3://fake-bucket/{key}"

    def upload_text(self, text, key, content_type="text/plain"):
        self.objects[key] = text
        return f"s3://fake-bucket/{key}"

    def copy_object(self, source_key, destination_key):
        self.objects[destination_key] = self.objects[source_key]
        return f"s3://fake-bucket/{destination_key}"

    def read_text(self, key):
        return str(self.objects[key])


class FakeRepository:
    def __init__(self) -> None:
        self.docs: dict[int, dict] = {}
        self.docs_by_hash: dict[str, dict] = {}
        self.predictions: dict[int, dict] = {}
        self.metadata: dict[int, dict] = {}
        self.audit_logs: list[dict] = []
        self.next_doc_id = 1

    def get_document_by_hash(self, file_hash):
        return self.docs_by_hash.get(file_hash)

    def get_existing_prediction_by_doc_id(self, doc_id):
        return self.predictions.get(doc_id)

    def save_document(self, document_data):
        doc = {"doc_id": self.next_doc_id, **document_data}
        self.next_doc_id += 1
        self.docs[doc["doc_id"]] = doc
        self.docs_by_hash[doc["file_hash"]] = doc
        return doc

    def get_document_by_id(self, doc_id):
        return self.docs.get(doc_id)

    def update_document_status(self, doc_id, document_status, verified_label=None, used_for_training=None):
        doc = self.docs[doc_id]
        doc["document_status"] = document_status
        if verified_label is not None:
            doc["verified_label"] = verified_label
        if used_for_training is not None:
            doc["used_for_training"] = used_for_training
        return doc

    def save_prediction(self, doc_id, prediction_data):
        prediction = {"prediction_id": len(self.predictions) + 1, "doc_id": doc_id, **prediction_data}
        self.predictions[doc_id] = prediction
        return prediction

    def save_chunks(self, doc_id, chunks):
        return [{"chunk_id": index + 1, "doc_id": doc_id, **chunk} for index, chunk in enumerate(chunks)]

    def save_chunk_predictions(self, doc_id, chunk_predictions):
        return [{"chunk_prediction_id": index + 1, "doc_id": doc_id, **row} for index, row in enumerate(chunk_predictions)]

    def save_document_metadata(self, metadata_data):
        self.metadata[metadata_data["doc_id"]] = metadata_data
        return metadata_data

    def get_latest_document_metadata(self, doc_id):
        return self.metadata.get(doc_id)

    def update_latest_document_metadata(self, doc_id, values):
        self.metadata.setdefault(doc_id, {}).update(values)
        return self.metadata[doc_id]

    def save_audit_log(self, event_type, entity_type, entity_id, message, details=None):
        row = {
            "event_type": event_type,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "message": message,
            "details": details,
        }
        self.audit_logs.append(row)
        return row

    def agentic_metrics(self):
        return {"total_uploaded_documents": len(self.docs)}


class FakeRAGIndexer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def index_document(self, **kwargs):
        self.calls.append(kwargs)
        return len(kwargs["chunk_texts"])


def predictor(label: str, confidence: float):
    def _predict(_text: str) -> dict:
        return {
            "predicted_label": label,
            "confidence": confidence,
            "model_confidence": confidence,
            "vote_confidence": 1.0,
            "margin_confidence": 1.0,
            "requires_review": confidence < 0.8,
            "decision_status": "auto_classify" if confidence >= 0.8 else "human_review",
        }

    return _predict


def make_pipeline(repository, s3, rag, confidence: float, threshold: float = 0.9) -> AgenticTMFFilingPipeline:
    return AgenticTMFFilingPipeline(
        agentic_config=AgenticFilingConfig(auto_approval_threshold=threshold, min_confidence_gap=0.1),
        cloud_config=CloudConfig(aws_s3_bucket_name="fake-bucket"),
        repository=repository,
        s3_manager=s3,
        review_queue=ManualReviewQueue(client=None),
        predictor=predictor("protocol", confidence),
        rag_indexer=rag,
    )


def test_high_confidence_document_is_auto_filed_and_rag_ingested() -> None:
    repo, s3, rag = FakeRepository(), FakeS3(), FakeRAGIndexer()
    pipeline = make_pipeline(repo, s3, rag, confidence=0.99)

    result = asyncio.run(pipeline.run(FakeUpload("protocol.txt", b"This protocol describes the study objective.")))

    assert result["agentic_action"] == "auto_filed"
    assert result["final_class"] == "protocol"
    assert result["document_status"] == "pending_training_approval"
    assert len(rag.calls) == 1
    assert any("agentic_tmf_workspace/tmf/protocol/" in key for key in s3.objects)


def test_low_confidence_document_goes_to_manual_review_without_rag_ingestion() -> None:
    repo, s3, rag = FakeRepository(), FakeS3(), FakeRAGIndexer()
    pipeline = make_pipeline(repo, s3, rag, confidence=0.50, threshold=1.01)

    result = asyncio.run(pipeline.run(FakeUpload("unknown.txt", b"This document is ambiguous.")))

    assert result["agentic_action"] == "manual_review_required"
    assert result["document_status"] == "pending_review"
    assert pipeline.list_pending_reviews()
    assert rag.calls == []


def test_human_correction_files_document_and_then_ingests_rag() -> None:
    repo, s3, rag = FakeRepository(), FakeS3(), FakeRAGIndexer()
    pipeline = make_pipeline(repo, s3, rag, confidence=0.50, threshold=1.01)
    result = asyncio.run(pipeline.run(FakeUpload("review.txt", b"This reviewed protocol has inclusion criteria.")))

    corrected = pipeline.submit_manual_review(result["doc_id"], corrected_class="protocol")

    assert corrected["final_class"] == "protocol"
    assert corrected["document_status"] == "pending_training_approval"
    assert len(rag.calls) == 1


def test_duplicate_document_upload_is_skipped() -> None:
    repo, s3, rag = FakeRepository(), FakeS3(), FakeRAGIndexer()
    pipeline = make_pipeline(repo, s3, rag, confidence=0.99)
    upload = FakeUpload("duplicate.txt", b"Same protocol document.")

    first = asyncio.run(pipeline.run(upload))
    second = asyncio.run(pipeline.run(FakeUpload("duplicate.txt", b"Same protocol document.")))

    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert second["doc_id"] == first["doc_id"]


def test_rag_ingestion_happens_only_after_final_class_confirmation() -> None:
    repo, s3, rag = FakeRepository(), FakeS3(), FakeRAGIndexer()
    pipeline = make_pipeline(repo, s3, rag, confidence=0.40, threshold=1.01)
    result = asyncio.run(pipeline.run(FakeUpload("needs-review.txt", b"Needs human TMF filing.")))

    assert rag.calls == []
    pipeline.submit_manual_review(result["doc_id"], corrected_class="safety_report")

    assert rag.calls[0]["predicted_class"] == "safety_report"
