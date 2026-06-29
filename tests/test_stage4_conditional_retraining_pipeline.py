from src.pipeline.conditional_retraining_pipeline import ConditionalRetrainingPipeline


class FakeRetrainingRepository:
    def __init__(self, verified_documents=None) -> None:
        self.verified_documents = verified_documents or []
        self.audit_logs = []
        self.model_versions = []

    def get_active_model_version(self):
        return {"model_version": "model_v1"}

    def get_new_verified_documents(self):
        return self.verified_documents

    def save_audit_log(self, **kwargs):
        self.audit_logs.append(kwargs)

    def save_model_version(self, data):
        record = {"model_version_id": 1, **data}
        self.model_versions.append(record)
        return record


def test_retraining_pipeline_skips_when_no_verified_data() -> None:
    repo = FakeRetrainingRepository()
    result = ConditionalRetrainingPipeline(repository=repo).run()

    assert result["status"] == "skipped"
    assert "No new verified data" in result["message"]
    assert repo.audit_logs


def test_retraining_pipeline_starts_when_verified_data_exists() -> None:
    repo = FakeRetrainingRepository(verified_documents=[{"doc_id": 1, "verified_label": "protocol"}])
    result = ConditionalRetrainingPipeline(repository=repo).run()

    assert result["status"] == "started"
    assert result["candidate_model_version"] == "model_v2"
    assert repo.model_versions


def test_retraining_version_from_semantic_model_version() -> None:
    pipeline = ConditionalRetrainingPipeline(repository=FakeRetrainingRepository())

    assert pipeline._next_model_version({"model_version": "v1.0.0"}) == "model_v2"
