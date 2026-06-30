from fastapi.testclient import TestClient

import app as api_app


client = TestClient(api_app.app)


class FakeRAGService:
    @classmethod
    def is_configured(cls) -> bool:
        return True

    def ask(
        self,
        question: str,
        document_id=None,
        predicted_class=None,
        file_name=None,
        source_type=None,
        verification_status=None,
        scope="master",
    ):
        if not question.strip():
            raise ValueError("question must be a non-empty string")
        return {
            "answer": "The protocol describes study objectives.",
            "sources": [
                {
                    "document_id": document_id or "1",
                    "file_name": "protocol.pdf",
                    "page_no": None,
                    "chunk_id": "1:0",
                    "chunk_index": 0,
                    "score": 0.91,
                }
            ],
            "cache_hit": False,
            "cache_type": "none",
            "retrieved_chunks": [{"chunk_id": "1:0", "chunk_text": "Study objectives"}],
            "latency_ms": 12,
            "retrieval_scope": scope,
            "matched_file_name": file_name,
            "clarification_required": False,
            "candidate_files": [],
        }

    def list_documents(self):
        return [
            {
                "document_id": "1",
                "file_name": "protocol.pdf",
                "predicted_class": "protocol",
                "source_type": "MASTER_DATA",
                "verification_status": "verified",
                "uploaded_by": "admin",
                "created_at": "2026-06-30T10:00:00Z",
            }
        ]

    def status(self, document_id: str):
        return "indexed"


class DisabledRAGService(FakeRAGService):
    @classmethod
    def is_configured(cls) -> bool:
        return False


def test_rag_ask_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "RAGService", FakeRAGService)

    response = client.post("/rag/ask", json={"question": "What does this document describe?", "document_id": "1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "The protocol describes study objectives."
    assert payload["sources"][0]["chunk_id"] == "1:0"


def test_rag_ask_empty_question_returns_400(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "RAGService", FakeRAGService)

    response = client.post("/rag/ask", json={"question": "   "})

    assert response.status_code == 400


def test_rag_documents_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "RAGService", FakeRAGService)

    response = client.get("/rag/documents")

    assert response.status_code == 200
    assert response.json()[0]["document_id"] == "1"


def test_rag_status_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "RAGService", FakeRAGService)

    response = client.get("/rag/status/1")

    assert response.status_code == 200
    assert response.json() == {"document_id": "1", "status": "indexed"}


def test_rag_metrics_endpoint_does_not_require_rag_configuration(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "log_rag_metrics_to_mlflow", lambda params, metrics: None)

    response = client.get("/rag/metrics")

    assert response.status_code == 200
    assert "total_questions" in response.json()["metrics"]


def test_rag_ask_returns_503_when_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "RAGService", DisabledRAGService)

    response = client.post("/rag/ask", json={"question": "What does this document describe?"})

    assert response.status_code == 503


def test_rag_index_master_data_endpoint(monkeypatch) -> None:
    class FakeMasterDataPipeline:
        @classmethod
        def is_configured(cls) -> bool:
            return True

        def run(self):
            return {
                "indexed_documents": 1,
                "skipped_documents": 0,
                "indexed_chunks": 3,
                "source_type": "MASTER_DATA",
                "master_data_dir": "MASTER_DATA",
                "message": "MASTER_DATA indexing complete.",
            }

    monkeypatch.setattr(api_app, "MasterDataIngestionPipeline", FakeMasterDataPipeline)

    response = client.post("/rag/index-master-data")

    assert response.status_code == 200
    assert response.json()["source_type"] == "MASTER_DATA"
