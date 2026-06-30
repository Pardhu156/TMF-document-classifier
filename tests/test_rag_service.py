from src.config import RAGConfig
from src.rag.generator import NOT_FOUND_ANSWER
from src.rag.service import RAGService


class FakeVectorStore:
    def list_documents(self):
        return [
            {
                "document_id": "1",
                "file_name": "protocol.pdf",
                "predicted_class": "protocol",
                "uploaded_by": "admin",
                "created_at": "2026-06-30T10:00:00Z",
            }
        ]

    def document_status(self, document_id: str):
        return "indexed"


class FakeEmbeddingClient:
    def embed_query(self, question: str):
        return [1.0, 0.0]


class FakeCache:
    def __init__(self, exact=None, semantic=None) -> None:
        self.exact = exact
        self.semantic = semantic
        self.saved = False

    def get_exact(self, question: str, scope: str):
        return self.exact

    def search_semantic(self, question_embedding, scope: str):
        return self.semantic

    def set(self, **kwargs) -> None:
        self.saved = True

    def get_document_status(self, document_id: str):
        return None


class FakeGenerator:
    def generate(self, question: str, chunks):
        return "The document describes study objectives."


class FakeRetriever:
    def retrieve(self, question: str, document_id=None, predicted_class=None, query_embedding=None, filters=None):
        self.filters = filters
        return (
            [
                {
                    "document_id": "1",
                    "file_name": "protocol.pdf",
                    "chunk_id": "1:0",
                    "chunk_index": 0,
                    "chunk_text": "Study objectives",
                    "hybrid_score": 0.9,
                }
            ],
            query_embedding,
        )


def _service(cache=None) -> RAGService:
    service = RAGService(
        config=RAGConfig(redis_url=None),
        vector_store=FakeVectorStore(),
        embedding_client=FakeEmbeddingClient(),
        cache=cache or FakeCache(),
        generator=FakeGenerator(),
    )
    service.retriever = FakeRetriever()
    return service


def test_rag_service_generates_and_caches_answer() -> None:
    cache = FakeCache()
    service = _service(cache=cache)

    response = service.ask("What are the objectives?", document_id="1")

    assert response["answer"] == "The document describes study objectives."
    assert response["cache_hit"] is False
    assert response["cache_type"] == "none"
    assert response["sources"][0]["chunk_id"] == "1:0"
    assert cache.saved is True


def test_rag_service_returns_exact_cache_hit() -> None:
    service = _service(
        cache=FakeCache(
            exact={
                "answer": "Cached answer",
                "source_chunks": [{"chunk_id": "1:0", "document_id": "1"}],
            }
        )
    )

    response = service.ask("What are the objectives?", document_id="1")

    assert response["answer"] == "Cached answer"
    assert response["cache_hit"] is True
    assert response["cache_type"] == "exact"


def test_rag_not_found_answer_constant_matches_contract() -> None:
    assert NOT_FOUND_ANSWER == "I could not find enough information in the uploaded documents."
