from src.config import RAGConfig
from src.rag.access_control import ACCESS_PERMISSION_MESSAGE
from src.rag.generator import NOT_FOUND_ANSWER
from src.rag.service import RAGService


class FakeVectorStore:
    def list_documents(self):
        return [
            {
                "document_id": "1",
                "file_name": "protocol.pdf",
                "predicted_class": "protocol",
                "source_type": "MASTER_DATA",
                "verification_status": "verified",
                "access_level": "User",
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


class RoleAwareVectorStore(FakeVectorStore):
    def list_documents(self):
        return [
            {
                "document_id": "user-doc",
                "file_name": "user_protocol.pdf",
                "predicted_class": "protocol",
                "source_type": "MASTER_DATA",
                "verification_status": "verified",
                "access_level": "User",
                "uploaded_by": "user@test.com",
                "created_at": "2026-06-30T10:00:00Z",
            },
            {
                "document_id": "manager-doc",
                "file_name": "manager_protocol.pdf",
                "predicted_class": "protocol",
                "source_type": "MASTER_DATA",
                "verification_status": "verified",
                "access_level": "Manager",
                "uploaded_by": "manager@test.com",
                "created_at": "2026-06-30T10:00:00Z",
            },
            {
                "document_id": "admin-doc",
                "file_name": "admin_protocol.pdf",
                "predicted_class": "protocol",
                "source_type": "MASTER_DATA",
                "verification_status": "verified",
                "access_level": "Admin",
                "uploaded_by": "admin@test.com",
                "created_at": "2026-06-30T10:00:00Z",
            },
        ]


class RoleAwareRetriever:
    def __init__(self) -> None:
        self.filters = None

    def retrieve(self, question: str, document_id=None, predicted_class=None, query_embedding=None, filters=None):
        self.filters = filters
        allowed_levels = set(filters.get("access_level") or [])
        requested_document = filters.get("document_id")
        chunks = [
            {
                "document_id": "user-doc",
                "file_name": "user_protocol.pdf",
                "chunk_id": "user-doc:0",
                "chunk_index": 0,
                "chunk_text": "User protocol objectives",
                "hybrid_score": 0.9,
                "access_level": "User",
            },
            {
                "document_id": "manager-doc",
                "file_name": "manager_protocol.pdf",
                "chunk_id": "manager-doc:0",
                "chunk_index": 0,
                "chunk_text": "Manager protocol objectives",
                "hybrid_score": 0.8,
                "access_level": "Manager",
            },
            {
                "document_id": "admin-doc",
                "file_name": "admin_protocol.pdf",
                "chunk_id": "admin-doc:0",
                "chunk_index": 0,
                "chunk_text": "Admin protocol objectives",
                "hybrid_score": 0.7,
                "access_level": "Admin",
            },
        ]
        filtered = [chunk for chunk in chunks if chunk["access_level"] in allowed_levels]
        if requested_document:
            filtered = [chunk for chunk in filtered if chunk["document_id"] == requested_document]
        return filtered, query_embedding


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


def _rbac_service() -> tuple[RAGService, RoleAwareRetriever]:
    service = RAGService(
        config=RAGConfig(redis_url=None),
        vector_store=RoleAwareVectorStore(),
        embedding_client=FakeEmbeddingClient(),
        cache=FakeCache(),
        generator=FakeGenerator(),
    )
    retriever = RoleAwareRetriever()
    service.retriever = retriever
    return service, retriever


def test_user_cannot_retrieve_manager_or_admin_documents() -> None:
    service, retriever = _rbac_service()

    response = service.ask(
        "What are the manager objectives?",
        document_id="manager-doc",
        current_user={"id": 1, "role": "User"},
    )

    assert response["answer"] == ACCESS_PERMISSION_MESSAGE
    assert response["sources"] == []
    assert retriever.filters is None


def test_manager_retrieves_user_and_manager_documents() -> None:
    service, retriever = _rbac_service()

    response = service.ask("What are the objectives?", scope="all", current_user={"id": 2, "role": "Manager"})

    assert response["answer"] == "The document describes study objectives."
    assert {source["document_id"] for source in response["sources"]} == {"user-doc", "manager-doc"}
    assert retriever.filters["access_level"] == ["User", "Manager"]


def test_admin_retrieves_all_access_levels() -> None:
    service, retriever = _rbac_service()

    response = service.ask("What are the objectives?", scope="all", current_user={"id": 3, "role": "Admin"})

    assert {source["document_id"] for source in response["sources"]} == {"user-doc", "manager-doc", "admin-doc"}
    assert retriever.filters["access_level"] == ["User", "Manager", "Admin"]


def test_authorized_query_continues_to_work_normally() -> None:
    service, retriever = _rbac_service()

    response = service.ask(
        "What are the user objectives?",
        document_id="user-doc",
        current_user={"id": 1, "role": "User"},
    )

    assert response["answer"] == "The document describes study objectives."
    assert response["sources"][0]["document_id"] == "user-doc"
    assert retriever.filters["access_level"] == ["User"]
