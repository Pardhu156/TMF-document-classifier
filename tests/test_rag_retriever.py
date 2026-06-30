from src.config import RAGConfig
from src.rag.retriever import HybridRetriever


class FakeVectorStore:
    def __init__(self) -> None:
        self.semantic_embedding_seen = None

    def semantic_search(self, query_embedding, top_k, document_id=None, predicted_class=None, filters=None):
        self.semantic_embedding_seen = query_embedding
        self.filters_seen = filters
        return [
            {"chunk_id": "a", "chunk_text": "protocol objectives", "semantic_score": 0.9},
            {"chunk_id": "b", "chunk_text": "safety narrative", "semantic_score": 0.4},
        ]

    def keyword_search(self, question, top_k, document_id=None, predicted_class=None, filters=None):
        return [
            {"chunk_id": "a", "chunk_text": "protocol objectives", "keyword_score": 0.3},
            {"chunk_id": "c", "chunk_text": "analysis plan", "keyword_score": 0.7},
        ]


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, question: str):
        self.calls += 1
        return [0.1, 0.2]


def test_hybrid_retriever_merges_semantic_and_keyword_results() -> None:
    vector_store = FakeVectorStore()
    embedding_client = FakeEmbeddingClient()
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        config=RAGConfig(semantic_top_k=2, keyword_top_k=2, final_top_k=3),
    )

    chunks, query_embedding = retriever.retrieve("objectives")

    assert query_embedding == [0.1, 0.2]
    assert embedding_client.calls == 1
    assert [chunk["chunk_id"] for chunk in chunks] == ["a", "c", "b"]
    assert chunks[0]["hybrid_score"] == 1.2


def test_hybrid_retriever_reuses_existing_query_embedding() -> None:
    vector_store = FakeVectorStore()
    embedding_client = FakeEmbeddingClient()
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        config=RAGConfig(final_top_k=2),
    )

    _, query_embedding = retriever.retrieve("objectives", query_embedding=[0.4, 0.5])

    assert query_embedding == [0.4, 0.5]
    assert vector_store.semantic_embedding_seen == [0.4, 0.5]
    assert embedding_client.calls == 0


def test_hybrid_retriever_passes_metadata_filters_before_search() -> None:
    vector_store = FakeVectorStore()
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_client=FakeEmbeddingClient(),
        config=RAGConfig(final_top_k=2),
    )

    retriever.retrieve("objectives", filters={"source_type": "MASTER_DATA"})

    assert vector_store.filters_seen == {"source_type": "MASTER_DATA"}
