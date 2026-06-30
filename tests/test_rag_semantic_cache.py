import json

from src.config import RAGConfig
from src.rag.semantic_cache import RedisSemanticCache, cosine_similarity, document_scope


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: str) -> None:
        self.store[key] = value

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value

    def scan_iter(self, pattern: str):
        prefix = pattern.replace("*", "")
        for key in list(self.store):
            if key.startswith(prefix):
                yield key


def test_document_scope_is_generic() -> None:
    assert document_scope() == "all_documents"
    assert document_scope(document_id="42") == "document:42"
    assert document_scope(predicted_class="protocol") == "class:protocol"


def test_cosine_similarity_handles_close_vectors() -> None:
    score = cosine_similarity([1.0, 0.0], [0.9, 0.1])

    assert score > 0.99


def test_semantic_cache_returns_only_matching_scope() -> None:
    fake_redis = FakeRedis()
    config = RAGConfig(
        redis_url="redis://localhost:6379/0",
        semantic_cache_threshold=0.85,
        semantic_cache_ttl_seconds=60,
    )
    cache = RedisSemanticCache(config=config, client=fake_redis)

    cache.set(
        question="What are the objectives?",
        question_embedding=[1.0, 0.0],
        answer="The objectives are listed in the protocol.",
        scope="document:1",
        sources=[{"chunk_id": "1:0"}],
        document_id="1",
    )
    fake_redis.setex(
        "rag:cache:other-scope",
        60,
        json.dumps(
            {
                "question": "What are the objectives?",
                "question_embedding": [1.0, 0.0],
                "answer": "Wrong scope",
                "document_scope": "document:2",
                "source_chunks": [],
            }
        ),
    )

    result = cache.search_semantic([0.95, 0.05], "document:1")

    assert result is not None
    assert result["answer"] == "The objectives are listed in the protocol."
    assert result["semantic_cache_score"] >= 0.85


def test_document_status_uses_expected_redis_key() -> None:
    fake_redis = FakeRedis()
    cache = RedisSemanticCache(
        config=RAGConfig(redis_url="redis://localhost:6379/0"),
        client=fake_redis,
    )

    cache.set_document_status("abc", "indexed")

    assert fake_redis.store["doc:status:abc"] == "indexed"
    assert cache.get_document_status("abc") == "indexed"
