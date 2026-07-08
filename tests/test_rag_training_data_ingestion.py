from pathlib import Path

from src.config import DataIngestionConfig, RAGConfig
from src.rag.training_data_ingestion import TrainingDataIngestionPipeline


class FakeVectorStore:
    def __init__(self) -> None:
        self.hashes_seen: list[str] = []

    def get_document_by_file_hash(self, file_hash: str, source_type: str | None = None):
        self.hashes_seen.append(file_hash)
        return None


class FakeIndexer:
    def __init__(self) -> None:
        self.indexed: list[dict] = []

    def index_document(self, **kwargs):
        self.indexed.append(kwargs)
        return len(kwargs["chunk_texts"])


def test_training_data_ingestion_indexes_class_folder_metadata(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    class_dir = data_dir / "protocol"
    class_dir.mkdir(parents=True)
    (class_dir / "prot.txt").write_text("This protocol describes study objectives and inclusion criteria. " * 20)

    fake_indexer = FakeIndexer()
    pipeline = TrainingDataIngestionPipeline(
        rag_config=RAGConfig(),
        data_config=DataIngestionConfig(raw_data_dir=data_dir, chunk_size=20, chunk_overlap=5),
        vector_store=FakeVectorStore(),
        indexer=fake_indexer,
    )
    monkeypatch.setattr(TrainingDataIngestionPipeline, "is_configured", classmethod(lambda cls: True))

    result = pipeline.run(data_dir=data_dir)

    assert result["indexed_documents"] == 1
    assert result["source_type"] == "TRAINING_DATA"
    assert result["classes_seen"] == ["protocol"]
    assert fake_indexer.indexed[0]["predicted_class"] == "protocol"
    assert fake_indexer.indexed[0]["source_type"] == "TRAINING_DATA"
    assert fake_indexer.indexed[0]["verification_status"] == "verified"
    assert fake_indexer.indexed[0]["access_level"] == "User"
    assert fake_indexer.indexed[0]["owner_id"] == "training_data_ingestion"
    assert fake_indexer.indexed[0]["file_name"] == "protocol/prot.txt"
