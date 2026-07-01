from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.db_connection import create_tables
from src.database.repository import TMFRepository
from src.auth import hash_password


def _repo() -> TMFRepository:
    engine = create_engine("sqlite:///:memory:")
    create_tables(engine)
    return TMFRepository(sessionmaker(bind=engine, autoflush=False, autocommit=False))


def test_repository_saves_document_chunks_and_prediction() -> None:
    repo = _repo()

    document = repo.save_document(
        {
            "filename": "doc.txt",
            "file_hash": "abc123",
            "document_status": "predicted_unverified",
            "used_for_training": False,
        }
    )
    chunks = repo.save_chunks(
        document["doc_id"],
        [{"chunk_index": 0, "chunk_hash": "chunk123", "chunk_word_count": 5}],
    )
    prediction = repo.save_prediction(
        document["doc_id"],
        {
            "predicted_label": "protocol",
            "confidence": 0.9,
            "decision_status": "auto_classify",
            "requires_review": False,
            "chunk_predictions": {"protocol": 1},
            "num_chunks": 1,
        },
    )

    assert repo.get_document_by_hash("abc123")["doc_id"] == document["doc_id"]
    assert chunks[0]["doc_id"] == document["doc_id"]
    assert prediction["predicted_label"] == "protocol"


def test_repository_gets_new_verified_documents_and_marks_used() -> None:
    repo = _repo()
    document = repo.save_document(
        {
            "filename": "verified.txt",
            "file_hash": "hash_verified",
            "document_status": "verified",
            "verified_label": "protocol",
            "used_for_training": False,
        }
    )

    verified = repo.get_new_verified_documents()
    updated_count = repo.mark_documents_used_for_training([document["doc_id"]], "dataset_v2")

    assert len(verified) == 1
    assert updated_count == 1


def test_repository_verifies_document_for_future_retraining() -> None:
    repo = _repo()
    document = repo.save_document(
        {
            "filename": "review_me.txt",
            "file_hash": "hash_review_me",
            "document_status": "predicted_unverified",
            "used_for_training": False,
        }
    )

    verified = repo.verify_document(document["doc_id"], "protocol")

    assert verified["verified_label"] == "protocol"
    assert verified["document_status"] == "verified"
    assert verified["used_for_training"] is False


def test_repository_upserts_users_idempotently() -> None:
    repo = _repo()
    user_data = {
        "name": "Demo User",
        "email": "USER@test.com",
        "hashed_password": hash_password("user123"),
        "role": "User",
        "is_active": True,
    }

    created = repo.upsert_user(user_data)
    updated = repo.upsert_user({**user_data, "name": "Demo User Updated"})

    assert created["id"] == updated["id"]
    assert updated["email"] == "user@test.com"
    assert updated["name"] == "Demo User Updated"
