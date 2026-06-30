from scripts.seed_rag_database import _count_supported_documents


def test_count_supported_documents_detects_dvc_restored_files(tmp_path) -> None:
    (tmp_path / "protocol").mkdir()
    (tmp_path / "protocol" / "sample.pdf").write_text("pdf placeholder", encoding="utf-8")
    (tmp_path / "notes.md").write_text("unsupported", encoding="utf-8")

    assert _count_supported_documents(tmp_path) == 1


def test_count_supported_documents_returns_zero_for_empty_dvc_folder(tmp_path) -> None:
    assert _count_supported_documents(tmp_path) == 0
