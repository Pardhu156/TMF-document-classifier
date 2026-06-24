from src.config import MetadataConfig, MLOpsConfig
from src.metadata_manager import update_version_history
from src.utils import load_json, save_json


def test_save_and_load_json(tmp_path) -> None:
    path = tmp_path / "nested" / "sample.json"

    save_json({"value": 1}, path)

    assert load_json(path) == {"value": 1}


def test_version_history_can_be_created_in_a_temporary_location(tmp_path) -> None:
    metadata_config = MetadataConfig(
        metadata_dir=tmp_path,
        dataset_metadata_path=tmp_path / "dataset_metadata.json",
        model_metadata_path=tmp_path / "model_metadata.json",
        training_run_metadata_path=tmp_path / "training_run_metadata.json",
        evaluation_metadata_path=tmp_path / "evaluation_metadata.json",
        version_history_path=tmp_path / "version_history.json",
        experiment_notes_path=tmp_path / "experiment_notes.md",
    )

    history = update_version_history(metadata_config, MLOpsConfig(), "test metadata history")

    assert len(history) == 1
    assert metadata_config.version_history_path.exists()
