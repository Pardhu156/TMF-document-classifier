from src.config import (
    DataIngestionConfig,
    EvaluationConfig,
    MLOpsConfig,
    MetadataConfig,
    ModelTrainingConfig,
    PredictionConfig,
)


def test_config_paths_are_initialized() -> None:
    data_config = DataIngestionConfig()
    metadata_config = MetadataConfig()
    training_config = ModelTrainingConfig()

    assert data_config.artifact_dir.exists()
    assert metadata_config.metadata_dir.exists()
    assert training_config.output_dir.exists()


def test_mlops_config_has_safe_defaults() -> None:
    config = MLOpsConfig()

    assert config.model_version
    assert config.dataset_version
    assert config.environment


def test_api_related_configs_initialize() -> None:
    prediction_config = PredictionConfig()
    evaluation_config = EvaluationConfig()

    assert prediction_config.model_dir.parent.exists()
    assert evaluation_config.metrics_path.parent.exists()
