from src.config import (
    DataIngestionConfig,
    DatabaseConfig,
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


def test_database_config_escapes_special_password_characters() -> None:
    config = DatabaseConfig(
        postgres_host="localhost",
        postgres_user="user",
        postgres_password="p@ss/word+with:symbols",
        postgres_db="tmf_classifier",
    )

    assert "p%40ss%2Fword+with%3Asymbols" in config.sqlalchemy_url
