"""Central, pathlib-based configuration for the Stage 1 pipeline."""

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Keeps local imports safe until requirements are installed.
    def load_dotenv(*args, **kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class DataIngestionConfig:
    raw_data_dir: Path = PROJECT_ROOT / "data"
    artifact_dir: Path = PROJECT_ROOT / "artifacts"
    preprocessed_data_path: Path = PROJECT_ROOT / "artifacts" / "preprocessed_dataset.csv"
    balanced_data_path: Path = PROJECT_ROOT / "artifacts" / "preprocessed_dataset_balanced.csv"
    # Set balance_classes=True only if one class has far more chunks than others.
    balance_classes: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 50
    train_data_path: Path = PROJECT_ROOT / "artifacts" / "train.csv"
    validation_data_path: Path = PROJECT_ROOT / "artifacts" / "validation.csv"
    test_data_path: Path = PROJECT_ROOT / "artifacts" / "test.csv"
    validation_size: float = 0.2
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.preprocessed_data_path,
            self.balanced_data_path,
            self.train_data_path,
            self.validation_data_path,
            self.test_data_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelTrainingConfig:
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    # Change num_epochs here when you want more/less training epochs.
    num_epochs: int = 3
    learning_rate: float = 2e-5
    train_batch_size: int = 8
    eval_batch_size: int = 8
    max_length: int = 512
    output_dir: Path = PROJECT_ROOT / "artifacts" / "training_results"
    save_model_dir: Path = PROJECT_ROOT / "artifacts" / "saved_bioclinicalbert_tmf_3class"
    label_encoder_path: Path = PROJECT_ROOT / "artifacts" / "label_encoder.pkl"

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_model_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoder_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationConfig:
    metrics_path: Path = PROJECT_ROOT / "artifacts" / "metrics.json"
    confusion_matrix_path: Path = PROJECT_ROOT / "artifacts" / "confusion_matrix.csv"
    run_metadata_path: Path = PROJECT_ROOT / "artifacts" / "run_metadata.json"

    def __post_init__(self) -> None:
        for path in (self.metrics_path, self.confusion_matrix_path, self.run_metadata_path):
            path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class PredictionConfig:
    model_dir: Path = PROJECT_ROOT / "artifacts" / "saved_bioclinicalbert_tmf_3class"
    label_encoder_path: Path = PROJECT_ROOT / "artifacts" / "label_encoder.pkl"
    max_length: int = 512

    def __post_init__(self) -> None:
        self.model_dir.parent.mkdir(parents=True, exist_ok=True)
        self.label_encoder_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class MLOpsConfig:
    dagshub_username: str | None = None
    dagshub_repo_name: str | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "tmf_classifier_experiments"
    dvc_remote_name: str = "origin"
    dvc_remote_url: str | None = None
    model_version: str = "v1.0.0"
    dataset_version: str = "v1.0.0"
    environment: str = "local"

    def __post_init__(self) -> None:
        self.dagshub_username = self.dagshub_username or os.getenv("DAGSHUB_USERNAME")
        self.dagshub_repo_name = self.dagshub_repo_name or os.getenv("DAGSHUB_REPO_NAME")
        self.mlflow_tracking_uri = self.mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", self.mlflow_experiment_name)
        self.dvc_remote_name = os.getenv("DVC_REMOTE_NAME", self.dvc_remote_name)
        self.dvc_remote_url = self.dvc_remote_url or os.getenv("DVC_REMOTE_URL")
        self.model_version = os.getenv("MODEL_VERSION", self.model_version)
        self.dataset_version = os.getenv("DATASET_VERSION", self.dataset_version)
        self.environment = os.getenv("ENVIRONMENT", self.environment)


@dataclass
class MetadataConfig:
    metadata_dir: Path = PROJECT_ROOT / "metadata"
    dataset_metadata_path: Path = PROJECT_ROOT / "metadata" / "dataset_metadata.json"
    model_metadata_path: Path = PROJECT_ROOT / "metadata" / "model_metadata.json"
    training_run_metadata_path: Path = PROJECT_ROOT / "metadata" / "training_run_metadata.json"
    evaluation_metadata_path: Path = PROJECT_ROOT / "metadata" / "evaluation_metadata.json"
    version_history_path: Path = PROJECT_ROOT / "metadata" / "version_history.json"
    experiment_notes_path: Path = PROJECT_ROOT / "metadata" / "experiment_notes.md"

    def __post_init__(self) -> None:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        for path in (
            self.dataset_metadata_path,
            self.model_metadata_path,
            self.training_run_metadata_path,
            self.evaluation_metadata_path,
            self.version_history_path,
            self.experiment_notes_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
