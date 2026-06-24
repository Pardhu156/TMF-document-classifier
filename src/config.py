"""Central, pathlib-based configuration for the Stage 1 pipeline."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    test_data_path: Path = PROJECT_ROOT / "artifacts" / "test.csv"
    test_size: float = 0.2
    random_state: int = 42

    def __post_init__(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


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
