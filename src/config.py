"""Central, pathlib-based configuration for the Stage 1 pipeline."""

from dataclasses import dataclass
import os
from pathlib import Path

from sqlalchemy.engine import URL

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


@dataclass
class CloudConfig:
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str = "us-east-1"
    aws_s3_bucket_name: str | None = None
    allow_duplicate_documents: bool = False

    raw_documents_prefix: str = "raw_documents/"
    processed_documents_prefix: str = "processed_documents/"
    model_artifacts_prefix: str = "model_artifacts/"
    reports_prefix: str = "reports/"
    tmf_prefix: str = "agentic_tmf_workspace/tmf/"
    pending_review_prefix: str = "agentic_tmf_workspace/pending_review/"
    pending_training_prefix: str = "agentic_tmf_workspace/pending_training/"
    approved_training_prefix: str = "agentic_tmf_workspace/approved_training/"
    rejected_training_prefix: str = "agentic_tmf_workspace/rejected_training/"
    metadata_prefix: str = "agentic_tmf_workspace/metadata/"
    local_cloud_root: Path = PROJECT_ROOT / "agentic_tmf_workspace"

    def __post_init__(self) -> None:
        self.aws_access_key_id = self.aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = self.aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", self.aws_region)
        self.aws_s3_bucket_name = self.aws_s3_bucket_name or os.getenv("AWS_S3_BUCKET_NAME")
        self.allow_duplicate_documents = os.getenv(
            "ALLOW_DUPLICATE_DOCUMENTS",
            str(self.allow_duplicate_documents),
        ).strip().lower() in {"1", "true", "yes", "y"}
        self.local_cloud_root = Path(os.getenv("LOCAL_CLOUD_ROOT", str(self.local_cloud_root)))

    @property
    def is_configured(self) -> bool:
        return bool(self.aws_s3_bucket_name)


@dataclass
class AgenticFilingConfig:
    """Configuration for Stage 6 confidence-based filing decisions."""

    auto_approval_threshold: float = 0.90
    min_confidence_gap: float = 0.10
    manual_review_queue_name: str = "manual_review:pending"

    def __post_init__(self) -> None:
        self.auto_approval_threshold = float(
            os.getenv("AUTO_APPROVAL_THRESHOLD", str(self.auto_approval_threshold))
        )
        self.min_confidence_gap = float(os.getenv("MIN_CONFIDENCE_GAP", str(self.min_confidence_gap)))
        self.manual_review_queue_name = os.getenv("MANUAL_REVIEW_QUEUE_NAME", self.manual_review_queue_name)


@dataclass
class DatabaseConfig:
    postgres_host: str | None = None
    postgres_port: int = 5432
    postgres_db: str = "tmf_classifier"
    postgres_user: str | None = None
    postgres_password: str | None = None
    database_url: str | None = None

    def __post_init__(self) -> None:
        self.postgres_host = self.postgres_host or os.getenv("POSTGRES_HOST")
        self.postgres_port = int(os.getenv("POSTGRES_PORT", str(self.postgres_port)))
        self.postgres_db = os.getenv("POSTGRES_DB", self.postgres_db)
        self.postgres_user = self.postgres_user or os.getenv("POSTGRES_USER")
        self.postgres_password = self.postgres_password or os.getenv("POSTGRES_PASSWORD")
        if not self.database_url and not any((self.postgres_host, self.postgres_user, self.postgres_password)):
            self.database_url = os.getenv("DATABASE_URL")

    @property
    def sqlalchemy_url(self) -> str | None:
        if self.database_url:
            return self.database_url
        if not all((self.postgres_host, self.postgres_user, self.postgres_password, self.postgres_db)):
            return None
        return (
            URL.create(
                "postgresql+psycopg2",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
            )
            .render_as_string(hide_password=False)
        )

    @property
    def is_configured(self) -> bool:
        return self.sqlalchemy_url is not None


@dataclass
class AuthConfig:
    """JWT and password-hashing configuration for API authentication."""

    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    def __post_init__(self) -> None:
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", self.jwt_secret_key)
        self.jwt_algorithm = os.getenv("JWT_ALGORITHM", self.jwt_algorithm)
        self.access_token_expire_minutes = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(self.access_token_expire_minutes))
        )


@dataclass
class RetrainingConfig:
    retrain_min_new_documents: int = 1
    retrain_only_verified_data: bool = True
    evaluation_threshold_macro_f1: float = 0.0

    def __post_init__(self) -> None:
        self.retrain_min_new_documents = int(
            os.getenv("RETRAIN_MIN_NEW_DOCUMENTS", str(self.retrain_min_new_documents))
        )
        self.retrain_only_verified_data = os.getenv(
            "RETRAIN_ONLY_VERIFIED_DATA",
            str(self.retrain_only_verified_data),
        ).strip().lower() in {"1", "true", "yes", "y"}
        self.evaluation_threshold_macro_f1 = float(
            os.getenv("RETRAIN_EVALUATION_THRESHOLD_MACRO_F1", str(self.evaluation_threshold_macro_f1))
        )


@dataclass
class RAGConfig:
    master_data_dir: Path = PROJECT_ROOT / "MASTER_DATA"
    auto_index_master_data: bool = False
    embedding_provider: str = "local"
    local_embedding_model: str = "NeuML/pubmedbert-base-embeddings"
    local_model_dir: Path = Path("/models/pubmedbert-base-embeddings")
    local_embedding_device: str = "cpu"
    local_embedding_batch_size: int = 8
    gemini_api_key: str | None = None
    gemini_embedding_model: str = "models/gemini-embedding-001"
    gemini_generation_model: str = "models/gemini-flash-lite-latest"
    embedding_dimension: int = 768
    semantic_top_k: int = 10
    keyword_top_k: int = 10
    final_top_k: int = 5
    reranker_enabled: bool = False
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.85
    redis_url: str | None = None
    semantic_cache_ttl_seconds: int = 86400
    model_backup_s3_bucket: str | None = None
    model_backup_s3_prefix: str = "rag-artifacts/embedding-models/pubmedbert/"
    rag_artifacts_s3_prefix: str = "rag-artifacts/"
    rag_ingestion_reports_s3_prefix: str = "rag-artifacts/ingestion-reports/"
    rag_evaluation_s3_prefix: str = "rag-artifacts/rag-evaluation/"
    rag_failed_ingestions_s3_prefix: str = "rag-artifacts/failed-ingestions/"

    def __post_init__(self) -> None:
        self.master_data_dir = Path(os.getenv("MASTER_DATA_DIR", str(self.master_data_dir)))
        self.auto_index_master_data = os.getenv(
            "AUTO_INDEX_MASTER_DATA",
            str(self.auto_index_master_data),
        ).strip().lower() in {"1", "true", "yes", "y"}
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", self.embedding_provider).strip().lower()
        self.local_embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", self.local_embedding_model)
        self.local_model_dir = Path(os.getenv("LOCAL_MODEL_DIR", str(self.local_model_dir)))
        self.local_embedding_device = os.getenv("LOCAL_EMBEDDING_DEVICE", self.local_embedding_device)
        self.local_embedding_batch_size = int(
            os.getenv("LOCAL_EMBEDDING_BATCH_SIZE", str(self.local_embedding_batch_size))
        )
        self.gemini_api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", self.gemini_embedding_model)
        self.gemini_generation_model = os.getenv("GEMINI_GENERATION_MODEL", self.gemini_generation_model)
        self.embedding_dimension = int(os.getenv("RAG_EMBEDDING_DIMENSION", str(self.embedding_dimension)))
        self.semantic_top_k = int(os.getenv("RAG_SEMANTIC_TOP_K", str(self.semantic_top_k)))
        self.keyword_top_k = int(os.getenv("RAG_KEYWORD_TOP_K", str(self.keyword_top_k)))
        self.final_top_k = int(os.getenv("RAG_FINAL_TOP_K", str(self.final_top_k)))
        self.reranker_enabled = os.getenv("RAG_RERANKER_ENABLED", str(self.reranker_enabled)).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        self.semantic_cache_enabled = os.getenv(
            "SEMANTIC_CACHE_ENABLED",
            str(self.semantic_cache_enabled),
        ).strip().lower() in {"1", "true", "yes", "y"}
        self.semantic_cache_threshold = float(
            os.getenv("SEMANTIC_CACHE_THRESHOLD", str(self.semantic_cache_threshold))
        )
        self.redis_url = self.redis_url or os.getenv("REDIS_URL")
        self.semantic_cache_ttl_seconds = int(
            os.getenv("SEMANTIC_CACHE_TTL_SECONDS", str(self.semantic_cache_ttl_seconds))
        )
        self.model_backup_s3_bucket = (
            self.model_backup_s3_bucket
            or os.getenv("MODEL_BACKUP_S3_BUCKET")
            or os.getenv("AWS_S3_BUCKET_NAME")
        )
        self.model_backup_s3_prefix = os.getenv("MODEL_BACKUP_S3_PREFIX", self.model_backup_s3_prefix)
        self.rag_artifacts_s3_prefix = os.getenv("RAG_ARTIFACTS_S3_PREFIX", self.rag_artifacts_s3_prefix)
        self.rag_ingestion_reports_s3_prefix = os.getenv(
            "RAG_INGESTION_REPORTS_S3_PREFIX",
            self.rag_ingestion_reports_s3_prefix,
        )
        self.rag_evaluation_s3_prefix = os.getenv("RAG_EVALUATION_S3_PREFIX", self.rag_evaluation_s3_prefix)
        self.rag_failed_ingestions_s3_prefix = os.getenv(
            "RAG_FAILED_INGESTIONS_S3_PREFIX",
            self.rag_failed_ingestions_s3_prefix,
        )

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def redis_configured(self) -> bool:
        return bool(self.redis_url)

    @property
    def uses_local_embeddings(self) -> bool:
        return self.embedding_provider == "local"
