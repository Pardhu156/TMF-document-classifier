from pathlib import Path

from src.config import CloudConfig
from src.pipeline.cloud_bootstrap_pipeline import CloudBootstrapPipeline


class FakeS3Manager:
    def __init__(self) -> None:
        self.uploaded_files = []
        self.uploaded_directories = []
        self.existing_keys = set()
        self.config = CloudConfig(aws_s3_bucket_name="tmf-classifier-stage4-bucket")

    def upload_directory(self, local_dir, prefix, skip_existing=True):
        self.uploaded_directories.append((str(local_dir), prefix, skip_existing))
        return [f"s3://tmf-classifier-stage4-bucket/{prefix}/file.txt"]

    def file_exists(self, key):
        return key in self.existing_keys

    def generate_s3_uri(self, key):
        return f"s3://tmf-classifier-stage4-bucket/{key}"

    def upload_file(self, local_path, key, content_type=None):
        self.uploaded_files.append((str(local_path), key, content_type))
        return self.generate_s3_uri(key)


def test_cloud_bootstrap_uploads_expected_asset_groups(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "model"
    metadata_dir = tmp_path / "metadata"
    artifact_dir = tmp_path / "artifacts"
    for directory in (data_dir, model_dir, metadata_dir, artifact_dir):
        directory.mkdir()
    (data_dir / "doc.txt").write_text("raw", encoding="utf-8")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    label_encoder = artifact_dir / "label_encoder.pkl"
    label_encoder.write_bytes(b"encoder")
    metrics = artifact_dir / "metrics.json"
    metrics.write_text("{}", encoding="utf-8")
    confusion = artifact_dir / "confusion_matrix.csv"
    confusion.write_text("a,b", encoding="utf-8")
    run_metadata = artifact_dir / "run_metadata.json"
    run_metadata.write_text("{}", encoding="utf-8")

    fake_s3 = FakeS3Manager()
    pipeline = CloudBootstrapPipeline(s3_manager=fake_s3)
    pipeline.data_config.raw_data_dir = data_dir
    pipeline.training_config.save_model_dir = model_dir
    pipeline.training_config.label_encoder_path = label_encoder
    pipeline.metadata_config.metadata_dir = metadata_dir
    pipeline.evaluation_config.metrics_path = metrics
    pipeline.evaluation_config.confusion_matrix_path = confusion
    pipeline.evaluation_config.run_metadata_path = run_metadata

    manifest = pipeline.run()

    assert manifest["bucket"] == "tmf-classifier-stage4-bucket"
    assert set(manifest["uploads"]) == {"raw_training_data", "model_artifacts", "reports"}
    assert any(prefix == "raw_training_data/data" for _, prefix, _ in fake_s3.uploaded_directories)
    assert any(prefix.startswith("model_artifacts/model_v1") for _, prefix, _ in fake_s3.uploaded_directories)
    assert any(key == "model_artifacts/model_v1/label_encoder.pkl" for _, key, _ in fake_s3.uploaded_files)


def test_cloud_bootstrap_skips_existing_single_file(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    label_encoder = artifact_dir / "label_encoder.pkl"
    label_encoder.write_bytes(b"encoder")
    fake_s3 = FakeS3Manager()
    fake_s3.existing_keys.add("model_artifacts/model_v1/label_encoder.pkl")
    pipeline = CloudBootstrapPipeline(s3_manager=fake_s3)
    pipeline.data_config.raw_data_dir = tmp_path / "missing_data"
    pipeline.training_config.save_model_dir = tmp_path / "missing_model"
    pipeline.training_config.label_encoder_path = label_encoder
    pipeline.metadata_config.metadata_dir = tmp_path / "missing_metadata"
    pipeline.evaluation_config.metrics_path = tmp_path / "missing_metrics.json"
    pipeline.evaluation_config.confusion_matrix_path = tmp_path / "missing_confusion.csv"
    pipeline.evaluation_config.run_metadata_path = tmp_path / "missing_run_metadata.json"

    manifest = pipeline.run(skip_existing=True)

    assert manifest["uploads"]["model_artifacts"] == [
        "s3://tmf-classifier-stage4-bucket/model_artifacts/model_v1/label_encoder.pkl"
    ]
    assert fake_s3.uploaded_files == []
