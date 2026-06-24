from src import mlflow_tracking


def test_setup_mlflow_is_safe_without_credentials(monkeypatch) -> None:
    for name in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_EXPERIMENT_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

    assert mlflow_tracking.setup_mlflow() is False
