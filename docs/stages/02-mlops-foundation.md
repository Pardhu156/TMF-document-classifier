# Stage 2: MLOps Foundation

## Tech Stack

- DVC
- MLflow
- DagsHub
- JSON metadata files
- Python dotenv
- GitHub-safe secret handling

## Key Steps

1. Added `.env`-based configuration.
2. Added MLflow/DagsHub tracking hooks.
3. Added DVC dataset and artifact versioning guidance.
4. Created structured metadata files for dataset, model, training, evaluation, and version history.
5. Added safe behavior when MLflow credentials are not configured.
6. Added tests for configuration, metadata, and MLflow safety.

## What Was Implemented

Stage 2 made the ML workflow explainable and reproducible. Instead of only having a saved model artifact, the project now stores metadata about:

- dataset version
- model version
- class names
- split strategy
- chunking strategy
- training hyperparameters
- evaluation metrics
- package versions
- version history

## Important Metadata

| Metadata | Value |
| --- | --- |
| Dataset version | `v1.0.0` |
| Model version | `v1.0.0` |
| Model name | `emilyalsentzer/Bio_ClinicalBERT` |
| Training status | `legacy_checkpoint_imported` |
| Epochs | 3 |
| Learning rate | `2e-5` |
| Batch size | 8 |

## Files And Artifacts

| Path | Purpose |
| --- | --- |
| `src/mlflow_tracking.py` | MLflow setup and safe logging helpers |
| `src/metadata_manager.py` | JSON metadata creation and version history |
| `metadata/dataset_metadata.json` | Dataset summary |
| `metadata/model_metadata.json` | Model summary |
| `metadata/training_run_metadata.json` | Training run summary |
| `metadata/evaluation_metadata.json` | Evaluation summary |
| `metadata/version_history.json` | Dataset/model version history |
| `.env.example` | Safe environment variable template |

## Limitations

- MLflow model registry promotion is not fully automated.
- DVC remote setup must be configured manually.
- Production model governance is future work.
