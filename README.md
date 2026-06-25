# TMF Classifier

Stage 1 is a small, modular BioClinicalBERT pipeline for classifying Trial Master File documents. It currently supports three classes:

- `protocol`
- `safety_report`
- `statistical_analysis_plan`

Long documents are split into text chunks because BERT has a maximum token limit. The data split happens at the document (`file_name`) level, so chunks from the same document never appear in both train and test sets. This prevents leakage.

## Workflow

1. Place raw PDF, DOCX, or TXT files in the three class folders under `data/`.
2. Run preprocessing to extract and clean text, create overlapping chunks, and save `artifacts/preprocessed_dataset.csv`, `artifacts/train.csv`, `artifacts/validation.csv`, and `artifacts/test.csv`.
3. Train on Colab/GPU when needed; the call in `main.py` is deliberately commented out.
4. Place the saved model and `label_encoder.pkl` inside `artifacts/` for local evaluation and prediction.

Training uses a document-level validation split for checkpoint selection; the test split is reserved for final evaluation. Evaluation reports chunk-level accuracy and macro F1, then groups chunk predictions by `file_name` and applies majority voting for document-level accuracy and macro F1. Metrics, run metadata, and both confusion matrices are stored in `artifacts/`.

## Local use

Install dependencies with `pip install -r requirements.txt`, then run:

```bash
python main.py
```

The local workflow is designed to use an already trained model in `artifacts/saved_bioclinicalbert_tmf_3class/`. Training is best run in Colab because fine-tuning BioClinicalBERT benefits from a GPU.

## Layout

```text
src/                 Modular preprocessing, training, evaluation, and prediction code
artifacts/           Preprocessed data, train/test splits, model, and evaluation outputs
data/<class>/         Raw files in protocol, safety_report, and statistical_analysis_plan folders
logs/                Application logs
```

## Stage 2: MLOps Foundation

Stage 2 adds optional MLflow experiment tracking through DagsHub, DVC data versioning guidance, structured metadata in `metadata/`, and `.env`-based secret management. Local execution remains safe when DagsHub credentials are not configured: the pipeline logs that MLflow is skipped and continues.

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env manually with your DagsHub values and access token.
python main.py
```

Never commit `.env` or a DagsHub token. `MLFLOW_TRACKING_PASSWORD` must contain a DagsHub access token, not an account password.

### Included in Stage 2

- MLflow and DagsHub experiment-tracking configuration
- DVC data-versioning setup instructions
- Structured dataset, model, training, evaluation, and version-history metadata
- `.env`-based secret management
- Lightweight configuration, metadata, and MLflow safety tests

## DVC + DagsHub Setup

Run these commands manually after creating a DagsHub repository:

```bash
dvc init

dvc add data/
dvc add artifacts/preprocessed_dataset.csv
dvc add artifacts/train.csv
dvc add artifacts/test.csv

git add data.dvc artifacts/preprocessed_dataset.csv.dvc artifacts/train.csv.dvc artifacts/test.csv.dvc .gitignore .dvc/config
git commit -m "Track datasets with DVC"

dvc remote add origin https://dagshub.com/<username>/<repo>.dvc

dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <dagshub_token>

dvc push
```

Git stores small `.dvc` metadata files; DagsHub stores the large data files. Never commit a DagsHub token. Track `artifacts/validation.csv` with DVC too because it is part of the model-selection split.

## Stage 3: API + Docker + CI/CD

Stage 3 adds a FastAPI serving layer, API tests, Docker packaging, GitHub Actions CI, and DockerHub image publishing.

Included in Stage 3:

- FastAPI app in `app.py`
- `/`, `/health`, `/model-info`, `/predict`, and `/predict-file` endpoints
- Pydantic request/response schemas
- Unit and integration tests with `pytest` and FastAPI `TestClient`
- Dockerfile and Docker Compose setup
- GitHub Actions workflow for tests and DockerHub image push

### Local API

```bash
uvicorn app:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Example prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This document describes study objectives and inclusion criteria."}'
```

Example file-upload prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict-file" \
  -F "file=@/path/to/study_protocol.pdf"
```

The file-upload endpoint supports `.pdf`, `.docx`, and `.txt` files. It extracts text, applies the same cleaning and chunking strategy used in preprocessing, predicts every chunk, and returns a document-level prediction using majority voting. In Swagger UI, open `/docs`, choose `POST /predict-file`, click **Choose File**, and then click **Execute**.

Document-level confidence combines three signals:

- `model_confidence`: average model confidence for chunks that voted for the winning class
- `vote_confidence`: winning-class votes divided by total chunks
- `margin_confidence`: gap between winning votes and second-best votes, divided by total chunks

The returned `confidence` is:

```text
model_confidence * vote_confidence * (1 + margin_confidence)
```

The score is capped at `1.0`. Close votes are penalized, so a document with chunk votes like `32` vs `30` receives a lower confidence than a document with stronger agreement like `40` vs `10`. The `decision_status` field is passive metadata for future workflows only; it does not trigger any agent behavior.

### Tests

```bash
pytest
```

The tests do not run training. Prediction tests are skipped automatically if saved model artifacts are unavailable.


### Docker

```bash
docker build -t tmf-classifier .
docker run -p 8000:8000 \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  -v "$(pwd)/metadata:/app/metadata:ro" \
  tmf-classifier
```

The Docker image uses `requirements-api.txt`, which contains only API/inference dependencies and CPU-only PyTorch. The full `requirements.txt` remains for local development, testing, training utilities, DVC, and MLflow. Large local folders such as `artifacts/` and `data/` are excluded from the image; mount `artifacts/` at runtime for local inference.

### Docker Compose

```bash
docker-compose up --build
```

### GitHub Actions CI/CD

The workflow in `.github/workflows/ci-cd.yml`:

- runs tests on pushes and pull requests to `main`
- builds a Docker image on pushes to `main`
- pushes Docker image tags to DockerHub:
  - `<dockerhub_username>/tmf-classifier:latest`
  - `<dockerhub_username>/tmf-classifier:<git_sha>`

GitHub Secrets required for DockerHub CD:

- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

DockerHub image format:

```text
pardhu156/tmf-classifier:latest
```

Do not commit DockerHub credentials, DagsHub tokens, `.env`, raw datasets, or large model files directly to Git.
