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
docker run -p 8000:8000 tmf-classifier
```

The Docker image uses `requirements-api.txt`, which contains only API/inference dependencies and CPU-only PyTorch. The full `requirements.txt` remains for local development, testing, training utilities, DVC, and MLflow.

For easier demos, the Docker image includes only the inference artifacts required to run predictions:

- `artifacts/saved_bioclinicalbert_tmf_3class/`
- `artifacts/label_encoder.pkl`
- `metadata/`

Raw `data/`, generated CSVs, logs, `.env`, and DVC cache files are excluded from the image.

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

## Stage 4: Cloud Persistence + Conditional Retraining Foundation

Stage 4 adds cloud persistence and retraining orchestration without changing the existing local inference workflow.

Included in Stage 4:

- AWS S3 storage manager
- PostgreSQL SQLAlchemy models and repository layer
- idempotent document upload/ingestion pipeline
- duplicate detection using SHA256 file hashes
- chunk hashing using SHA256 text hashes
- persisted document, chunk, prediction, chunk prediction, model version, and audit log records
- conditional retraining coordinator
- `/retrain` admin/manual endpoint
- mock-based tests that do not hit real AWS or PostgreSQL

### Required external services

Create an AWS S3 bucket for cloud artifacts. Recommended folders/prefixes:

```text
raw_documents/
processed_documents/
model_artifacts/
reports/
```

Create a PostgreSQL database:

```text
tmf_classifier
```

The application creates SQLAlchemy tables automatically when the database connection is configured.

### Stage 4 environment variables

Add these to `.env`:

```env
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1
AWS_S3_BUCKET_NAME=tmf-classifier-stage4-bucket

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=tmf_classifier
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password

ALLOW_DUPLICATE_DOCUMENTS=False
RETRAIN_MIN_NEW_DOCUMENTS=1
RETRAIN_ONLY_VERIFIED_DATA=True
```

Never commit `.env`.

### Manual cloud bootstrap uploads

Use this only for assets that already exist locally and are not uploaded through `/predict-file`:

- `data/` training source files
- `artifacts/saved_bioclinicalbert_tmf_3class/`
- `artifacts/label_encoder.pkl`
- `metadata/`
- evaluation reports from `artifacts/`

Run:

```bash
python scripts/bootstrap_cloud_uploads.py
```

By default, existing S3 keys are skipped. To overwrite:

```bash
python scripts/bootstrap_cloud_uploads.py --overwrite
```

This uploads to:

```text
s3://tmf-classifier-stage4-bucket/raw_training_data/data/
s3://tmf-classifier-stage4-bucket/model_artifacts/model_v1/
s3://tmf-classifier-stage4-bucket/reports/
```

This bootstrap pipeline does not insert PostgreSQL rows. PostgreSQL is populated by `/predict-file`, manual verification, `/retrain`, and the application repository layer.

### Cloud upload workflow

```text
POST /predict-file
→ calculate file SHA256
→ check PostgreSQL for duplicate file_hash
→ upload raw file to S3 raw_documents/
→ extract and clean text
→ upload extracted text to S3 processed_documents/
→ chunk document
→ hash chunks
→ run prediction
→ save document/chunks/prediction/chunk predictions/audit log to PostgreSQL
→ return API prediction response
```

If AWS/PostgreSQL are not configured, `/predict-file` still performs local extraction and inference for backward compatibility, but persistence is disabled.

### Duplicate document workflow

When `ALLOW_DUPLICATE_DOCUMENTS=False`:

```text
same file hash found
→ skip S3 upload
→ skip extraction
→ skip prediction
→ return existing document metadata and latest prediction
→ create audit log
```

### Conditional retraining workflow

Manual endpoint:

```text
POST /retrain
```

If no verified unused data exists:

```json
{
  "status": "skipped",
  "message": "No new verified data found. Retraining skipped."
}
```

If verified unused data exists:

```text
fetch verified documents
→ prepare next model version record, e.g. model_v2
→ create audit log
→ return retraining started metadata
```

### Manual admin review workflow

For now, the project uses a simple admin/manual review endpoint. In production, this should be replaced with RBAC and authenticated reviewer permissions.

After `/predict-file` stores a document as `predicted_unverified`, an admin can verify the trusted label:

```text
POST /documents/{doc_id}/verify
```

Example request:

```json
{
  "verified_label": "protocol",
  "reviewer": "admin",
  "notes": "Reviewed document title and content."
}
```

This updates:

```text
verified_label = protocol
document_status = verified
used_for_training = False
```

Then `/retrain` can use that `verified_label`. Retraining never uses the model's own `predicted_label` as training truth.

The Stage 4 retraining pipeline intentionally does not overwrite the existing model. Future GPU fine-tuning should continue from the active checkpoint and save a new version under:

```text
model_artifacts/model_vX/
```

Only after the new model passes evaluation should it be marked active.
