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
TOKEN=$(curl -s -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@test.com", "password": "user123"}' | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"text": "This document describes study objectives and inclusion criteria."}'
```

Example file-upload prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict-file" \
  -H "Authorization: Bearer ${TOKEN}" \
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


### Docker Compose RAG stack

For the full local RAG system, use Docker Compose instead of running only the API container.

Expected setup:

```bash
git clone <repo-url>
cd <repo>
cp .env.example .env
# edit .env and add required keys, especially GEMINI_API_KEY
dvc pull
docker compose up --build
```

The Compose stack starts:

- `rag-app`: FastAPI application
- `postgres`: PostgreSQL with pgvector for RAG documents/chunks
- `redis`: exact and semantic cache
- `db-seeder`: one-time idempotent seed job

The seeder checks PostgreSQL before indexing:

- it creates/updates development demo users idempotently
- if `rag_chunks` already has rows, ingestion is skipped
- if `rag_chunks` is empty, it indexes DVC-restored `MASTER_DATA/`

Data is not committed to GitHub and is not baked into the public Docker image. Run `dvc pull` first so these local folders exist:

```text
MASTER_DATA/
data/
artifacts/
```

Open the API after startup:

```text
http://127.0.0.1:8000/docs
```

Shutdown:

```bash
docker compose down
```

Shutdown and remove local PostgreSQL volume:

```bash
docker compose down -v
```

Only use `down -v` when you intentionally want to delete the local indexed pgvector data.

Troubleshooting:

- If `db-seeder` says `MASTER_DATA directory was not found`, run `dvc pull`.
- If PostgreSQL port `5434` is already used, change `POSTGRES_HOST_PORT` in `.env`.
- If Redis port `6379` is already used, change `REDIS_HOST_PORT` in `.env`.
- If `/rag/ask` returns a Gemini quota error, use a valid `GEMINI_API_KEY` or wait for quota reset.
- If RAG returns no chunks, check seeder logs:

```bash
docker compose logs db-seeder
```

Rebuild from scratch:

```bash
docker compose down -v
docker compose up --build
```

### GitHub Actions CI/CD

The workflow in `.github/workflows/ci-cd.yml`:

- runs tests on pushes and pull requests to `main`
- builds a Docker image on pushes to `main`
- pushes Docker image tags to DockerHub:
  - `<dockerhub_username>/tmf-classifier:latest`
  - `<dockerhub_username>/tmf-classifier:<git_sha>`

GitHub Secrets required for DockerHub CD:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

DockerHub image format:

```text
<dockerhub_username>/tmf-classifier:latest
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
AWS_S3_BUCKET_NAME=your_s3_bucket_name

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
s3://<your_s3_bucket_name>/raw_training_data/data/
s3://<your_s3_bucket_name>/model_artifacts/model_v1/
s3://<your_s3_bucket_name>/reports/
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

The manual admin review endpoint is protected by Stage 7.1 RBAC and requires an Admin token.

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

## Stage 5: RAG + Redis Semantic Cache

Stage 5 adds a document-level question-answering layer over uploaded/indexed TMF documents. It does not replace the classifier. Classification still predicts TMF class labels; RAG uses document chunks and embeddings so users can ask questions about the uploaded document corpus.

Included in Stage 5:

- local PubMedBERT embeddings for document chunks and user questions
- PostgreSQL `pgvector` storage for chunk embeddings
- PostgreSQL full-text search for exact clinical/TMF terms
- hybrid retrieval using semantic search + keyword search
- reranker interface with hybrid-score fallback
- Gemini grounded answer generation
- Redis semantic cache scoped by document, class, or all documents
- RAG metrics and MLflow logging hooks
- Swagger endpoints for RAG question answering
- manual `MASTER_DATA` ingestion for the trusted knowledge base

### Stage 5 environment variables

Add these to `.env`:

```env
MASTER_DATA_DIR=MASTER_DATA
AUTO_INDEX_MASTER_DATA=False

GEMINI_API_KEY=your_gemini_api_key
GEMINI_GENERATION_MODEL=models/gemini-flash-lite-latest

EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=NeuML/pubmedbert-base-embeddings
LOCAL_MODEL_DIR=/models/pubmedbert-base-embeddings
LOCAL_EMBEDDING_DEVICE=cpu
LOCAL_EMBEDDING_BATCH_SIZE=8
RAG_EMBEDDING_DIMENSION=768
RAG_SEMANTIC_TOP_K=10
RAG_KEYWORD_TOP_K=10
RAG_FINAL_TOP_K=5
RAG_RERANKER_ENABLED=False

SEMANTIC_CACHE_ENABLED=True
SEMANTIC_CACHE_THRESHOLD=0.85
SEMANTIC_CACHE_TTL_SECONDS=86400
REDIS_URL=redis://localhost:6379/0

MODEL_BACKUP_S3_PREFIX=rag-artifacts/embedding-models/pubmedbert/
RAG_ARTIFACTS_S3_PREFIX=rag-artifacts/
RAG_INGESTION_REPORTS_S3_PREFIX=rag-artifacts/ingestion-reports/
RAG_EVALUATION_S3_PREFIX=rag-artifacts/rag-evaluation/
RAG_FAILED_INGESTIONS_S3_PREFIX=rag-artifacts/failed-ingestions/
```

PostgreSQL must have the `pgvector` extension available. The app runs:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Trusted master knowledge base

Create a local folder for trusted RAG source documents:

```text
MASTER_DATA/
```

Put trusted `.pdf`, `.docx`, or `.txt` documents there. These are indexed with:

```text
source_type = MASTER_DATA
verification_status = verified
```

The app does not index this folder automatically by default:

```env
AUTO_INDEX_MASTER_DATA=False
```

Run indexing manually through either Swagger/API:

```bash
curl -X POST "http://127.0.0.1:8000/rag/index-master-data" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}"
```

or CLI:

```bash
python scripts/index_master_data.py
```

Optional path override:

```bash
python scripts/index_master_data.py --master-data-dir /path/to/MASTER_DATA
```

The pipeline computes a file hash and skips files already indexed into pgvector, so rerunning it is safe.

### Optional: index the labeled `data/` folder into pgvector

If you want your existing labeled training folders to be searchable by RAG, run:

```bash
python scripts/index_training_data.py
```

Optional path override:

```bash
python scripts/index_training_data.py --data-dir /path/to/data
```

Expected structure:

```text
data/
├── protocol/
├── safety_report/
└── statistical_analysis_plan/
```

These documents are indexed with:

```text
source_type = TRAINING_DATA
verification_status = verified
predicted_class = <class folder name>
```

This is separate from `MASTER_DATA`. Default RAG questions still search only `MASTER_DATA`. To include training data in retrieval, use `scope="all"` or a class-scoped request.

### Embedding model change and re-indexing

Current RAG embeddings use:

```text
NeuML/pubmedbert-base-embeddings
```

This local model produces `768`-dimension vectors. Local indexing defaults to CPU with a small batch size to avoid Apple MPS/GPU out-of-memory errors. Do not mix old Gemini `3072`-dimension vectors with PubMedBERT vectors. After changing embedding models, reset the RAG vector tables and re-index:

```bash
python scripts/reset_rag_vector_store.py
python scripts/index_master_data.py
```

Optional training data indexing:

```bash
python scripts/index_training_data.py
```

### RAG indexing flow

When `/predict-file` receives a new document and PostgreSQL + local embeddings are configured:

```text
upload document
→ extract text
→ clean and chunk text
→ classify document
→ embed chunks using local PubMedBERT
→ store chunks + vectors in PostgreSQL pgvector
→ store predicted_class as metadata
→ update Redis document status
```

Redis document status uses:

```text
doc:status:{document_id}
```

Statuses include:

```text
uploaded → extracted → chunked → embedded → indexed
```

If RAG indexing fails, the classifier upload flow still returns its prediction and logs a warning.

### Docker model packaging

The Docker image downloads and packages the PubMedBERT embedding model during build:

```text
/models/pubmedbert-base-embeddings
```

At runtime the app loads from `LOCAL_MODEL_DIR`, so requests do not download from Hugging Face.

Build locally:

```bash
docker build -t pardhu156/tmf-classifier:latest .
```

### S3 RAG artifacts

RAG artifacts use the existing Stage 4 S3 bucket configured by `AWS_S3_BUCKET_NAME`. The app does not create a new bucket.

Idempotent prefixes:

```text
rag-artifacts/
rag-artifacts/embedding-models/pubmedbert/
rag-artifacts/ingestion-reports/
rag-artifacts/rag-evaluation/
rag-artifacts/failed-ingestions/
```

Use S3 for:

- embedding model backup
- indexing summaries/reports
- evaluation reports
- failed ingestion logs

Do not store active vector embeddings in S3. Active embeddings stay in PostgreSQL + pgvector.

Upload optional RAG artifacts:

```bash
python scripts/upload_rag_artifacts.py
```

The Stage 4 bootstrap script also ensures these RAG prefixes exist:

```bash
python scripts/bootstrap_cloud_uploads.py
```

### DockerHub CI/CD

Current CI/CD flow:

```text
GitHub push
→ GitHub Actions
→ build Docker image with local PubMedBERT model
→ push image to Docker Hub
```

Required GitHub secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Cloud deployment can be added later. The current workflow only builds/tests and pushes the Docker image to DockerHub.

### RAG retrieval scoping

RAG uses metadata filtering before pgvector similarity search or PostgreSQL full-text search. This avoids pulling irrelevant chunks from unrelated documents.

Supported metadata filters:

- `source_type`
- `verification_status`
- `document_id`
- `predicted_class`
- `file_name`

Retrieval priority:

1. If `document_id` is provided, search only that document.
2. If no `document_id` is provided, try fuzzy filename matching from the question.
3. If no filename match is found, default to `scope="master"` and search only `MASTER_DATA`.

Optional `scope` values:

```text
document
master
verified
all
class
```

Default:

```text
scope = master
```

Use `scope="all"` only when you explicitly want retrieval across master data and uploaded documents.

### RAG API

Ask a question across all indexed documents:

```bash
curl -X POST "http://127.0.0.1:8000/rag/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the study objective?", "scope": "all"}'
```

Ask within one document:

```bash
curl -X POST "http://127.0.0.1:8000/rag/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the primary endpoint?", "document_id": "1"}'
```

Ask from trusted master data only, which is the default:

```bash
curl -X POST "http://127.0.0.1:8000/rag/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the study objective?"}'
```

Ask within one predicted TMF class:

```bash
curl -X POST "http://127.0.0.1:8000/rag/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the inclusion criteria?", "predicted_class": "protocol", "scope": "class"}'
```

Other RAG endpoints:

```text
GET /rag/documents
GET /rag/status/{document_id}
GET /rag/metrics
POST /rag/index-master-data
```

The answer generator is instructed to use only retrieved chunks. If there is not enough evidence, it returns:

```text
I could not find enough information in the uploaded documents.
```

### Semantic cache behavior

The Redis semantic cache stores question embeddings and answers. It is scoped to prevent reusing an answer from the wrong document set:

```text
all_documents
document:{document_id}
class:{predicted_class}
```

If a new question is semantically similar to a previous question in the same scope and the similarity is above `SEMANTIC_CACHE_THRESHOLD`, the cached answer is returned.

## Stage 6: Agentic TMF Filer

Stage 6 adds a confidence-based filing layer around the existing upload, prediction, cloud persistence, PostgreSQL, Redis, and RAG workflow.

The existing classifier and RAG pipelines are reused. Stage 6 only decides whether a document can be auto-filed or must wait for human review.

### Configuration

Add these values to `.env`:

```env
AUTO_APPROVAL_THRESHOLD=0.90
MIN_CONFIDENCE_GAP=0.10
MANUAL_REVIEW_QUEUE_NAME=manual_review:pending
```

Decision rules:

- if `confidence >= AUTO_APPROVAL_THRESHOLD` and top-1/top-2 confidence gap is safe, the document is auto-filed
- otherwise the document is pushed to the Redis manual-review queue
- low-confidence documents are not ingested into final RAG until a human confirms the final class

### Agentic filing storage layout

Stage 6 uses these logical S3 prefixes:

```text
agentic_tmf_workspace/
├── tmf/<class>/
├── pending_review/
├── pending_training/<class>/
├── approved_training/
└── rejected_training/
```

The same structure is mirrored locally under `agentic_tmf_workspace/` for easier local review/debugging:

```text
agentic_tmf_workspace/
├── tmf/<class>/
├── pending_review/
├── pending_training/<class>/
├── approved_training/
├── rejected_training/
└── metadata/
```

The local `agentic_tmf_workspace/` folder is ignored by Git.

New documents first go to `pending_training/` after filing. They are not approved for model training until an admin/manager approves them.

Metadata is written in structured folders:

```text
agentic_tmf_workspace/metadata/<status>/<class_or_unclassified>/<doc_id>.json
```

If older Stage 6 files already exist in S3 under `cloud/`, migrate them once:

```bash
python scripts/migrate_agentic_s3_prefix.py --dry-run true
python scripts/migrate_agentic_s3_prefix.py --dry-run false
python scripts/migrate_agentic_s3_prefix.py --dry-run false --delete-old true
```

Only run the final command after confirming the copied files exist under `agentic_tmf_workspace/`.

After manual review succeeds, the document is removed from `pending_review/` and copied into the final TMF class folder plus `pending_training/`. After training approval/rejection, the document is removed from `pending_training/` and copied into `approved_training/` or `rejected_training/`.

During upload/review, progress messages are logged in the Uvicorn terminal, for example:

```text
Stage 6 progress [1/7]: calculated file hash
Stage 6 progress [4/7]: running classifier on chunks
Stage 6 progress [7/7]: metadata/audit saved
```

### Main upload flow

Use the existing upload endpoint:

```text
POST /predict-file
```

High-confidence response includes:

```json
{
  "agentic_action": "auto_filed",
  "final_class": "protocol",
  "document_status": "pending_training_approval",
  "rag_ingestion_status": "rag_ingested"
}
```

Low-confidence response includes:

```json
{
  "agentic_action": "manual_review_required",
  "final_class": null,
  "document_status": "pending_review",
  "rag_ingestion_status": "not_started"
}
```

### Manual review endpoints

List pending review items:

```text
GET /agentic/reviews
```

Submit a corrected class:

```bash
curl -X POST "http://127.0.0.1:8000/agentic/reviews/1/submit" \
  -H "Content-Type: application/json" \
  -d '{"corrected_class": "protocol", "reviewer_id": "admin"}'
```

After correction, the document is filed, metadata is updated, RAG ingestion runs, and the document becomes pending training approval.

### Training feedback endpoints

Approve for future training:

```text
POST /agentic/training/{doc_id}/approve
```

Reject from future training:

```text
POST /agentic/training/{doc_id}/reject
```

Correct an already auto-filed document:

```text
POST /agentic/documents/{doc_id}/correct
```

Metrics:

```text
GET /agentic/metrics
```

Metrics include auto-file rate, manual-review rate, duplicate count, average confidence, RAG additions, and training approval counts.

Stage 7.1 RBAC now protects upload, RAG, review, training approval, metrics, and audit/user-management APIs according to role.

## Stage 7.1: Authentication + RBAC

Stage 7.1 adds database-backed users, secure password hashing, bearer-token authentication, and role-based access control without changing the classifier, RAG, or agentic filing internals.

### Demo login credentials

These accounts are for local development and testing only:

| Email | Password | Role |
| --- | --- | --- |
| `user@test.com` | `user123` | User |
| `manager@test.com` | `manager123` | Manager |
| `admin@test.com` | `admin123` | Admin |

Production should replace demo accounts with proper user management or enterprise SSO.

### Auth environment variables

Add these to `.env`:

```env
JWT_SECRET_KEY=replace_with_a_long_random_secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

Never commit real JWT secrets.

### Authentication flow

Login with email and password:

```bash
curl -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@test.com", "password": "admin123"}'
```

The response includes:

- `access_token`
- `token_type`
- `expires_in_minutes`
- safe `user` details with no password hash
- `dashboard`, one of `User Dashboard`, `Manager Dashboard`, or `Admin Dashboard`

Use the token on protected endpoints:

```bash
curl "http://127.0.0.1:8000/auth/me" \
  -H "Authorization: Bearer <access_token>"
```

Logout is stateless:

```text
POST /auth/logout
```

Clients should discard the bearer token. Expired or invalid tokens return `401`; insufficient roles return `403`.

### RBAC permissions

User:

- `POST /predict`
- `POST /predict-file`
- `GET /documents/my-uploads`
- `POST /rag/ask`
- `GET /rag/documents`
- `GET /rag/status/{document_id}`

Manager:

- everything User can do
- `GET /agentic/reviews`
- `POST /agentic/reviews/{doc_id}/submit`
- `POST /agentic/documents/{doc_id}/correct`

Admin:

- everything Manager can do
- `GET /users`
- `POST /users`
- `PATCH /users/{user_id}`
- `POST /documents/{doc_id}/verify`
- `GET /agentic/training/pending`
- `POST /agentic/training/{doc_id}/approve`
- `POST /agentic/training/{doc_id}/reject`
- `GET /agentic/metrics`
- `GET /rag/metrics`
- `POST /rag/index-master-data`
- `GET /audit-logs`
- `POST /retrain`

Public health/probe endpoints remain open: `/`, `/health`, and `/model-info`.

### Managing users manually

Admin API:

```bash
curl -X POST "http://127.0.0.1:8000/users" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"name":"Study Lead","email":"lead@test.com","password":"lead123","role":"Manager","is_active":true}'

curl -X PATCH "http://127.0.0.1:8000/users/4" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"role":"Admin"}'
```

Database/CLI path:

```bash
python scripts/seed_demo_users.py
```

`scripts/seed_demo_users.py` is idempotent and is run automatically by `docker compose up --build` through the `db-seeder` service.

### RAG document access filtering

RAG retrieval uses document metadata for RBAC. Files stay in the same S3/local storage locations; access is controlled by `access_level` metadata stored on `rag_documents`, `rag_chunks`, and `document_metadata`.

Access rules:

- User retrieves `User` documents.
- Manager retrieves `User` and `Manager` documents.
- Admin retrieves `User`, `Manager`, and `Admin` documents.

If a user asks for a document outside their role scope, the API returns:

```text
You do not have permission to access documents relevant to this query.
```

The classifier still predicts only TMF classes such as `protocol`, `safety_report`, and `statistical_analysis_plan`; RBAC is an additional retrieval metadata filter.

## Stage 8 AWS deployment

The production-like POC deployment plan is documented in [docs/aws-deployment.md](docs/aws-deployment.md). It keeps the target architecture low-cost: S3 + CloudFront for the frontend, Elastic Beanstalk single-instance Docker for FastAPI, RDS PostgreSQL, ElastiCache Redis, S3 document storage, Docker Hub, GitHub Actions, and CloudWatch.
