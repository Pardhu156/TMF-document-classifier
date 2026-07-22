# Stage 3: FastAPI, Docker, And CI/CD

## Tech Stack

- FastAPI
- Pydantic
- Uvicorn
- pytest
- Docker
- Docker Compose
- GitHub Actions
- Docker Hub

## Key Steps

1. Added a FastAPI serving layer.
2. Added health, model info, text prediction, and file prediction endpoints.
3. Added request/response schemas.
4. Added tests with FastAPI `TestClient`.
5. Packaged the app with Docker.
6. Added Docker Compose for local services.
7. Added GitHub Actions workflow.
8. Added Docker Hub image publishing.

## What Was Implemented

The classifier became a service instead of only a local script. The API supports:

- health checks
- model metadata
- text classification
- file upload classification
- later auth/RBAC, RAG, review, and admin workflows

## Core Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Basic API probe |
| `GET /health` | Health check |
| `GET /model-info` | Model artifact details |
| `POST /predict` | Classify raw text |
| `POST /predict-file` | Upload and classify PDF/DOCX/TXT |

## CI/CD Flow

```text
GitHub push
-> GitHub Actions
-> run tests
-> build Docker image
-> push image to Docker Hub
```

Evidence:

- `docs/demo-evidence/ci-cd-evidence/01-github-actions-workflow-success.png`
- `docs/demo-evidence/ci-cd-evidence/06-docker-hub-image-tags.png`

## Files

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI routes |
| `Dockerfile` | Backend image |
| `docker-compose.yml` | Local app/PostgreSQL/Redis stack |
| `.github/workflows/ci-cd.yml` | CI/CD workflow |
| `tests/` | API and pipeline tests |

## Limitations

- Docker image can be large because ML artifacts and embedding model support are included.
- Production deployment needs secrets and cloud services configured outside the repo.
