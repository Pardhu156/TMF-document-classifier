# TMF Document Classifier And AI Document Intelligence Platform

Enterprise-style Trial Master File (TMF) document intelligence POC built with BioClinicalBERT, FastAPI, PostgreSQL, Redis, S3-style storage, RAG, JWT authentication, RBAC, Docker, GitHub Actions, and a role-based frontend console.

This project started as a 3-class TMF classifier and was expanded into a production-like regulated document workflow:

- classify uploaded TMF documents
- file high-confidence documents automatically
- send low-confidence documents to manager review
- store document, chunk, prediction, and audit metadata
- answer questions over authorized documents using RAG
- accelerate repeated RAG queries with Redis exact and semantic cache
- protect workflows with JWT authentication and role-based access control
- expose User, Manager, and Admin dashboards
- package and deploy with Docker, GitHub Actions, Docker Hub, S3, Elastic Beanstalk, RDS, and CloudWatch

## Current POC Status

| Area | Status |
| --- | --- |
| Classifier | Complete POC with BioClinicalBERT and document-level voting |
| RAG | Complete POC with pgvector, hybrid retrieval, Redis cache, citations, and RBAC filters |
| Auth/RBAC | Complete POC with JWT, password hashing, user roles, and protected APIs |
| Agentic Filing | Complete POC with auto-file/manual-review/training-approval flow |
| Frontend | Complete role-based SaaS-style console |
| MLOps | DVC guidance, MLflow/Dagshub hooks, metadata files, CI/CD, Docker Hub |
| AWS Hosting | POC evidence captured for S3, Elastic Beanstalk, RDS, and CloudWatch |

## Tech Stack

| Layer | Tools |
| --- | --- |
| ML/NLP | BioClinicalBERT, Hugging Face Transformers, scikit-learn |
| RAG | PubMedBERT embeddings, PostgreSQL pgvector, keyword search, hybrid retrieval, Gemini generation |
| Backend | FastAPI, Pydantic, SQLAlchemy |
| Auth | JWT bearer tokens, password hashing, role dependencies |
| Database | PostgreSQL, pgvector |
| Cache/Queues | Redis exact cache, semantic cache, manual review queue |
| Storage | AWS S3-compatible document/artifact storage, local mirrored workspace |
| Frontend | React/Vite-style static console served by FastAPI/S3 |
| MLOps | DVC, MLflow/Dagshub hooks, structured metadata, evaluation artifacts |
| Deployment | Docker, Docker Compose, GitHub Actions, Docker Hub, Elastic Beanstalk, S3, RDS, CloudWatch |
| Testing | pytest, FastAPI TestClient, mocked S3/PostgreSQL/Redis paths |

## Key Metrics

| Metric | Value |
| --- | --- |
| TMF classes | `protocol`, `safety_report`, `statistical_analysis_plan` |
| Source documents | 44 |
| Total chunks | 2,151 |
| Train/validation/test documents | 26 / 9 / 9 |
| Train/validation/test chunks | 1,418 / 359 / 374 |
| Chunk-level accuracy | 61.50% |
| Chunk-level macro F1 | 68.03% |
| Document-level accuracy | 77.78% |
| Document-level macro F1 | 75.00% |
| RAG retrieved chunks per query | 5 average |
| RAG citation presence | 100% in sample evaluation |
| RAG keyword coverage | 100% in sample evaluation |
| Redis cached latency reduction | 83.81% average |
| Redis exact cache speedup | 7.05x average |
| Redis semantic cache speedup | 5.61x average |
| Test files | 24 pytest files |

Detailed metrics are documented in [docs/stage-metrics.md](docs/stage-metrics.md).

## Architecture

```text
User / Manager / Admin
        |
        v
Role-Based Frontend Console
        |
        v
FastAPI Backend
   |        |          |           |
   v        v          v           v
JWT/RBAC  Classifier  RAG Service  Agentic Filing
            |          |           |
            v          v           v
       BioClinicalBERT pgvector     Redis Review Queue
                       |
                       v
             Redis Exact/Semantic Cache
                       |
                       v
PostgreSQL Metadata + S3 Document Storage + Audit Logs
```

AWS POC target:

```text
GitHub -> GitHub Actions -> Docker Hub -> Elastic Beanstalk -> FastAPI
                                                      |
                                                      +-> RDS PostgreSQL
                                                      +-> S3 Documents
                                                      +-> Redis/ElastiCache
                                                      +-> CloudWatch Logs

Frontend Static Files -> S3 Static Website Hosting
```

## Main Features

### Classification

- Extracts PDF, DOCX, and TXT text.
- Cleans and chunks long documents with overlap.
- Runs BioClinicalBERT on chunks.
- Aggregates chunk predictions into document-level class using voting.
- Computes confidence from model confidence, vote confidence, and vote margin.
- Stores document, chunk, prediction, and chunk-level prediction metadata.

### Agentic TMF Filing

- Auto-files high-confidence documents.
- Sends low-confidence documents to manager manual review.
- Lets managers correct predicted class before final filing.
- Keeps documents pending training approval after filing.
- Lets admins approve or reject documents for future fine-tuning.
- Writes audit logs for important actions.

### RAG

- Indexes trusted master data and filed documents.
- Stores chunk embeddings in PostgreSQL pgvector.
- Uses semantic search plus keyword search.
- Supports all-doc retrieval and selected-document retrieval.
- Uses Redis exact cache and semantic cache for repeated/paraphrased questions.
- Returns grounded answers with citations/sources.
- Applies RBAC metadata filters before retrieval.

### Authentication And RBAC

Roles:

- `User`: upload/classify documents, view own documents, ask RAG questions.
- `Manager`: User features plus manual review, class correction, team documents, team analytics.
- `Admin`: Manager features plus user management, training approval, model management, re-indexing, metrics, audit logs, system health.

Demo users:

| Email | Password | Role |
| --- | --- | --- |
| `user@test.com` | `user123` | User |
| `manager@test.com` | `manager123` | Manager |
| `admin@test.com` | `admin123` | Admin |

Demo users are for development only. A production version should use enterprise SSO or proper user management.

## API Overview

| Area | Endpoints |
| --- | --- |
| Health | `GET /`, `GET /health`, `GET /model-info` |
| Auth | `POST /auth/login`, `POST /auth/logout`, `GET /auth/me` |
| Users | `GET /users`, `POST /users`, `PATCH /users/{user_id}` |
| Classification | `POST /predict`, `POST /predict-file` |
| Documents | `GET /documents/my-uploads`, `POST /documents/{doc_id}/verify` |
| Manager Review | `GET /agentic/reviews`, `POST /agentic/reviews/{doc_id}/submit`, `POST /agentic/documents/{doc_id}/correct` |
| Training Approval | `GET /agentic/training/pending`, `POST /agentic/training/{doc_id}/approve`, `POST /agentic/training/{doc_id}/reject` |
| RAG | `POST /rag/ask`, `GET /rag/documents`, `GET /rag/status/{document_id}`, `GET /rag/metrics`, `POST /rag/index-master-data` |
| Admin/Metrics | `GET /agentic/metrics`, `GET /audit-logs`, `POST /retrain` |

## Quick Start

Full setup is documented in [how-to-start.md](how-to-start.md).

Minimal local flow:

```bash
cp .env.example .env
docker compose up postgres redis
python scripts/seed_demo_users.py
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000/console
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Full Docker Compose flow:

```bash
cp .env.example .env
dvc pull
docker compose up --build
```

## Documentation

| Document | Purpose |
| --- | --- |
| [how-to-start.md](how-to-start.md) | Step-by-step local startup and testing guide |
| [docs/stage-metrics.md](docs/stage-metrics.md) | Metrics and proof points for each stage |
| [docs/poc-prototype-readiness.md](docs/poc-prototype-readiness.md) | POC/prototype readiness notes and next improvements |
| [docs/aws-deployment.md](docs/aws-deployment.md) | AWS deployment plan and manual setup notes |
| [docs/demo-evidence/README.md](docs/demo-evidence/README.md) | Screenshot evidence index for UI, AWS, CI/CD, and local verification |
| [docs/stages/](docs/stages) | Detailed stage-by-stage implementation history |

Stage documentation:

- [Stage 1: Classifier](docs/stages/01-classifier.md)
- [Stage 2: MLOps Foundation](docs/stages/02-mlops-foundation.md)
- [Stage 3: API, Docker, CI/CD](docs/stages/03-api-docker-cicd.md)
- [Stage 4: Cloud Persistence And Retraining Foundation](docs/stages/04-cloud-persistence-retraining.md)
- [Stage 5: RAG And Redis Cache](docs/stages/05-rag-redis.md)
- [Stage 6: Agentic TMF Filing](docs/stages/06-agentic-filing.md)
- [Stage 7: Auth, RBAC, And Frontend](docs/stages/07-auth-rbac-frontend.md)
- [Stage 8: AWS Deployment POC](docs/stages/08-aws-deployment.md)

## Security Notes

- Do not commit `.env`, AWS keys, JWT secrets, DagsHub tokens, database passwords, uploaded private documents, or local cache files.
- Password hashes are never returned by API responses.
- JWT authentication protects user workflows.
- Backend role dependencies enforce authorization; frontend route guards are only UX support.
- RAG applies role-based metadata filters before semantic search.
- Demo credentials are development-only.

## Repository Layout

```text
app.py                         FastAPI app and API routes
frontend/                      Static role-based frontend console
src/                           Classifier, RAG, auth, storage, database, and agentic workflow modules
scripts/                       Seed, indexing, bootstrap, reset, and upload utilities
tests/                         Unit and integration-style tests
artifacts/                     Model, label encoder, split files, and evaluation artifacts
metadata/                      Dataset/model/training/evaluation metadata
reports/                       RAG/Redis metrics reports
docs/                          Project documentation and demo evidence
docker-compose.yml             Local PostgreSQL, Redis, seeder, and API stack
Dockerfile                     Backend Docker image definition
```

## Portfolio Summary

This is a strong AI/ML portfolio POC because it demonstrates more than model training. It shows the surrounding system expected in real regulated document workflows: ingestion, metadata, storage, auditability, RAG, caching, RBAC, review queues, CI/CD, containerization, and cloud deployment evidence.
