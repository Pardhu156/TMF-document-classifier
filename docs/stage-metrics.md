# Stage Metrics And Evidence

This document summarizes measurable proof points for the TMF Document Classifier POC. Values come from checked-in metadata and report artifacts where available.

## Source Artifacts

| Artifact | Purpose |
| --- | --- |
| `metadata/dataset_metadata.json` | Dataset and split summary |
| `metadata/evaluation_metadata.json` | Final classifier evaluation summary |
| `artifacts/metrics.json` | Chunk-level and document-level classification reports |
| `artifacts/run_metadata.json` | Training/evaluation run metadata and package versions |
| `reports/rag_metrics_summary.txt` | Human-readable RAG and Redis cache metrics |
| `reports/rag_metrics_summary.json` | Structured RAG and Redis cache metrics |
| `docs/demo-evidence/` | UI, AWS, CI/CD, and local verification screenshots |

## Stage 1: Classifier Metrics

Tech stack:

- BioClinicalBERT: `emilyalsentzer/Bio_ClinicalBERT`
- Transformers
- scikit-learn
- pandas
- document-level split to prevent leakage

Dataset:

| Metric | Value |
| --- | --- |
| TMF classes | 3 |
| Classes | `protocol`, `safety_report`, `statistical_analysis_plan` |
| Total documents | 44 |
| Total chunks | 2,151 |
| Protocol documents | 15 |
| Safety report documents | 15 |
| Statistical analysis plan documents | 14 |
| Protocol chunks | 1,046 |
| Safety report chunks | 456 |
| Statistical analysis plan chunks | 649 |
| Chunk size | 512 |
| Chunk overlap | 50 |

Split:

| Split | Documents | Chunks |
| --- | ---: | ---: |
| Train | 26 | 1,418 |
| Validation | 9 | 359 |
| Test | 9 | 374 |

Evaluation:

| Metric | Value |
| --- | ---: |
| Chunk-level accuracy | 61.50% |
| Chunk-level macro F1 | 68.03% |
| Document-level accuracy | 77.78% |
| Document-level macro F1 | 75.00% |

Per-class document-level F1:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Protocol | 100.00% | 33.33% | 50.00% |
| Safety report | 100.00% | 100.00% | 100.00% |
| Statistical analysis plan | 60.00% | 100.00% | 75.00% |

POC interpretation:

- The model proves the end-to-end classification workflow.
- Document-level voting improves usability over raw chunk predictions.
- Protocol recall is the main model weakness and is a good target for future data expansion.

## Stage 2: MLOps Metrics

Tech stack:

- DVC for dataset/model artifact versioning
- MLflow/Dagshub hooks
- JSON metadata files
- `.env` secret separation

Tracked metadata:

| Metadata | Value |
| --- | --- |
| Dataset version | `v1.0.0` |
| Model version | `v1.0.0` |
| Training status | `legacy_checkpoint_imported` |
| Epochs configured | 3 |
| Learning rate | `2e-5` |
| Train batch size | 8 |
| Eval batch size | 8 |
| Max length | 512 |

Package versions from `artifacts/run_metadata.json`:

| Package | Version |
| --- | --- |
| torch | 2.2.2 |
| transformers | 4.38.2 |
| datasets | 5.0.0 |
| scikit-learn | 1.7.2 |
| pandas | 2.3.3 |
| numpy | 1.26.4 |

POC interpretation:

- The project has enough metadata to explain how a model/checkpoint was produced.
- DVC and MLflow are integrated as a portfolio-grade MLOps foundation.
- Full production model registry and automated promotion are future improvements.

## Stage 3: API, Docker, And CI/CD Metrics

Tech stack:

- FastAPI
- Pydantic
- Docker
- Docker Compose
- GitHub Actions
- Docker Hub
- pytest

Measured/visible proof:

| Area | Evidence |
| --- | --- |
| API serving | `app.py` exposes health, auth, prediction, RAG, review, metrics, and admin endpoints |
| CI/CD | GitHub Actions screenshots in `docs/demo-evidence/ci-cd-evidence/` |
| Container registry | Docker Hub screenshot in `docs/demo-evidence/ci-cd-evidence/06-docker-hub-image-tags.png` |
| Test suite | 24 pytest files |
| Dockerized dependencies | PostgreSQL pgvector and Redis through `docker-compose.yml` |

POC interpretation:

- The app is reproducible locally and packaged for deployment.
- CI/CD proof is strong enough for a portfolio demo.

## Stage 4: Cloud Persistence Metrics

Tech stack:

- AWS S3-compatible storage
- PostgreSQL
- SQLAlchemy
- SHA256 duplicate detection
- audit logging

Stored objects and tables:

| Item | Stored In |
| --- | --- |
| Raw uploaded file | S3/local workspace |
| Extracted text | S3/local workspace |
| Metadata JSON | S3/local workspace |
| Document row | PostgreSQL `documents` |
| Chunk rows | PostgreSQL `chunks` |
| Prediction row | PostgreSQL `predictions` |
| Chunk prediction rows | PostgreSQL `chunk_predictions` |
| User-facing document metadata | PostgreSQL `document_metadata` |
| Audit events | PostgreSQL `audit_logs` |

Evidence:

- S3 bucket screenshots in `docs/demo-evidence/aws-evidence/`.
- RDS PostgreSQL screenshot in `docs/demo-evidence/aws-evidence/06-rds-postgresql-instance.png`.

POC interpretation:

- Persistence is modular and can run locally or with AWS credentials.
- S3 stores files/artifacts; PostgreSQL stores queryable metadata.

## Stage 5: RAG And Redis Metrics

Tech stack:

- PubMedBERT embeddings
- PostgreSQL pgvector
- PostgreSQL keyword search
- hybrid retriever
- Gemini generation
- Redis exact cache
- Redis semantic cache
- MLflow/Dagshub logging hooks

RAG evaluation from `reports/rag_metrics_summary.txt`:

| Metric | Value |
| --- | ---: |
| Evaluation questions | 3 |
| API calls | 9 |
| Average retrieved chunks | 5.00 |
| Average top chunk score | 1.01 |
| Average retrieval score | 0.75 |
| Source verification success rate | 100.00% |
| Citation presence rate | 100.00% |
| Answer availability rate | 100.00% |
| Average keyword coverage | 100.00% |

Redis cache performance:

| Metric | Value |
| --- | ---: |
| Exact cache hit rate | 33.33% |
| Semantic cache hit rate | 11.11% |
| Cache miss rate | 55.56% |
| Average uncached latency | 4,816.40 ms |
| Average cached latency | 779.75 ms |
| Average semantic cache latency | 682.00 ms |
| Exact cache speedup | 7.05x |
| Semantic cache speedup | 5.61x |
| Overall latency reduction | 83.81% |
| Total cache time saved | 24,222 ms |
| Total cache time saved percent | 88.59% |

Per-question examples:

| Question | Uncached | Exact Cache | Semantic Run | Top File |
| --- | ---: | ---: | ---: | --- |
| What is the study objective? | 9,962 ms | 913 ms | 682 ms | `protocol/Prot_015.docx` |
| What are the inclusion criteria? | 3,597 ms | 581 ms | 3,157 ms | `protocol/Prot_015.docx` |
| What are the adverse event reporting requirements? | 3,820 ms | 943 ms | 3,546 ms | `protocol/Prot_011.pdf` |

POC interpretation:

- Redis materially improves repeated query latency.
- Semantic cache works, but threshold/data tuning could improve hit rate.
- Retrieval is grounded with citations and source verification.

## Stage 6: Agentic Filing Metrics

Tech stack:

- Existing classifier
- S3/local document workspace
- PostgreSQL metadata
- Redis manual review queue
- RAG ingestion hook
- audit logs

Important configurable thresholds:

| Setting | Default |
| --- | --- |
| `AUTO_APPROVAL_THRESHOLD` | 0.90 |
| `MIN_CONFIDENCE_GAP` | 0.10 |
| `MANUAL_REVIEW_QUEUE_NAME` | `manual_review:pending` |

Workflow metrics exposed by `GET /agentic/metrics`:

- auto-file count/rate
- manual-review count/rate
- duplicate count
- average confidence
- RAG ingestion count
- pending training approval count
- approved/rejected training counts

POC interpretation:

- The project demonstrates human-in-the-loop document filing.
- Low-confidence documents are blocked from final RAG ingestion until corrected/approved.

## Stage 7: Auth, RBAC, And Frontend Metrics

Tech stack:

- JWT bearer authentication
- password hashing
- PostgreSQL users table
- FastAPI role dependencies
- role-based frontend routes
- RAG metadata access filters

Roles:

| Role | Retrieval Access |
| --- | --- |
| User | `User` documents |
| Manager | `User`, `Manager` documents |
| Admin | `User`, `Manager`, `Admin` documents |

Key tests/evidence:

- `tests/test_auth_rbac.py`
- `tests/test_rag_service.py`
- `tests/test_rag_retriever.py`
- UI screenshots in `docs/demo-evidence/app-ui/`
- RBAC denial screenshot: `docs/demo-evidence/app-ui/04-ai-document-assistant-rbac-denial.png`

POC interpretation:

- Backend remains the source of authorization truth.
- Frontend role navigation supports user experience but does not replace backend checks.

## Stage 8: AWS Deployment Evidence

Tech stack:

- S3 static website hosting
- Elastic Beanstalk Docker environment
- RDS PostgreSQL
- S3 document bucket
- CloudWatch log groups
- GitHub Actions
- Docker Hub

Captured evidence:

| Area | Screenshot |
| --- | --- |
| S3 buckets | `docs/demo-evidence/aws-evidence/01-s3-buckets-overview.png` |
| S3 document storage | `docs/demo-evidence/aws-evidence/02-s3-document-bucket-root.png` |
| Uploaded classified document | `docs/demo-evidence/aws-evidence/04-s3-uploaded-safety-report-files.png` |
| RDS PostgreSQL | `docs/demo-evidence/aws-evidence/06-rds-postgresql-instance.png` |
| Elastic Beanstalk | `docs/demo-evidence/aws-evidence/07-elastic-beanstalk-environment.png` |
| CloudWatch logs | `docs/demo-evidence/aws-evidence/08-cloudwatch-log-groups.png` |
| EB log details | `docs/demo-evidence/aws-evidence/09-cloudwatch-elastic-beanstalk-log-details.png` |

POC interpretation:

- The screenshots are enough to prove cloud deployment effort before terminating resources for cost control.
- For a portfolio demo, local execution plus AWS evidence is acceptable if live hosting is turned off.

## Overall Ratings For Portfolio POC

| Area | Rating |
| --- | ---: |
| AI/ML | 8.0 / 10 |
| MLOps | 8.0 / 10 |
| Backend | 8.5 / 10 |
| System Design | 8.5 / 10 |
| Resume Value | 9.0 / 10 |
| Production Readiness | 6.5 / 10 |

Production readiness is lower because a true production system would need enterprise SSO, stronger monitoring, security hardening, automated model promotion, backups, IaC, and stronger scale/load testing.
