# TMF Classifier Demo Evidence

This folder contains screenshots captured for the portfolio POC before terminating AWS resources for cost control.

The application evidence is split into:

- `app-ui/`: role-based product UI, RAG, classification, approval, admin, and monitoring screens.
- `aws-evidence/`: AWS deployment proof for S3, RDS PostgreSQL, Elastic Beanstalk, and CloudWatch.
- `ci-cd-evidence/`: GitHub Actions and Docker Hub delivery proof.
- `mlops-evidence/`: DagsHub, DVC, and MLflow/model tracking proof.
- `local-verification/`: local runtime/API verification proof.
- `troubleshooting/`: non-demo screenshots kept separately so they do not interrupt the main portfolio flow.

## App UI Evidence

| Screenshot | Shows |
| --- | --- |
| [01-user-workspace-dashboard.png](app-ui/01-user-workspace-dashboard.png) | User dashboard and role-based workspace |
| [02-upload-and-classify.png](app-ui/02-upload-and-classify.png) | Upload and classification flow |
| [03-my-documents.png](app-ui/03-my-documents.png) | User document table |
| [04-ai-document-assistant-rbac-denial.png](app-ui/04-ai-document-assistant-rbac-denial.png) | RAG assistant with RBAC denial behavior |
| [05-classification-history.png](app-ui/05-classification-history.png) | Classification history |
| [06-query-history.png](app-ui/06-query-history.png) | RAG query history |
| [07-manager-review-dashboard.png](app-ui/07-manager-review-dashboard.png) | Manager review dashboard |
| [08-manager-manual-classification-review.png](app-ui/08-manager-manual-classification-review.png) | Manager class correction workflow |
| [09-manager-approved-documents.png](app-ui/09-manager-approved-documents.png) | Manager-approved documents |
| [10-manager-team-analytics-redis-metrics.png](app-ui/10-manager-team-analytics-redis-metrics.png) | Team analytics and Redis cache metrics |
| [11-manager-ai-assistant.png](app-ui/11-manager-ai-assistant.png) | Manager RAG assistant |
| [12-admin-system-overview.png](app-ui/12-admin-system-overview.png) | Admin system dashboard |
| [13-admin-user-management.png](app-ui/13-admin-user-management.png) | Admin user management |
| [14-admin-document-repository.png](app-ui/14-admin-document-repository.png) | Admin document repository |
| [15-admin-classification-pipeline.png](app-ui/15-admin-classification-pipeline.png) | Classification pipeline monitoring |
| [16-admin-model-management-training-approval.png](app-ui/16-admin-model-management-training-approval.png) | Model management and fine-tuning approval queue |
| [17-admin-embedding-vector-index.png](app-ui/17-admin-embedding-vector-index.png) | Embedding and vector index page |
| [18-admin-redis-cache-monitor.png](app-ui/18-admin-redis-cache-monitor.png) | Redis cache monitor |
| [19-admin-ai-retrieval-pipeline.png](app-ui/19-admin-ai-retrieval-pipeline.png) | RAG pipeline observability |
| [20-manager-team-documents.png](app-ui/20-manager-team-documents.png) | Manager team documents |
| [21-admin-system-analytics.png](app-ui/21-admin-system-analytics.png) | Admin analytics |
| [22-admin-audit-logs.png](app-ui/22-admin-audit-logs.png) | Audit logs |
| [23-admin-system-health.png](app-ui/23-admin-system-health.png) | System health |
| [24-rag-answer-with-citations.png](app-ui/24-rag-answer-with-citations.png) | RAG answer with citations |
| [25-rag-selected-document-query.png](app-ui/25-rag-selected-document-query.png) | RAG retrieval scoped to a selected document |
| [26-landing-page.png](app-ui/26-landing-page.png) | Landing page |
| [27-login-page.png](app-ui/27-login-page.png) | Login page |

## AWS Evidence

| Screenshot | Shows |
| --- | --- |
| [01-s3-buckets-overview.png](aws-evidence/01-s3-buckets-overview.png) | S3 bucket overview |
| [02-s3-document-bucket-root.png](aws-evidence/02-s3-document-bucket-root.png) | TMF document storage bucket |
| [03-s3-agentic-workspace-prefix.png](aws-evidence/03-s3-agentic-workspace-prefix.png) | Agentic TMF workspace prefixes |
| [04-s3-uploaded-safety-report-files.png](aws-evidence/04-s3-uploaded-safety-report-files.png) | Uploaded classified document in S3 |
| [05-s3-document-bucket-prefixes.png](aws-evidence/05-s3-document-bucket-prefixes.png) | S3 document bucket structure |
| [06-rds-postgresql-instance.png](aws-evidence/06-rds-postgresql-instance.png) | RDS PostgreSQL instance |
| [07-elastic-beanstalk-environment.png](aws-evidence/07-elastic-beanstalk-environment.png) | Elastic Beanstalk Docker environment |
| [08-cloudwatch-log-groups.png](aws-evidence/08-cloudwatch-log-groups.png) | CloudWatch log groups |
| [09-cloudwatch-elastic-beanstalk-log-details.png](aws-evidence/09-cloudwatch-elastic-beanstalk-log-details.png) | Elastic Beanstalk logs in CloudWatch |

## CI/CD Evidence

| Screenshot | Shows |
| --- | --- |
| [01-github-actions-workflow-success.png](ci-cd-evidence/01-github-actions-workflow-success.png) | Successful GitHub Actions workflow run |
| [02-github-actions-job-steps.png](ci-cd-evidence/02-github-actions-job-steps.png) | CI/CD job step overview |
| [03-github-actions-build-push-docker-image.png](ci-cd-evidence/03-github-actions-build-push-docker-image.png) | Docker image build and push step |
| [04-github-actions-elastic-beanstalk-deploy.png](ci-cd-evidence/04-github-actions-elastic-beanstalk-deploy.png) | Elastic Beanstalk deployment step |
| [05-github-actions-frontend-s3-deploy.png](ci-cd-evidence/05-github-actions-frontend-s3-deploy.png) | Frontend deployment to S3 step |
| [06-docker-hub-image-tags.png](ci-cd-evidence/06-docker-hub-image-tags.png) | Docker Hub image/tag evidence |

## MLOps Evidence

| Screenshot | Shows |
| --- | --- |
| [01-dagshub-repository-dvc-files.png](mlops-evidence/01-dagshub-repository-dvc-files.png) | DagsHub repository with DVC-tracked project/data artifacts |
| [02-dagshub-experiments-table.png](mlops-evidence/02-dagshub-experiments-table.png) | DagsHub experiment tracking table |
| [03-mlflow-rag-metrics-run.png](mlops-evidence/03-mlflow-rag-metrics-run.png) | MLflow run details for RAG metrics |

## Local Verification

| Screenshot | Shows |
| --- | --- |
| [01-rag-rbac-api-and-mlflow-logs.png](local-verification/01-rag-rbac-api-and-mlflow-logs.png) | RAG, RBAC logging, API success responses, and MLflow/Dagshub run links |
| [02-docker-compose-postgres-redis-running.png](local-verification/02-docker-compose-postgres-redis-running.png) | PostgreSQL and Redis running through Docker Compose |

## Troubleshooting

| Screenshot | Shows |
| --- | --- |
| [01-uvicorn-reload-after-moving-photos-folder.png](troubleshooting/01-uvicorn-reload-after-moving-photos-folder.png) | Development-only Uvicorn reload issue after moving the screenshot folder |

## Deployment Note

The final POC deployment used S3 for static frontend hosting, Elastic Beanstalk for the Dockerized FastAPI backend, RDS PostgreSQL for metadata, S3 for uploaded TMF documents, Redis for cache/RAG acceleration, CloudWatch for logs, DVC for dataset/model artifact versioning, and MLflow/DagsHub for model/RAG experiment tracking. CloudFront and Fargate are not required for the simplified low-cost portfolio deployment.
