# Stage 8: AWS Deployment POC

## Tech Stack

- GitHub Actions
- Docker Hub
- Elastic Beanstalk Docker platform
- Amazon S3
- Amazon RDS PostgreSQL
- Redis/ElastiCache target
- CloudWatch logs
- S3 static website hosting

## Key Steps

1. Prepared Docker image for Elastic Beanstalk.
2. Added deployment configuration files.
3. Added GitHub Actions deployment support.
4. Used Docker Hub as the image registry.
5. Created/used S3 buckets for frontend static files and TMF document storage.
6. Created RDS PostgreSQL for metadata.
7. Captured Elastic Beanstalk environment proof.
8. Captured CloudWatch log group proof.
9. Documented cost-saving setup and teardown guidance.

## Intended Low-Cost POC Architecture

```text
GitHub
  -> GitHub Actions
  -> Docker Hub
  -> Elastic Beanstalk
  -> FastAPI backend
       -> RDS PostgreSQL
       -> S3 documents
       -> Redis cache
       -> CloudWatch logs

Frontend static files
  -> S3 static website hosting
```

## Evidence Captured

| Evidence | Location |
| --- | --- |
| S3 bucket overview | `docs/demo-evidence/aws-evidence/01-s3-buckets-overview.png` |
| S3 document bucket | `docs/demo-evidence/aws-evidence/02-s3-document-bucket-root.png` |
| Uploaded safety report in S3 | `docs/demo-evidence/aws-evidence/04-s3-uploaded-safety-report-files.png` |
| RDS PostgreSQL instance | `docs/demo-evidence/aws-evidence/06-rds-postgresql-instance.png` |
| Elastic Beanstalk environment | `docs/demo-evidence/aws-evidence/07-elastic-beanstalk-environment.png` |
| CloudWatch log groups | `docs/demo-evidence/aws-evidence/08-cloudwatch-log-groups.png` |
| CloudWatch EB logs | `docs/demo-evidence/aws-evidence/09-cloudwatch-elastic-beanstalk-log-details.png` |
| GitHub Actions deployment | `docs/demo-evidence/ci-cd-evidence/` |
| Docker Hub image | `docs/demo-evidence/ci-cd-evidence/06-docker-hub-image-tags.png` |

## Cost-Saving Decisions

- Single-instance Elastic Beanstalk.
- Small/free-tier style RDS where available.
- No ECS/Fargate.
- No ECR.
- No CodePipeline.
- No NAT Gateway.
- CloudFront optional, not required for simplified POC.
- Terminate AWS resources after demo screenshots to avoid cost.

## Files

| Path | Purpose |
| --- | --- |
| `Dockerfile` | Backend image for deployment |
| `.github/workflows/ci-cd.yml` | CI/CD and deployment workflow |
| `.ebextensions/01_environment.config` | Elastic Beanstalk environment options |
| `deploy/elasticbeanstalk/Dockerrun.aws.json.template` | EB Docker image template |
| `docs/aws-deployment.md` | Full deployment guide |
| `docs/demo-evidence/aws-evidence/` | AWS screenshots |

## Limitations

- Live AWS environment may be terminated for cost reasons.
- Portfolio proof can rely on screenshots plus local run.
- Production needs IaC, automated backups, alarms, stricter secrets management, and environment separation.
