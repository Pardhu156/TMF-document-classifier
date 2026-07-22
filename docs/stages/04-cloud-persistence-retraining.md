# Stage 4: Cloud Persistence And Retraining Foundation

## Tech Stack

- AWS S3-compatible storage
- PostgreSQL
- SQLAlchemy
- SHA256 hashing
- audit logs
- conditional retraining coordinator

## Key Steps

1. Added S3 manager for document/artifact uploads.
2. Added PostgreSQL SQLAlchemy models.
3. Added repository layer for documents, chunks, predictions, metadata, users, model versions, and audit logs.
4. Added duplicate detection using file SHA256.
5. Added chunk hash tracking.
6. Added conditional retraining coordinator.
7. Added admin verification endpoint.
8. Added cloud bootstrap script for existing local assets.

## What Was Implemented

Uploaded documents can now persist beyond a single API response. The system stores:

- raw uploaded file
- extracted/cleaned text
- metadata JSON
- document row
- chunk rows
- document prediction
- chunk predictions
- audit log events

## Storage Responsibilities

| Data | Storage |
| --- | --- |
| Raw file | S3/local workspace |
| Extracted text | S3/local workspace |
| Metadata JSON | S3/local workspace |
| Queryable metadata | PostgreSQL |
| Chunk text/hash rows | PostgreSQL |
| Predictions | PostgreSQL |
| Audit logs | PostgreSQL |

## Conditional Retraining

The retraining foundation does not automatically fine-tune on every upload. It only prepares the workflow:

1. Admin verifies trusted labels.
2. Verified unused documents become candidates.
3. `/retrain` checks whether enough verified data exists.
4. A new model version record can be created.
5. Future GPU training can consume approved data.

## Files

| Path | Purpose |
| --- | --- |
| `src/cloud/s3_manager.py` | S3 upload/download helpers |
| `src/database/models.py` | SQLAlchemy models |
| `src/database/repository.py` | Repository abstraction |
| `src/pipeline/conditional_retraining_pipeline.py` | Retraining coordinator |
| `scripts/bootstrap_cloud_uploads.py` | Optional S3 bootstrap |
| `tests/test_stage4_*.py` | Stage 4 tests |

## Limitations

- Retraining orchestration is a foundation, not full GPU fine-tuning automation.
- Production would require model promotion gates and rollback.
- S3 and PostgreSQL credentials must be handled through environment variables.
