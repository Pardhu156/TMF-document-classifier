# Stage 6: Agentic TMF Filing

## Tech Stack

- Existing BioClinicalBERT classifier
- FastAPI upload endpoint
- PostgreSQL metadata
- Redis manual review queue
- S3/local workspace
- RAG ingestion hook
- audit logging

## Key Steps

1. Wrapped upload/classification in an agentic filing workflow.
2. Added confidence threshold checks.
3. Added top-1/top-2 confidence gap checks.
4. Auto-filed high-confidence documents.
5. Sent low-confidence documents to Redis manual review queue.
6. Blocked low-confidence documents from final RAG until manager approval.
7. Added manager correction workflow.
8. Added admin training approval/rejection workflow.
9. Added audit logs and metrics.

## Decision Rules

| Rule | Outcome |
| --- | --- |
| confidence >= `AUTO_APPROVAL_THRESHOLD` and gap >= `MIN_CONFIDENCE_GAP` | auto-file |
| confidence below threshold or unsafe gap | manual review |

Defaults:

- `AUTO_APPROVAL_THRESHOLD=0.90`
- `MIN_CONFIDENCE_GAP=0.10`

## Storage Flow

High confidence:

```text
upload -> classify -> auto-file -> pending_training_approval -> RAG indexing
```

Low confidence:

```text
upload -> classify -> pending_review -> Redis queue -> manager correction -> final filing -> RAG indexing
```

## Workspace Layout

```text
agentic_tmf_workspace/
├── tmf/<class>/
├── pending_review/
├── pending_training/<class>/
├── approved_training/
├── rejected_training/
└── metadata/
```

## Files

| Path | Purpose |
| --- | --- |
| `src/agentic_filing/pipeline.py` | Agentic filing workflow |
| `src/agentic_filing/review_queue.py` | Redis review queue |
| `app.py` | Review, correction, approval, and metrics endpoints |
| `tests/test_agentic_filing_pipeline.py` | Workflow tests |

## Limitations

- Human review UI is POC-grade.
- Production would need richer reviewer assignment, SLA tracking, immutable audit workflows, and e-signature compliance.
