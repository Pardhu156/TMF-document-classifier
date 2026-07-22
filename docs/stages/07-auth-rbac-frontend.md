# Stage 7: Authentication, RBAC, RAG Access Control, And Frontend

## Tech Stack

- JWT bearer tokens
- password hashing
- PostgreSQL users table
- FastAPI dependencies
- RBAC route guards
- RAG metadata filters
- static role-based frontend console

## Key Steps

1. Added email/password login.
2. Added secure password hashing.
3. Added JWT access tokens.
4. Added `/auth/me` and logout endpoint.
5. Added PostgreSQL-backed demo users.
6. Added User, Manager, and Admin roles.
7. Protected backend endpoints with role dependencies.
8. Added document-level RAG access metadata.
9. Applied RBAC metadata filters before retrieval.
10. Built a role-based enterprise frontend console.

## Roles

| Role | Main Permissions |
| --- | --- |
| User | Upload/classify, view own documents, ask RAG |
| Manager | User features plus approval queue, class correction, team docs, team analytics |
| Admin | Manager features plus users, metrics, audit logs, retraining, re-indexing, system health |

## Demo Users

| Email | Password | Role |
| --- | --- | --- |
| `user@test.com` | `user123` | User |
| `manager@test.com` | `manager123` | Manager |
| `admin@test.com` | `admin123` | Admin |

## RAG Access Rules

| Role | Authorized `access_level` Values |
| --- | --- |
| User | `User` |
| Manager | `User`, `Manager` |
| Admin | `User`, `Manager`, `Admin` |

If authorized retrieval returns no usable documents, the service returns:

```text
You do not have permission to access documents relevant to this query.
```

## Frontend Areas

User:

- Upload & Classify
- My Documents
- AI Document Assistant
- Classification History
- Query History

Manager:

- Approval Queue
- Manual Classification Review
- Team Documents
- Team Analytics
- Manager AI Assistant

Admin:

- User Management
- Model Management
- Training Approval Queue
- Redis Cache Monitor
- Embedding & Vector Index
- RAG Pipeline
- Audit Logs
- System Health

## Files

| Path | Purpose |
| --- | --- |
| `src/auth.py` | Password hashing, JWT, auth dependencies |
| `src/database/models.py` | User and metadata schema |
| `scripts/seed_demo_users.py` | Idempotent demo users |
| `src/rag/service.py` | RBAC-aware RAG orchestration |
| `src/rag/retriever.py` | Metadata filters before retrieval |
| `frontend/` | Role-based static frontend console |
| `tests/test_auth_rbac.py` | Auth/RBAC tests |
| `tests/test_rag_service.py` | RAG RBAC tests |

## Limitations

- Demo users are not production identity.
- Logout is stateless and client-side token clearing.
- Production should use SSO, MFA, refresh token policy, password reset, and audit-grade session controls.
