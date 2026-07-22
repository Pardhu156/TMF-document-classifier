# How To Start And Test Locally

This guide starts the TMF Document Classifier POC locally without changing the application code.

## Prerequisites

Tech stack used locally:

- Python virtual environment
- FastAPI with Uvicorn
- PostgreSQL with pgvector
- Redis
- Docker Compose
- Optional DVC for restoring datasets/artifacts
- Optional AWS S3 credentials for cloud document persistence

## Option A: Fast Local Development

Use this when you want to run FastAPI in VS Code and only start PostgreSQL/Redis through Docker.

1. Start Docker Desktop.

2. From the project root, start PostgreSQL and Redis:

```bash
docker compose up postgres redis
```

3. In a second terminal, activate the virtual environment:

```bash
source venv/bin/activate
```

4. Copy the environment template if needed:

```bash
cp .env.example .env
```

5. Make sure local database/cache values are set:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5434
POSTGRES_DB=tmf_classifier
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
REDIS_URL=redis://localhost:6379/0
```

6. Seed demo users:

```bash
python scripts/seed_demo_users.py
```

7. Start FastAPI:

```bash
uvicorn app:app --reload
```

8. Open the frontend console:

```text
http://127.0.0.1:8000/console
```

9. Open Swagger:

```text
http://127.0.0.1:8000/docs
```

## Option B: Full Docker Compose

Use this when you want Docker Compose to run the app, PostgreSQL, Redis, and the seed job.

1. Start Docker Desktop.

2. Restore DVC data if your local data/model artifacts are missing:

```bash
dvc pull
```

3. Copy and edit `.env`:

```bash
cp .env.example .env
```

4. Start everything:

```bash
docker compose up --build
```

5. Open:

```text
http://127.0.0.1:8000/console
```

The `db-seeder` service runs:

```text
python scripts/seed_demo_users.py
python scripts/seed_rag_database.py --master-data-dir /app/MASTER_DATA
```

It is idempotent. Running it multiple times does not duplicate users or indexed RAG chunks.

## Demo Login Accounts

| Role | Email | Password | Expected Redirect |
| --- | --- | --- | --- |
| User | `user@test.com` | `user123` | `/console#/user/dashboard` |
| Manager | `manager@test.com` | `manager123` | `/console#/manager/home` |
| Admin | `admin@test.com` | `admin123` | `/console#/admin/dashboard` |

## Basic API Tests

Login:

```bash
curl -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@test.com", "password": "admin123"}'
```

Set token:

```bash
TOKEN=$(curl -s -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@test.com", "password": "admin123"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

Check current user:

```bash
curl "http://127.0.0.1:8000/auth/me" \
  -H "Authorization: Bearer ${TOKEN}"
```

Run text classification:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"text": "The protocol describes study objectives, eligibility criteria, endpoints, and visit schedule."}'
```

Run file upload/classification:

```bash
curl -X POST "http://127.0.0.1:8000/predict-file" \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@/path/to/document.pdf"
```

Ask RAG across all authorized documents:

```bash
curl -X POST "http://127.0.0.1:8000/rag/ask" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the adverse event reporting requirements?", "scope": "all"}'
```

Ask RAG from one selected document:

```bash
curl -X POST "http://127.0.0.1:8000/rag/ask" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the study objective?", "document_id": "1"}'
```

## Role Access Checks

User should fail on manager/admin endpoints:

```bash
USER_TOKEN=$(curl -s -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@test.com", "password": "user123"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -i "http://127.0.0.1:8000/agentic/reviews" \
  -H "Authorization: Bearer ${USER_TOKEN}"
```

Expected: `403 Forbidden`.

Manager should fail on admin endpoints:

```bash
MANAGER_TOKEN=$(curl -s -X POST "http://127.0.0.1:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "manager@test.com", "password": "manager123"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -i "http://127.0.0.1:8000/users" \
  -H "Authorization: Bearer ${MANAGER_TOKEN}"
```

Expected: `403 Forbidden`.

Admin should access admin metrics:

```bash
curl "http://127.0.0.1:8000/agentic/metrics" \
  -H "Authorization: Bearer ${TOKEN}"
```

## UI Smoke Test

1. Open `http://127.0.0.1:8000/console`.
2. Login as User.
3. Confirm User navigation only shows Upload & Classify, My Documents, AI Document Assistant, Classification History, Query History, and Profile-style user pages.
4. Logout.
5. Login as Manager.
6. Confirm Manager navigation includes Approval Queue, class correction, Team Documents, Team Analytics, and Manager AI Assistant.
7. Try manually opening `/console#/admin/dashboard`.
8. Confirm Unauthorized page appears.
9. Logout.
10. Login as Admin.
11. Confirm Admin can see User Management, Model Management, Redis Cache Monitor, Vector Index, RAG Pipeline, Audit Logs, and System Health.

## Common Troubleshooting

Docker daemon error:

```text
Cannot connect to the Docker daemon
```

Fix: start Docker Desktop, then rerun `docker compose up postgres redis`.

Seeder waits for PostgreSQL forever:

- Confirm PostgreSQL container is healthy.
- Confirm `.env` uses `POSTGRES_PORT=5434` for local host access.
- If port `5434` is busy, change `POSTGRES_HOST_PORT`.

Redis port conflict:

- Change `REDIS_HOST_PORT` in `.env`.

RAG returns no documents:

- Confirm `MASTER_DATA/` exists.
- Run `dvc pull` if files are DVC-tracked.
- Check seeder logs:

```bash
docker compose logs db-seeder
```

Frontend says non-JSON API response:

- Confirm backend URL is correct.
- Locally, use `http://127.0.0.1:8000/console`, not only a static file.

## Shut Down

Stop services:

```bash
docker compose down
```

Delete local PostgreSQL volume only when you want to reset indexed data:

```bash
docker compose down -v
```
