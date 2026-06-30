# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV EMBEDDING_PROVIDER=local
ENV LOCAL_EMBEDDING_MODEL=NeuML/pubmedbert-base-embeddings
ENV LOCAL_MODEL_DIR=/models/pubmedbert-base-embeddings
ENV LOCAL_EMBEDDING_DEVICE=cpu
ENV LOCAL_EMBEDDING_BATCH_SIZE=8
ENV RAG_EMBEDDING_DIMENSION=768

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2+cpu

COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-api.txt

COPY scripts/download_embedding_model.py scripts/download_embedding_model.py
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python scripts/download_embedding_model.py

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
