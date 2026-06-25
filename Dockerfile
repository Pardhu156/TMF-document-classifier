# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2+cpu

COPY requirements-api.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-api.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
