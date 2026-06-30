"""Embedding clients for RAG.

Default production mode uses a local PubMedBERT sentence-transformers model.
Gemini embeddings remain available only as a compatibility fallback when
EMBEDDING_PROVIDER=gemini.
"""

from __future__ import annotations

from pathlib import Path

from src.config import CloudConfig, RAGConfig
from src.logger import logger


class RAGEmbeddingClient:
    """Embedding client with local PubMedBERT default and Gemini fallback."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig()
        self._local_model = None

    def embed_text(self, text: str) -> list[float]:
        if self.config.uses_local_embeddings:
            return self._embed_local(text)
        return self._embed_gemini(text, task_type="retrieval_document")

    def embed_query(self, question: str) -> list[float]:
        if self.config.uses_local_embeddings:
            return self._embed_local(question)
        return self._embed_gemini(question, task_type="retrieval_query")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.config.uses_local_embeddings:
            model = self._load_local_model()
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.local_embedding_batch_size,
            )
            return [[float(value) for value in embedding] for embedding in embeddings]
        return [self.embed_text(text) for text in texts]

    def _embed_local(self, text: str) -> list[float]:
        model = self._load_local_model()
        embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return [float(value) for value in embedding]

    def _load_local_model(self):
        if self._local_model is not None:
            return self._local_model
        model_path = self._resolve_local_model_path()
        from sentence_transformers import SentenceTransformer

        self._local_model = SentenceTransformer(str(model_path), device=self.config.local_embedding_device)
        logger.info(
            "Loaded local RAG embedding model from %s on device=%s with batch_size=%d",
            model_path,
            self.config.local_embedding_device,
            self.config.local_embedding_batch_size,
        )
        return self._local_model

    def _resolve_local_model_path(self) -> Path:
        model_dir = Path(self.config.local_model_dir)
        if model_dir.exists() and any(model_dir.iterdir()):
            return model_dir
        self._download_model_backup_from_s3(model_dir)
        if model_dir.exists() and any(model_dir.iterdir()):
            return model_dir
        logger.warning(
            "Local embedding model directory %s is missing; falling back to Hugging Face model id %s. "
            "For production Docker images, package the model at LOCAL_MODEL_DIR.",
            model_dir,
            self.config.local_embedding_model,
        )
        return Path(self.config.local_embedding_model)

    def _download_model_backup_from_s3(self, model_dir: Path) -> None:
        bucket = self.config.model_backup_s3_bucket
        if not bucket:
            return
        try:
            import boto3

            cloud_config = CloudConfig(aws_s3_bucket_name=bucket)
            client = boto3.client(
                "s3",
                region_name=cloud_config.aws_region,
                aws_access_key_id=cloud_config.aws_access_key_id,
                aws_secret_access_key=cloud_config.aws_secret_access_key,
            )
            prefix = self.config.model_backup_s3_prefix.rstrip("/") + "/"
            paginator = client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            downloaded = 0
            for page in pages:
                for item in page.get("Contents", []):
                    key = item["Key"]
                    if key.endswith("/"):
                        continue
                    relative_path = key[len(prefix) :]
                    if not relative_path:
                        continue
                    destination = model_dir / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    client.download_file(bucket, key, str(destination))
                    downloaded += 1
            if downloaded:
                logger.info("Downloaded %d embedding model backup files from s3://%s/%s", downloaded, bucket, prefix)
        except Exception as error:
            logger.warning("Could not download embedding model backup from S3; continuing with local/HF fallback: %s", error)

    def _embed_gemini(self, text: str, task_type: str) -> list[float]:
        if not self.config.gemini_configured:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")
        import google.generativeai as genai

        genai.configure(api_key=self.config.gemini_api_key)
        response = genai.embed_content(
            model=self.config.gemini_embedding_model,
            content=text,
            task_type=task_type,
        )
        embedding = response.get("embedding")
        if not embedding:
            raise RuntimeError("Gemini embedding response did not include an embedding.")
        return [float(value) for value in embedding]


# Backward-compatible import name used by the existing RAG modules.
GeminiEmbeddingClient = RAGEmbeddingClient
