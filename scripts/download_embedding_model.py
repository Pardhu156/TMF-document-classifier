"""Download/cache the local RAG embedding model for Docker or local setup."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    model_id = os.getenv("LOCAL_EMBEDDING_MODEL", "NeuML/pubmedbert-base-embeddings")
    model_dir = Path(os.getenv("LOCAL_MODEL_DIR", "/models/pubmedbert-base-embeddings"))
    model_dir.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    model.save(str(model_dir))
    print(f"Saved local embedding model {model_id} to {model_dir}")


if __name__ == "__main__":
    main()
