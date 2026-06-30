"""CLI entrypoint for manually indexing local data/ into pgvector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.training_data_ingestion import TrainingDataIngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Index labeled local data/ files into PostgreSQL pgvector.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional path override. Defaults to DataIngestionConfig.raw_data_dir, usually ./data.",
    )
    args = parser.parse_args()

    result = TrainingDataIngestionPipeline().run(data_dir=args.data_dir)
    print(result)


if __name__ == "__main__":
    main()
