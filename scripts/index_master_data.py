"""CLI entrypoint for manually indexing MASTER_DATA into pgvector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.master_data_ingestion import MasterDataIngestionPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Index trusted MASTER_DATA files into PostgreSQL pgvector.")
    parser.add_argument(
        "--master-data-dir",
        type=Path,
        default=None,
        help="Optional path override. Defaults to MASTER_DATA_DIR from .env or ./MASTER_DATA.",
    )
    args = parser.parse_args()

    result = MasterDataIngestionPipeline().run(master_data_dir=args.master_data_dir)
    print(result)


if __name__ == "__main__":
    main()
