"""Run manual Stage 4 S3 bootstrap uploads.

Usage:
    python scripts/bootstrap_cloud_uploads.py
    python scripts/bootstrap_cloud_uploads.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.cloud_bootstrap_pipeline import CloudBootstrapPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload existing TMF assets to S3.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Upload files even when the S3 key already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = CloudBootstrapPipeline().run(skip_existing=not args.overwrite)
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
