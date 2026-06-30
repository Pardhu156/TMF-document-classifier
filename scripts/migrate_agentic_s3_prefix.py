"""Copy old Stage 6 S3 prefix `cloud/` to `agentic_tmf_workspace/`.

Safe defaults:
- dry-run by default
- does not delete old objects unless --delete-old true is provided

Usage:
    python scripts/migrate_agentic_s3_prefix.py --dry-run false
    python scripts/migrate_agentic_s3_prefix.py --dry-run false --delete-old true
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cloud.s3_manager import S3Manager
from src.config import CloudConfig
from src.logger import logger


OLD_PREFIX = "cloud/"
NEW_PREFIX = "agentic_tmf_workspace/"


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Stage 6 S3 objects from cloud/ to agentic_tmf_workspace/.")
    parser.add_argument("--dry-run", default="true", help="true/false. Print actions without copying/deleting.")
    parser.add_argument("--delete-old", default="false", help="true/false. Delete old cloud/ objects after copy.")
    args = parser.parse_args()

    dry_run = parse_bool(args.dry_run)
    delete_old = parse_bool(args.delete_old)
    manager = S3Manager(CloudConfig())
    keys = manager.list_keys(OLD_PREFIX)

    print(f"Found {len(keys)} objects under s3://{manager.config.aws_s3_bucket_name}/{OLD_PREFIX}")
    copied = 0
    deleted = 0

    for source_key in keys:
        destination_key = NEW_PREFIX + source_key[len(OLD_PREFIX) :]
        print(f"COPY {source_key} -> {destination_key}")
        if not dry_run:
            manager.copy_object(source_key, destination_key)
            copied += 1
            if delete_old:
                print(f"DELETE {source_key}")
                manager.delete_object(source_key)
                deleted += 1

    summary = {
        "dry_run": dry_run,
        "delete_old": delete_old,
        "found": len(keys),
        "copied": copied,
        "deleted": deleted,
        "old_prefix": OLD_PREFIX,
        "new_prefix": NEW_PREFIX,
    }
    logger.info("Agentic S3 prefix migration summary: %s", summary)
    print(summary)


if __name__ == "__main__":
    main()
