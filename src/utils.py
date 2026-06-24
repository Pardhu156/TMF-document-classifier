"""Small, dependency-light utilities shared by pipeline stages."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
from typing import Iterable, Sequence


def majority_vote(labels: Iterable[str], class_order: Sequence[str]) -> str:
    """Return a deterministic majority label, resolving ties by class order."""
    counts = Counter(labels)
    if not counts:
        raise ValueError("Cannot select a majority label from an empty collection.")
    highest_count = max(counts.values())
    for label in class_order:
        if counts.get(label) == highest_count:
            return label
    return sorted(label for label, count in counts.items() if count == highest_count)[0]


def file_sha256(file_path: Path) -> str:
    """Return a SHA-256 digest for a file without loading it all into memory."""
    digest = sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    """Return installed versions for reproducibility metadata."""
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (including parents) when it does not yet exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict, path: Path | str) -> Path:
    """Persist JSON with stable formatting and a safely created parent directory."""
    destination = Path(path)
    ensure_dir(destination.parent)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return destination


def load_json(path: Path | str) -> dict:
    """Load a JSON object from disk."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def get_current_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def get_git_commit_hash() -> str | None:
    """Return the current commit hash, or None outside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def flatten_dict(data: dict, parent_key: str = "") -> dict[str, object]:
    """Flatten nested dictionaries using underscores between key components."""
    flattened: dict[str, object] = {}
    for key, value in data.items():
        combined_key = f"{parent_key}_{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, combined_key))
        else:
            flattened[combined_key] = value
    return flattened
