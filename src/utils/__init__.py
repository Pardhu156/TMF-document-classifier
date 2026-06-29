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


def decision_status_from_confidence(final_confidence: float, margin_confidence: float) -> str:
    """Return a passive decision label for future review workflows.

    This field is metadata only. It does not trigger any agentic or automated
    action in the current application.
    """
    if final_confidence >= 0.80 and margin_confidence >= 0.20:
        return "auto_classify"
    if final_confidence >= 0.60 and margin_confidence >= 0.15:
        return "agent_review"
    return "human_review"


def document_confidence_summary(
    chunk_results: Sequence[dict[str, object]],
    class_order: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compute scalable document-level confidence from chunk predictions."""
    if not chunk_results:
        raise ValueError("Cannot compute document confidence without chunk predictions.")

    labels = [str(result["predicted_label"]) for result in chunk_results]
    ordered_classes = list(class_order or [])
    for label in labels:
        if label not in ordered_classes:
            ordered_classes.append(label)

    predicted_label = majority_vote(labels, ordered_classes)
    label_counts = Counter(labels)
    total_chunks = len(chunk_results)
    winning_votes = int(label_counts[predicted_label])
    sorted_vote_counts = sorted(label_counts.values(), reverse=True)
    second_best_votes = int(sorted_vote_counts[1]) if len(sorted_vote_counts) > 1 else 0

    winning_confidences = [
        float(result["confidence"])
        for result in chunk_results
        if str(result["predicted_label"]) == predicted_label
    ]
    model_confidence = sum(winning_confidences) / len(winning_confidences)
    vote_confidence = winning_votes / total_chunks
    margin_confidence = (winning_votes - second_best_votes) / total_chunks
    final_confidence = min(model_confidence * vote_confidence * (1 + margin_confidence), 1.0)
    decision_status = decision_status_from_confidence(final_confidence, margin_confidence)

    return {
        "predicted_label": predicted_label,
        "confidence": float(final_confidence),
        "model_confidence": float(model_confidence),
        "vote_confidence": float(vote_confidence),
        "margin_confidence": float(margin_confidence),
        "requires_review": bool(final_confidence < 0.80),
        "decision_status": decision_status,
        "num_chunks": int(total_chunks),
        "winning_votes": winning_votes,
        "second_best_votes": second_best_votes,
        "chunk_predictions": dict(sorted(label_counts.items())),
    }


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
            cwd=Path(__file__).resolve().parents[2],
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
