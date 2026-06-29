"""SHA256 hashing helpers for documents and chunks."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def calculate_file_hash(file_path: Path | str) -> str:
    """Calculate a SHA256 hash for a file without loading it all into memory."""
    digest = sha256()
    with Path(file_path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calculate_text_hash(text: str) -> str:
    """Calculate a SHA256 hash for text after UTF-8 encoding."""
    return sha256(text.encode("utf-8")).hexdigest()
