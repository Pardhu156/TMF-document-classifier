"""Role-to-document access helpers for RAG metadata filtering."""

from __future__ import annotations

from typing import Any


ACCESS_USER = "User"
ACCESS_MANAGER = "Manager"
ACCESS_ADMIN = "Admin"
ACCESS_LEVELS = {ACCESS_USER, ACCESS_MANAGER, ACCESS_ADMIN}
ACCESS_PERMISSION_MESSAGE = "You do not have permission to access documents relevant to this query."


def normalize_access_level(access_level: str | None) -> str:
    """Return a supported access level, defaulting old metadata to User."""
    if not access_level:
        return ACCESS_USER
    normalized = access_level.strip().capitalize()
    return normalized if normalized in ACCESS_LEVELS else ACCESS_USER


def allowed_access_levels_for_role(role: str | None) -> list[str]:
    """Map a user role to the document access levels that may be retrieved."""
    normalized = normalize_access_level(role)
    if normalized == ACCESS_ADMIN:
        return [ACCESS_USER, ACCESS_MANAGER, ACCESS_ADMIN]
    if normalized == ACCESS_MANAGER:
        return [ACCESS_USER, ACCESS_MANAGER]
    return [ACCESS_USER]


def apply_access_filters(filters: dict[str, Any] | None, role: str | None) -> dict[str, Any]:
    """Add access-level constraints to an existing metadata filter dict."""
    return {**(filters or {}), "access_level": allowed_access_levels_for_role(role)}


def demo_access_level_for_index(index: int) -> str:
    """Assign deterministic demo access levels while preserving document class metadata."""
    mod = index % 7
    if mod == 0:
        return ACCESS_ADMIN
    if mod in {1, 2}:
        return ACCESS_MANAGER
    return ACCESS_USER
