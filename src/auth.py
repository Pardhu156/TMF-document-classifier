"""Authentication, password hashing, and RBAC helpers for the API."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from typing import Annotated, Iterable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import AuthConfig
from src.database.repository import TMFRepository


ROLE_USER = "User"
ROLE_MANAGER = "Manager"
ROLE_ADMIN = "Admin"
VALID_ROLES = {ROLE_USER, ROLE_MANAGER, ROLE_ADMIN}
ROLE_RANK = {ROLE_USER: 1, ROLE_MANAGER: 2, ROLE_ADMIN: 3}

PASSWORD_HASH_ITERATIONS = 210_000
PASSWORD_HASH_NAME = "sha256"
bearer_scheme = HTTPBearer(auto_error=False)


def normalize_role(role: str) -> str:
    """Return the canonical role name or raise for unsupported roles."""
    normalized = role.strip().capitalize()
    if normalized not in VALID_ROLES:
        raise ValueError(f"role must be one of: {', '.join(sorted(VALID_ROLES))}")
    return normalized


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a per-password salt."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        PASSWORD_HASH_NAME,
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return (
        f"pbkdf2_{PASSWORD_HASH_NAME}${PASSWORD_HASH_ITERATIONS}$"
        f"{base64.urlsafe_b64encode(salt).decode()}$"
        f"{base64.urlsafe_b64encode(digest).decode()}"
    )


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against this project's PBKDF2 hash format."""
    try:
        algorithm, iterations, salt_b64, digest_b64 = hashed_password.split("$", 3)
        if algorithm != f"pbkdf2_{PASSWORD_HASH_NAME}":
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(digest_b64.encode())
        actual = hashlib.pbkdf2_hmac(PASSWORD_HASH_NAME, password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def create_access_token(user: dict, config: AuthConfig | None = None) -> str:
    """Create a signed JWT access token for a user."""
    config = config or AuthConfig()
    if config.jwt_algorithm != "HS256":
        raise ValueError("Only HS256 is currently supported.")
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user["id"]),
        "email": user["email"],
        "role": user["role"],
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=config.access_token_expire_minutes)).timestamp()),
    }
    header = {"alg": config.jwt_algorithm, "typ": "JWT"}
    signing_input = ".".join(
        (
            _b64url_encode(json.dumps(header, separators=(",", ":")).encode()),
            _b64url_encode(json.dumps(payload, separators=(",", ":")).encode()),
        )
    )
    signature = hmac.new(config.jwt_secret_key.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def decode_access_token(token: str, config: AuthConfig | None = None) -> dict:
    """Decode and verify a signed JWT access token."""
    config = config or AuthConfig()
    try:
        header_b64, payload_b64, signature_b64 = token.split(".", 2)
        signing_input = f"{header_b64}.{payload_b64}"
        expected_signature = hmac.new(
            config.jwt_secret_key.encode("utf-8"),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        actual_signature = _b64url_decode(signature_b64)
        if not hmac.compare_digest(actual_signature, expected_signature):
            raise ValueError("invalid signature")
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != config.jwt_algorithm:
            raise ValueError("invalid algorithm")
        payload = json.loads(_b64url_decode(payload_b64))
        if int(payload.get("exp", 0)) < int(datetime.now(timezone.utc).timestamp()):
            raise ValueError("token expired")
        return payload
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error


def public_user(user: dict) -> dict:
    """Serialize a user without exposing password hashes."""
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "is_active": user["is_active"],
        "created_at": user["created_at"],
    }


def get_auth_repository() -> TMFRepository:
    try:
        return TMFRepository()
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication database is not configured.",
        ) from error


def authenticate_user(email: str, password: str, repository: TMFRepository) -> dict | None:
    """Return the active user when credentials are valid."""
    user = repository.get_user_by_email(email.strip().lower())
    if not user or not user["is_active"]:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    repository: Annotated[TMFRepository, Depends(get_auth_repository)],
) -> dict:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    user = repository.get_user_by_id(int(payload["sub"]))
    if not user or not user["is_active"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User is inactive or no longer exists.")
    return user


def require_roles(roles: Iterable[str]):
    """FastAPI dependency factory enforcing role membership."""
    allowed_roles = {normalize_role(role) for role in roles}

    def dependency(current_user: Annotated[dict, Depends(get_current_user)]) -> dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role permissions.")
        return current_user

    return dependency


def require_min_role(role: str):
    """FastAPI dependency factory enforcing a minimum role rank."""
    minimum_role = normalize_role(role)

    def dependency(current_user: Annotated[dict, Depends(get_current_user)]) -> dict:
        if ROLE_RANK[current_user["role"]] < ROLE_RANK[minimum_role]:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role permissions.")
        return current_user

    return dependency
