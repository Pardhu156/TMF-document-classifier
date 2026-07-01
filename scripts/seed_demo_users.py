"""Idempotently seed development/demo RBAC users."""

from __future__ import annotations

from pathlib import Path
import sys
import time

from sqlalchemy import text
from sqlalchemy.exc import OperationalError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.auth import hash_password
from src.database.db_connection import create_db_engine, create_tables
from src.database.repository import TMFRepository
from src.logger import logger


DEMO_USERS = [
    {"name": "Demo User", "email": "user@test.com", "password": "user123", "role": "User"},
    {"name": "Demo Manager", "email": "manager@test.com", "password": "manager123", "role": "Manager"},
    {"name": "Demo Admin", "email": "admin@test.com", "password": "admin123", "role": "Admin"},
]


def _wait_for_postgres(timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            engine = create_db_engine()
            with engine.begin() as connection:
                connection.execute(text("SELECT 1"))
            return
        except OperationalError as error:
            last_error = error
            logger.info("Waiting for PostgreSQL before seeding demo users...")
            time.sleep(2)
    raise TimeoutError(f"PostgreSQL was not ready after {timeout_seconds}s: {last_error}")


def seed_demo_users(timeout_seconds: int = 90) -> list[dict]:
    """Create or update demo users without creating duplicates."""
    _wait_for_postgres(timeout_seconds)
    engine = create_db_engine()
    create_tables(engine)
    repository = TMFRepository()
    seeded_users = []
    for demo_user in DEMO_USERS:
        user = repository.upsert_user(
            {
                "name": demo_user["name"],
                "email": demo_user["email"],
                "hashed_password": hash_password(demo_user["password"]),
                "role": demo_user["role"],
                "is_active": True,
            }
        )
        seeded_users.append({"email": user["email"], "role": user["role"], "is_active": user["is_active"]})
    return seeded_users


def main() -> None:
    users = seed_demo_users()
    print({"status": "seeded", "users": users})


if __name__ == "__main__":
    main()
