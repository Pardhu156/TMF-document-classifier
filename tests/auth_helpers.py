from datetime import datetime, timezone

import app as api_app
from src.auth import create_access_token, hash_password


def test_user(role: str = "User") -> dict:
    return {
        "id": {"User": 1, "Manager": 2, "Admin": 3}[role],
        "name": f"Test {role}",
        "email": f"{role.lower()}@test.com",
        "hashed_password": hash_password("password123"),
        "role": role,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }


class FakeAuthRepository:
    def __init__(self, users: list[dict] | None = None) -> None:
        self.users = users or [test_user("User"), test_user("Manager"), test_user("Admin")]

    def get_user_by_id(self, user_id: int) -> dict | None:
        return next((user for user in self.users if user["id"] == user_id), None)

    def get_user_by_email(self, email: str) -> dict | None:
        return next((user for user in self.users if user["email"] == email.lower()), None)

    def list_documents_by_uploader(self, uploaded_by: str) -> list[dict]:
        return []


def auth_headers(role: str = "User") -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(test_user(role))}"}


def install_auth_override(repo: FakeAuthRepository | None = None) -> FakeAuthRepository:
    repo = repo or FakeAuthRepository()
    api_app.app.dependency_overrides[api_app.get_auth_repository] = lambda: repo
    return repo
