from datetime import datetime, timezone

from fastapi.testclient import TestClient

import app as api_app
from src.auth import hash_password
from tests.auth_helpers import auth_headers


client = TestClient(api_app.app)


class AuthRepository:
    def __init__(self) -> None:
        self.users = [
            {
                "id": 1,
                "name": "Demo User",
                "email": "user@test.com",
                "hashed_password": hash_password("user123"),
                "role": "User",
                "is_active": True,
                "created_at": datetime.now(timezone.utc),
            },
            {
                "id": 2,
                "name": "Demo Manager",
                "email": "manager@test.com",
                "hashed_password": hash_password("manager123"),
                "role": "Manager",
                "is_active": True,
                "created_at": datetime.now(timezone.utc),
            },
            {
                "id": 3,
                "name": "Demo Admin",
                "email": "admin@test.com",
                "hashed_password": hash_password("admin123"),
                "role": "Admin",
                "is_active": True,
                "created_at": datetime.now(timezone.utc),
            },
        ]

    def get_user_by_id(self, user_id: int) -> dict | None:
        return next((user for user in self.users if user["id"] == user_id), None)

    def get_user_by_email(self, email: str) -> dict | None:
        return next((user for user in self.users if user["email"] == email.lower()), None)

    def list_users(self) -> list[dict]:
        return self.users

    def list_audit_logs(self, limit: int = 100) -> list[dict]:
        return []

    def list_documents_by_uploader(self, uploaded_by: str) -> list[dict]:
        return [{"doc_id": 10, "filename": "mine.txt", "uploaded_by": uploaded_by, "document_status": "auto_filed"}]

    def list_documents_by_status(self, statuses: list[str]) -> list[dict]:
        return [
            {
                "doc_id": 11,
                "filename": "ready-for-training.txt",
                "uploaded_by": "manager@test.com",
                "document_status": statuses[0],
            }
        ]


def install_repo() -> AuthRepository:
    repo = AuthRepository()
    api_app.app.dependency_overrides[api_app.get_auth_repository] = lambda: repo
    return repo


def test_login_as_each_demo_role_and_me() -> None:
    install_repo()

    for email, password, role, dashboard in [
        ("user@test.com", "user123", "User", "User Dashboard"),
        ("manager@test.com", "manager123", "Manager", "Manager Dashboard"),
        ("admin@test.com", "admin123", "Admin", "Admin Dashboard"),
    ]:
        response = client.post("/auth/login", json={"email": email, "password": password})

        assert response.status_code == 200
        payload = response.json()
        assert payload["user"]["role"] == role
        assert payload["dashboard"] == dashboard
        assert "hashed_password" not in payload["user"]

        me_response = client.get("/auth/me", headers={"Authorization": f"Bearer {payload['access_token']}"})
        assert me_response.status_code == 200
        assert me_response.json()["email"] == email


def test_invalid_login_fails() -> None:
    install_repo()

    response = client.post("/auth/login", json={"email": "user@test.com", "password": "wrong"})

    assert response.status_code == 401


def test_user_cannot_access_manager_or_admin_apis() -> None:
    install_repo()

    manager_response = client.get("/agentic/reviews", headers=auth_headers("User"))
    admin_response = client.get("/users", headers=auth_headers("User"))
    own_uploads_response = client.get("/documents/my-uploads", headers=auth_headers("User"))

    assert manager_response.status_code == 403
    assert admin_response.status_code == 403
    assert own_uploads_response.status_code == 200
    assert own_uploads_response.json()["items"][0]["uploaded_by"] == "user@test.com"


def test_manager_cannot_access_admin_apis() -> None:
    install_repo()

    response = client.get("/users", headers=auth_headers("Manager"))

    assert response.status_code == 403


def test_admin_has_full_rbac_access_to_admin_api() -> None:
    install_repo()

    users_response = client.get("/users", headers=auth_headers("Admin"))
    audit_response = client.get("/audit-logs", headers=auth_headers("Admin"))

    assert users_response.status_code == 200
    assert audit_response.status_code == 200


def test_training_approval_queue_is_admin_only() -> None:
    install_repo()

    user_response = client.get("/agentic/training/pending", headers=auth_headers("User"))
    manager_response = client.get("/agentic/training/pending", headers=auth_headers("Manager"))
    admin_response = client.get("/agentic/training/pending", headers=auth_headers("Admin"))

    assert user_response.status_code == 403
    assert manager_response.status_code == 403
    assert admin_response.status_code == 200
    assert admin_response.json()["items"][0]["document_status"] == "pending_training_approval"
