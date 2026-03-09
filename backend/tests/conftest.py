import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
import app.core.deps as deps


@pytest.fixture
def async_client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def student_a_token():
    return "token-student-a"


@pytest.fixture
def student_b_token():
    return "token-student-b"


@pytest.fixture(autouse=True)
def mock_auth(monkeypatch):
    async def ok_verify(token: str):
        if token == "token-student-a":
            return {"sub": "user-student-a", "email": "a@example.com", "aal": "aal1"}
        if token == "token-student-b":
            return {"sub": "user-student-b", "email": "b@example.com", "aal": "aal1"}
        if token == "token-admin":
            return {"sub": "user-admin", "email": "admin@example.com", "aal": "aal2"}
        return {"sub": "user-unknown", "email": "u@example.com", "aal": "aal1"}

    async def role_lookup(user_id: str) -> str:
        if user_id == "user-admin":
            return "admin"
        return "student"

    monkeypatch.setattr(deps, "verify_supabase_jwt", ok_verify)
    monkeypatch.setattr(deps, "fetch_user_role_from_db", role_lookup)
