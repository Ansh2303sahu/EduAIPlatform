import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
import app.core.deps as deps
from jose.exceptions import JWTError


@pytest.mark.asyncio
async def test_unauthenticated_me_401():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get("/api/me")
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_tampered_token_401(monkeypatch):
    async def bad_verify(_token: str):
        raise JWTError("bad token")

    monkeypatch.setattr(deps, "verify_supabase_jwt", bad_verify)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get(
            "/api/me",
            headers={"Authorization": "Bearer tampered"},
        )
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_student_cannot_call_admin_403(monkeypatch):
    async def ok_verify(_token: str):
        return {"sub": "user-student", "email": "s@example.com", "aal": "aal1"}

    async def role_lookup(_user_id: str) -> str:
        return "student"

    monkeypatch.setattr(deps, "verify_supabase_jwt", ok_verify)
    monkeypatch.setattr(deps, "fetch_user_role_from_db", role_lookup)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/api/admin/users/someone/role",
            headers={"Authorization": "Bearer good"},
            json={"role": "professor"},
        )
        assert r.status_code == 403


@pytest.mark.asyncio
async def test_professor_cannot_call_admin_403(monkeypatch):
    async def ok_verify(_token: str):
        return {"sub": "user-prof", "email": "p@example.com", "aal": "aal1"}

    async def role_lookup(_user_id: str) -> str:
        return "professor"

    monkeypatch.setattr(deps, "verify_supabase_jwt", ok_verify)
    monkeypatch.setattr(deps, "fetch_user_role_from_db", role_lookup)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/api/admin/users/someone/role",
            headers={"Authorization": "Bearer good"},
            json={"role": "student"},
        )
        assert r.status_code == 403


@pytest.mark.asyncio
async def test_admin_without_mfa_blocked_403(monkeypatch):
    async def ok_verify(_token: str):
        return {"sub": "user-admin", "email": "a@example.com", "aal": "aal1"}

    async def role_lookup(_user_id: str) -> str:
        return "admin"

    monkeypatch.setattr(deps, "verify_supabase_jwt", ok_verify)
    monkeypatch.setattr(deps, "fetch_user_role_from_db", role_lookup)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/api/admin/users/someone/role",
            headers={"Authorization": "Bearer good"},
            json={"role": "professor"},
        )
        assert r.status_code == 403


@pytest.mark.asyncio
async def test_admin_with_mfa_allowed(monkeypatch):
    async def ok_verify(_token: str):
        return {"sub": "user-admin", "email": "a@example.com", "aal": "aal2"}

    async def role_lookup(_user_id: str) -> str:
        return "admin"

    monkeypatch.setattr(deps, "verify_supabase_jwt", ok_verify)
    monkeypatch.setattr(deps, "fetch_user_role_from_db", role_lookup)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.get(
            "/api/me",
            headers={"Authorization": "Bearer good"},
        )
        assert r.status_code == 200
        assert r.json()["role"] == "admin"

