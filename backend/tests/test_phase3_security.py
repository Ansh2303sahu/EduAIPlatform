import pytest
from urllib.parse import urlparse


# IMPORTANT: This must match your router module import path
import app.api.routes as routes


class DummyResponse:
    def __init__(self, status_code: int, data=None, text: str = ""):
        self.status_code = status_code
        self._data = data
        self.text = text or ""

    def json(self):
        return self._data


class DummySupabase:
    """
    In-memory Supabase emulator for the specific endpoints used by Phase 3 routes:
    - POST /rest/v1/submissions
    - GET  /rest/v1/submissions?id=eq.<id>&select=*
    - POST /rest/v1/files
    - GET  /rest/v1/files?id=eq.<id>&select=*
    - POST /rest/v1/audit_logs (ignored)
    - POST /storage/v1/object/sign/{bucket}/{path} -> signedURL
    """

    def __init__(self):
        self.submissions = {}  # id -> row
        self.files = {}        # id -> row
        self._sub_counter = 1
        self._file_counter = 1

    def _new_id(self, prefix: str, counter: int) -> str:
        return f"{prefix}-{counter:04d}"

    def handle_post(self, path: str, json_body: dict | None):
        json_body = json_body or {}

        # submissions insert
        if path.endswith("/rest/v1/submissions"):
            sub_id = self._new_id("sub", self._sub_counter)
            self._sub_counter += 1

            row = {
                "id": sub_id,
                "user_id": json_body["user_id"],
                "content": json_body["content"],
                "created_at": "2026-01-01T00:00:00Z",
            }
            self.submissions[sub_id] = row
            return DummyResponse(201, [row])

        # files insert
        if path.endswith("/rest/v1/files"):
            file_id = self._new_id("file", self._file_counter)
            self._file_counter += 1

            row = {
                "id": file_id,
                "user_id": json_body["user_id"],
                "bucket": json_body["bucket"],
                "object_path": json_body["object_path"],
                "mime_type": json_body.get("mime_type"),
                "size_bytes": json_body.get("size_bytes"),
                "created_at": "2026-01-01T00:00:00Z",
            }
            self.files[file_id] = row
            return DummyResponse(201, [row])

        # audit logs insert (ignore)
        if path.endswith("/rest/v1/audit_logs"):
            return DummyResponse(201, [{"ok": True}])

        # storage signed url
        if "/storage/v1/object/sign/" in path:
            return DummyResponse(200, {"signedURL": "https://signed.example.com/fake"})

        return DummyResponse(404, {"error": "not found"}, text="not found")

    def handle_get(self, path: str):
        # submissions select by id
        if "/rest/v1/submissions" in path:
            if "id=eq." in path:
                sub_id = path.split("id=eq.", 1)[1].split("&", 1)[0]
                row = self.submissions.get(sub_id)
                return DummyResponse(200, [row] if row else [])
            return DummyResponse(200, list(self.submissions.values()))

        # files select by id
        if "/rest/v1/files" in path:
            if "id=eq." in path:
                file_id = path.split("id=eq.", 1)[1].split("&", 1)[0]
                row = self.files.get(file_id)
                return DummyResponse(200, [row] if row else [])
            return DummyResponse(200, list(self.files.values()))

        return DummyResponse(404, {"error": "not found"}, text="not found")


class DummyAsyncClient:
    """
    Drop-in replacement for httpx.AsyncClient used inside app.api.routes.
    """

    def __init__(self, supabase: DummySupabase, *args, **kwargs):
        self.supabase = supabase

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url: str, headers=None, json=None):
        parsed = urlparse(url)
        path = parsed.path  # POST endpoints don't depend on query string here
        return self.supabase.handle_post(path, json)

    async def get(self, url: str, headers=None):
        parsed = urlparse(url)
        path = parsed.path
        if parsed.query:
            path = f"{path}?{parsed.query}"  # keep query string for id=eq parsing
        return self.supabase.handle_get(path)

    async def patch(self, url: str, headers=None, json=None):
        return DummyResponse(404, {"error": "not implemented"}, text="not implemented")


@pytest.fixture(autouse=True)
def mock_supabase_http(monkeypatch):
    """
    Auto-mock httpx.AsyncClient inside app.api.routes so no real network happens.
    """
    supabase = DummySupabase()

    def factory(*args, **kwargs):
        return DummyAsyncClient(supabase, *args, **kwargs)

    monkeypatch.setattr(routes.httpx, "AsyncClient", factory)
    return supabase


@pytest.mark.asyncio
async def test_cross_user_submission_blocked(async_client, student_a_token, student_b_token):
    async with async_client as ac:
        # A creates
        r = await ac.post(
            "/api/submissions",
            headers={"Authorization": f"Bearer {student_a_token}"},
            json={"content": "A secret submission"},
        )
        assert r.status_code == 200
        sub_id = r.json()["id"]

        # B cannot read (404 preferred)
        r2 = await ac.get(
            f"/api/submissions/{sub_id}",
            headers={"Authorization": f"Bearer {student_b_token}"},
        )
        assert r2.status_code == 404


@pytest.mark.asyncio
async def test_cross_user_signed_url_blocked(async_client, student_a_token, student_b_token):
    async with async_client as ac:
        # A registers file metadata
        r = await ac.post(
            "/api/files/register",
            headers={"Authorization": f"Bearer {student_a_token}"},
            json={
                "bucket": "edu-uploads",
                "object_path": "fakepath/a.txt",
                "mime_type": "text/plain",
                "size_bytes": 1,
            },
        )
        assert r.status_code == 200
        file_id = r.json()["id"]

        # B cannot get signed url
        r2 = await ac.post(
            f"/api/files/{file_id}/signed-url",
            headers={"Authorization": f"Bearer {student_b_token}"},
        )
        assert r2.status_code == 404
