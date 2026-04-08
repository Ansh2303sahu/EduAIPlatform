"""Phase 14 — internal n8n → backend API package.

Routes in this package are mounted under /internal and must NOT be exposed
through the public nginx/Caddy gateway. They are protected by the
X-Internal-Secret header (shared secret, Docker-internal network only).
"""

from .assessment import router as assessment_router

__all__ = ["assessment_router"]
