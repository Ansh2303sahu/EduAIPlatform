from __future__ import annotations

from fastapi import Header, HTTPException, status
from app.core.config import settings


class Roles:
    STUDENT = "student"
    PROFESSOR = "professor"
    ADMIN = "admin"


def require_service_secret(
    x_ai_secret: str | None = Header(default=None, alias="x-ai-secret"),
) -> None:
    if not x_ai_secret or x_ai_secret != settings.ai_service_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid service secret",
        )


def require_role(allowed: set[str]):
    def _dep(
        x_role: str | None = Header(default=None, alias="x-role"),
    ) -> str:
        if not x_role or x_role not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Role not allowed",
            )
        return x_role

    return _dep
