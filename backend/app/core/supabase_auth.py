from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Any, Dict

import httpx
from fastapi import Depends, Header, HTTPException
from jose import jwt
from jose.exceptions import JWTError

from app.core.config import settings


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: Optional[str]
    role: str
    raw_claims: Dict[str, Any]


class SupabaseJWKSCache:
    def __init__(self, jwks_url: str, ttl_seconds: int = 3600):
        self.jwks_url = jwks_url
        self.ttl_seconds = ttl_seconds
        self._jwks: Optional[Dict[str, Any]] = None
        self._expires_at = 0.0

    async def get(self) -> Dict[str, Any]:
        now = time.time()
        if self._jwks is not None and now < self._expires_at:
            return self._jwks

        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(self.jwks_url)
            r.raise_for_status()
            jwks: Dict[str, Any] = r.json()  # ensure non-None return for type checkers
            self._jwks = jwks
            self._expires_at = now + self.ttl_seconds
            return jwks


_jwks_cache = SupabaseJWKSCache(
    jwks_url=f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json",
    ttl_seconds=3600,
)


def _get_issuer() -> str:
    return f"{settings.supabase_url.rstrip('/')}/auth/v1"


def _get_audience() -> str:
    # Supabase default
    return "authenticated"


async def verify_supabase_jwt(token: str) -> Dict[str, Any]:
    """
    Verifies Supabase JWT signature + standard claims.
    Returns decoded claims dict.
    """
    jwks = await _jwks_cache.get()

    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    if not kid:
        raise JWTError("Missing kid")

    key = None
    for jwk in jwks.get("keys", []):
        if jwk.get("kid") == kid:
            key = jwk
            break
    if not key:
        raise JWTError("No matching JWK")

    claims = jwt.decode(
        token,
        key,
        algorithms=[unverified_header.get("alg", "RS256")],
        audience=_get_audience(),
        issuer=_get_issuer(),
        options={"verify_at_hash": False},
    )
    return claims


async def fetch_user_role_from_db(user_id: str) -> str:
    """
    Looks up role from public.profiles using Supabase REST with SERVICE ROLE.
    """
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/profiles"
    params = {"id": f"eq.{user_id}", "select": "role"}
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        rows = r.json()

    if not rows:
        return "student"

    role = rows[0].get("role") or "student"
    return role if role in {"student", "professor", "admin"} else "student"


async def get_current_user(authorization: str = Header(default=None)) -> CurrentUser:
    """
    FastAPI dependency:
    - Extracts Bearer token
    - Verifies Supabase JWT
    - Loads role from DB (service role key)
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1].strip()

    try:
        claims = await verify_supabase_jwt(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing sub")

    email = claims.get("email")
    role = await fetch_user_role_from_db(user_id)

    return CurrentUser(id=user_id, email=email, role=role, raw_claims=claims)


def require_roles(*allowed: str):
    """
    FastAPI dependency factory:
      Depends(require_roles("admin"))
      Depends(require_roles("professor", "admin"))
    """
    allowed_set = set(allowed)

    async def _checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed_set:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user

    return _checker
