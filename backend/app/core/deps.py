from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import httpx
from fastapi import Depends, Header, HTTPException
from jose import jwt, jwk
from jose.exceptions import JWTError

from app.core.config import settings


@dataclass(frozen=True)
class CurrentUser:
    id: str
    email: Optional[str]
    role: str
    raw_claims: Dict[str, Any]
    access_token: Optional[str] = None


# -------------------------
# JWKS cache (Supabase)
# -------------------------

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
            jwks_data: Dict[str, Any] = r.json()

        self._jwks = jwks_data
        self._expires_at = now + self.ttl_seconds
        return jwks_data


def _require_supabase_url() -> None:
    if not settings.supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL is not configured")


def _require_service_role_key() -> None:
    if not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_ROLE_KEY is not configured")


def _issuer() -> str:
    # Supabase JWT issuer
    return f"{settings.supabase_url.rstrip('/')}/auth/v1"


def _allowed_audiences() -> Set[str]:
    # Supabase commonly uses "authenticated" (and sometimes "anon")
    return {"authenticated", "anon"}


def _allowed_algs() -> Set[str]:
    # JWKS verification algorithms typically RS256 or ES256
    return {"RS256", "ES256"}


def _aud_ok(aud_claim: Any) -> bool:
    allowed = _allowed_audiences()

    if aud_claim is None:
        return True

    if isinstance(aud_claim, str):
        return aud_claim in allowed

    if isinstance(aud_claim, (list, tuple, set)):
        return any(isinstance(x, str) and x in allowed for x in aud_claim)

    return False


# Build JWKS cache lazily after settings are loaded
_jwks_cache: Optional[SupabaseJWKSCache] = None


def _get_jwks_cache() -> SupabaseJWKSCache:
    global _jwks_cache
    _require_supabase_url()
    if _jwks_cache is None:
        _jwks_cache = SupabaseJWKSCache(
            jwks_url=f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json",
            ttl_seconds=3600,
        )
    return _jwks_cache


async def verify_supabase_jwt(token: str) -> Dict[str, Any]:
    """
    Verify Supabase JWT signature + issuer, and manually validate audience.
    Returns decoded claims dict.
    """
    _require_supabase_url()

    try:
        unverified_header = jwt.get_unverified_header(token)
    except Exception as e:
        raise JWTError(f"Invalid JWT header: {e}") from e

    kid = unverified_header.get("kid")
    alg = unverified_header.get("alg")

    if not kid:
        raise JWTError("Missing kid in token header")
    if not alg or alg not in _allowed_algs():
        raise JWTError(f"Unsupported token alg: {alg}")

    # Fetch JWKS
    try:
        jwks_data = await _get_jwks_cache().get()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Auth keyset unavailable: {e}") from e

    # Find key by kid
    jwk_dict = None
    for k in jwks_data.get("keys", []):
        if k.get("kid") == kid:
            jwk_dict = k
            break
    if not jwk_dict:
        raise JWTError("No matching JWK for kid")

    # Build PEM public key
    try:
        key_obj = jwk.construct(jwk_dict, algorithm=alg)
        public_pem = key_obj.to_pem().decode("utf-8")
    except Exception as e:
        raise JWTError(f"Failed to construct public key: {e}") from e

    # Decode (issuer verified; audience checked manually)
    try:
        claims = jwt.decode(
            token,
            public_pem,
            algorithms=[alg],
            issuer=_issuer(),
            options={
                "verify_aud": False,      # we validate ourselves
                "verify_at_hash": False,
            },
        )
    except JWTError:
        raise
    except Exception as e:
        raise JWTError(f"JWT decode failed: {e}") from e

    if not _aud_ok(claims.get("aud")):
        raise JWTError("Invalid audience")

    return claims


# -------------------------
# Role lookup via service role
# -------------------------

async def fetch_user_role_from_db(user_id: str) -> str:
    """
    Uses service role to read profiles.role.
    If service role missing/unavailable, fail-soft to "student".
    """
    if not settings.supabase_url:
        return "student"
    if not settings.supabase_service_role_key:
        return "student"

    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/profiles"
    params = {"id": f"eq.{user_id}", "select": "role"}
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            rows = r.json() or []
    except Exception:
        return "student"

    if not rows:
        return "student"

    role = rows[0].get("role") or "student"
    return role if role in {"student", "professor", "admin"} else "student"


# -------------------------
# FastAPI dependencies
# -------------------------

async def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        claims = await verify_supabase_jwt(token)
    except HTTPException:
        raise
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    user_id = claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing sub")

    email = claims.get("email")
    role = await fetch_user_role_from_db(user_id)

    return CurrentUser(id=user_id, email=email, role=role, raw_claims=claims, access_token=token)



def require_roles(*allowed: str):
    allowed_set = set(allowed)

    async def _checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed_set:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user

    return _checker


def require_admin_mfa():
    async def _checker(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role != "admin":
            raise HTTPException(status_code=403, detail="Insufficient role")

        aal = (user.raw_claims or {}).get("aal")
        if aal != "aal2":
            raise HTTPException(status_code=403, detail="MFA required (aal2)")

        return user

    return _checker
