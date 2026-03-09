from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Optional

from fastapi import HTTPException, status

_BUCKETS: Dict[str, Deque[float]] = {}


def _key(route: str, role: str, user_id: Optional[str], ip: Optional[str]) -> str:
    return f"{route}:{role}:{user_id or 'anon'}:{ip or 'noip'}"


def enforce_rate_limit(
    *,
    route: str,
    per_minute: int,
    role: str,
    user_id: Optional[str],
    ip: Optional[str],
) -> None:
    now = time.time()
    k = _key(route, role or "unknown", user_id, ip)

    dq = _BUCKETS.get(k)
    if dq is None:
        dq = deque()
        _BUCKETS[k] = dq

    cutoff = now - 60.0
    while dq and dq[0] < cutoff:
        dq.popleft()

    if len(dq) >= per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {per_minute}/min",
        )

    dq.append(now)