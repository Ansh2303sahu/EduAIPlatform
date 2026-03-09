import httpx
from fastapi import HTTPException, status
from ..core.config import settings

BOUNDARY_POLICY = {
    "no_external_browsing": True,
    "no_cross_user_access": True,
    "no_training": True,
    "stateless": True,
}

async def request_ai_feedback(*, user_id: str, role: str, assignment_text: str) -> dict:
    """
    Enforces AI boundaries:
    - Only scoped content is sent (assignment_text for this request)
    - Includes user/role context (for policy checks)
    - No database access from AI service
    """
    payload = {
        "context": {
            "user_id": user_id,
            "role": role,
            "policy": BOUNDARY_POLICY,
        },
        "input": {
            "assignment_text": assignment_text[:20000],  # basic size guard
        }
    }

    headers = {"X-AI-SECRET": settings.ai_service_secret}

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(f"http://localhost:{settings.ai_service_port}/v1/feedback", json=payload, headers=headers)
        if r.status_code != 200:
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"AI service error: {r.text}")
        return r.json()
    except httpx.RequestError:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="AI service unreachable")
