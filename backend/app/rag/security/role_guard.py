from __future__ import annotations


def enforce_audience_role_match(user_role: str, audience: str) -> None:
    user_role = (user_role or '').strip().lower()
    audience = (audience or '').strip().lower()
    if user_role == 'admin':
        return
    if user_role != audience:
        raise PermissionError(f'role {user_role!r} cannot access {audience!r} knowledge base')
