from fastapi import Depends
from app.core.deps import get_current_user
from app.core.supabase_auth import CurrentUser

def require_roles(*roles: str):
    async def _dep(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in set(roles):
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user
    return _dep
