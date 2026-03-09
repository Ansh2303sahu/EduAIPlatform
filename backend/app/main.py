from typing import Callable, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import Response
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from app.api import prof_results
from app.api.routes import api_router
from app.core.config import settings
from app.core.rate_limit import limiter

app = FastAPI(title="EduAIPlatform Backend")

app.state.limiter = limiter
rate_limit_handler = cast(
    Callable[[Request, Exception], Response],
    _rate_limit_exceeded_handler,
)
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"ok": True}

# ✅ single source of truth for /api prefix
app.include_router(api_router, prefix="/api")
