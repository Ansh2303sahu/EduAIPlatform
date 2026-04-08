from contextlib import asynccontextmanager
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
from app.api.internal import assessment_router
from app.events import close_client, close_redis, internal_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------- startup ----------
    yield
    # ---------- shutdown ----------
    # Close shared HTTP client used by the event emitter
    await close_client()
    # Close Redis client used by the idempotency / metrics router
    await close_redis()


app = FastAPI(title="EduAIPlatform Backend", lifespan=lifespan)

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


# Internal n8n → backend callbacks (idempotency, metrics, escalations, dead-letter)
# IMPORTANT: mount BEFORE the public /api router; no auth proxy should expose /internal.
app.include_router(internal_router)

# Phase 14 — multi-model assessment callbacks (/internal/assessment/*)
app.include_router(assessment_router)

# ✅ single source of truth for /api prefix
app.include_router(api_router, prefix="/api")
