"""
Docling RAG Backend — FastAPI application entry-point.

Run with::

    uvicorn src.main:app --host 0.0.0.0 --port 5008
"""

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.config import settings
from src.errors import ErrorResponse
from src.router import router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: create services, warmup models on startup, clean up on shutdown."""
    from src.services.model_lifecycle_service import ModelLifecycleService
    from src.services.vector_store_service import VectorStoreService
    from src.services.table_store_service import TableStoreService

    log = logging.getLogger(__name__)

    # ── Create shared service instances ──────────────────────────
    vec = VectorStoreService()
    table = TableStoreService()
    lc = ModelLifecycleService(vec_store=vec, reranker=vec.reranker)
    pg_pool = await asyncpg.create_pool(dsn=settings.postgres_dsn)

    app.state.vector_service = vec
    app.state.table_service = table
    app.state.lifecycle = lc
    app.state.pg_pool = pg_pool

    if settings.warmup_on_startup:
        log.info("WARMUP_ON_STARTUP=True — loading models into VRAM …")
        await lc.warmup()

    # Start the idle-watcher background task
    lc.start_watcher()

    yield  # application is running

    # ── Shutdown ─────────────────────────────────────────────
    lc.stop_watcher()
    await vec.close()
    await table.close()
    await pg_pool.close()
    log.info("FastAPI shutdown — all resources released")


app = FastAPI(title="RAG Backend", version="2.0.0", lifespan=lifespan)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions globally."""
    logging.error(f"Unhandled exception: {exc}\nTraceback:\n{traceback.format_exc()}")

    # Return a sanitized error message (no internal paths, DSNs, or stack traces)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error", detail="An unexpected error occurred"
        ),
    )


# Allow the OpenWebUI origin (and localhost dev setups) to call /v1/warmup
# from browser-side JavaScript.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5008)

    print("app started :-)")