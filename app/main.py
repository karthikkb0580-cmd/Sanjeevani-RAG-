"""
Sanjeevani AI – RAG Service
app/main.py

FastAPI application entrypoint.
Configures middleware, exception handlers, lifespan events, and API routes.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import chat, documents, health, retrieval
from app.config.settings import get_settings
from app.vectordb.qdrant_client import get_qdrant_manager

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan – startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle:
    - Startup: connect to Qdrant, ensure collection exists
    - Shutdown: close Qdrant connection
    """
    logger.info("═══════════════════════════════════════════")
    logger.info(" Sanjeevani RAG Service v%s starting …", settings.app_version)
    logger.info("═══════════════════════════════════════════")

    # Connect to Qdrant
    manager = get_qdrant_manager()
    await manager.connect()
    logger.info("Qdrant connection established")

    logger.info("Service ready – listening on %s:%s", settings.host, settings.port)
    logger.info("───────────────────────────────────────────")

    yield  # ← Application runs here

    # Shutdown
    logger.info("Shutting down Sanjeevani RAG Service …")
    await manager.disconnect()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def create_application() -> FastAPI:
    app = FastAPI(
        title="Sanjeevani AI – RAG Service",
        description=(
            "Enterprise-grade Retrieval-Augmented Generation service for "
            "scientific research papers. Index documents, retrieve relevant "
            "context, and generate grounded answers with citations."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "path": str(request.url),
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(retrieval.router)
    app.include_router(chat.router)

    logger.debug("Routers registered: health, documents, retrieval, chat")
    return app


app = create_application()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
        access_log=True,
    )
