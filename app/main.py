"""
=============================================================================
Application entrypoint
=============================================================================
Wires together config, logging, middleware, routers, and global exception
handling. Kept intentionally thin - all real logic lives in app/services/.

Run locally:   uvicorn app.main:app --reload --port 5001
Run in prod:   uvicorn app.main:app --host 0.0.0.0 --port $PORT
=============================================================================
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes_health import router as health_router
from app.api.routes_ocr import router as ocr_router
from app.config import get_settings
from app.core.request_context import RequestIdMiddleware
from app.logging_config import configure_logging, get_logger
from app.models.schemas import ErrorResponse
from app.services import ocr_engine
from app.utils.exceptions import OCRServiceError

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.GEMINI_API_KEY:
        logger.warning(
            "startup_warning_gemini_not_configured",
            extra={"detail": "GEMINI_API_KEY is not set - every scan will use the regex fallback parser."},
        )

    logger.info(
        "service_starting",
        extra={
            "service": settings.SERVICE_NAME,
            "environment": settings.ENVIRONMENT,
            "geminiModel": settings.GEMINI_MODEL,
            "geminiConfigured": bool(settings.GEMINI_API_KEY),
        },
    )

    # Load the (heavy) PaddleOCR model once, now, instead of on whichever
    # request happens to arrive first. Runs off the event loop thread so a
    # multi-second model load doesn't stall it. Never fatal: a failed
    # warmup just means the engine retries lazily on first use.
    warmed_up = await asyncio.to_thread(ocr_engine.warmup, settings)
    logger.info("service_started", extra={"paddleocrWarmedUp": warmed_up})

    yield
    logger.info("service_shutting_down")


app = FastAPI(
    title="Business Card OCR Microservice",
    description=(
        "Production OCR + Gemini AI parsing + QR/Barcode detection service "
        "for the Node.js business-card backend."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ocr_router)


# -----------------------------------------------------------------------
# Global exception handling - every error path returns the same
# { success, statusCode, message } envelope, and NEVER leaks a stack
# trace to the client.
# -----------------------------------------------------------------------


@app.exception_handler(OCRServiceError)
async def handle_ocr_service_error(request: Request, exc: OCRServiceError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    logger.warning(
        "request_rejected",
        extra={"requestId": request_id, "statusCode": exc.status_code, "message": exc.message},
    )
    body = ErrorResponse(statusCode=exc.status_code, message=exc.message, requestId=request_id)
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    body = ErrorResponse(
        statusCode=422,
        message="Invalid request payload.",
        requestId=request_id,
    )
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    body = ErrorResponse(statusCode=exc.status_code, message=str(exc.detail), requestId=request_id)
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    # Full detail goes to logs only - never to the client.
    logger.error(
        "unhandled_exception",
        extra={"requestId": request_id, "error": str(exc), "errorType": type(exc).__name__},
        exc_info=True,
    )
    body = ErrorResponse(
        statusCode=500,
        message="An unexpected error occurred while processing the scan.",
        requestId=request_id,
    )
    return JSONResponse(status_code=500, content=body.model_dump())


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "http_request",
        extra={
            "requestId": getattr(request.state, "request_id", ""),
            "method": request.method,
            "path": request.url.path,
            "statusCode": response.status_code,
            "durationMs": duration_ms,
        },
    )
    return response
