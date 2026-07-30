"""
=============================================================================
OCR route
=============================================================================
POST /api/ocr/process

Path and field names are NOT arbitrary - they are dictated by the existing
Node backend and must not change without also updating it:

  - Path: matches PYTHON_SERVICE_URL in the backend's .env
    (http://<host>/api/ocr/process).
  - Fields: "frontImage" / "backImage", matching the exact keys
    src/modules/ocr/scan.service.js appends to its FormData.

At least one of the two files is required (mirrors the same rule already
enforced in ocr.controller.js on the Node side).
=============================================================================
"""

from __future__ import annotations

import asyncio
import functools
import time

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.models.schemas import OCRProcessResponse
from app.services.pipeline import run_pipeline
from app.utils.exceptions import InvalidRequestError, ProcessingTimeoutError
from app.utils.validators import validate_and_load_image

logger = get_logger(__name__)

router = APIRouter(prefix="/api/ocr", tags=["ocr"])


@router.post("/process", response_model=OCRProcessResponse)
async def process_scan(
    request: Request,
    frontImage: UploadFile | None = File(default=None),
    backImage: UploadFile | None = File(default=None),
    settings: Settings = Depends(get_settings),
) -> OCRProcessResponse:
    request_id = getattr(request.state, "request_id", "")
    started = time.perf_counter()

    if frontImage is None and backImage is None:
        raise InvalidRequestError("At least one image (frontImage or backImage) is required.")

    front_pil = None
    back_pil = None

    if frontImage is not None:
        front_bytes = await frontImage.read()
        front_pil = validate_and_load_image(front_bytes, frontImage.filename or "frontImage", settings)

    if backImage is not None:
        back_bytes = await backImage.read()
        back_pil = validate_and_load_image(back_bytes, backImage.filename or "backImage", settings)

    logger.info(
        "ocr_scan_request_received",
        extra={
            "requestId": request_id,
            "hasFront": front_pil is not None,
            "hasBack": back_pil is not None,
        },
    )

    pipeline_call = functools.partial(
        run_pipeline,
        front_image=front_pil,
        back_image=back_pil,
        settings=settings,
        request_id=request_id,
    )

    try:
        # run_pipeline is synchronous (PaddleOCR and the Gemini SDK are
        # both blocking calls) - running it directly here would freeze the
        # event loop for every other concurrent request. Offloading it to
        # a worker thread keeps the server responsive under load.
        result = await asyncio.wait_for(
            asyncio.to_thread(pipeline_call),
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        logger.error(
            "ocr_scan_timed_out",
            extra={"requestId": request_id, "timeoutSeconds": settings.REQUEST_TIMEOUT_SECONDS},
        )
        raise ProcessingTimeoutError(
            "The scan took too long to process. Please try again."
        ) from exc

    logger.info(
        "ocr_scan_request_completed",
        extra={
            "requestId": request_id,
            "totalDurationMs": round((time.perf_counter() - started) * 1000, 2),
        },
    )

    return result
