"""
=============================================================================
OCR engine (PaddleOCR)
=============================================================================
Extracts raw text from a preprocessed image.

Pipeline:

Business Card Image
        ↓
Image Processing
        ↓
PaddleOCR
        ↓
Raw OCR Text
        ↓
Gemini AI

The function never raises OCR-related exceptions. Any OCR failure simply
returns an empty string so the downstream parser can gracefully continue.
=============================================================================
"""

from __future__ import annotations

import tempfile
import threading
import time
import traceback
from pathlib import Path

from paddleocr import PaddleOCR
from PIL import Image

from app.config import Settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_ocr_instance: PaddleOCR | None = None
_init_lock = threading.Lock()
_predict_semaphore: threading.Semaphore | None = None


def _get_semaphore(settings: Settings) -> threading.Semaphore:
    global _predict_semaphore
    if _predict_semaphore is None:
        with _init_lock:
            if _predict_semaphore is None:
                _predict_semaphore = threading.Semaphore(max(1, settings.OCR_MAX_CONCURRENCY))
    return _predict_semaphore


def _get_ocr(settings: Settings) -> PaddleOCR:
    """
    Initialize PaddleOCR exactly once and reuse the same instance across
    every request for the lifetime of the process. Thread-safe: concurrent
    first-callers block on `_init_lock` instead of racing to build two
    separate model instances.
    """
    global _ocr_instance

    if _ocr_instance is None:
        with _init_lock:
            if _ocr_instance is None:
                started = time.perf_counter()
                logger.info("initializing_paddleocr", extra={"lang": settings.OCR_LANGUAGE})

                _ocr_instance = PaddleOCR(
                lang=settings.OCR_LANGUAGE,
                use_angle_cls=False,
            )

                logger.info(
                    "paddleocr_initialized",
                    extra={"durationMs": round((time.perf_counter() - started) * 1000, 2)},
                )

    return _ocr_instance


def warmup(settings: Settings) -> bool:
    """
    Forces PaddleOCR's (heavy) model load to happen once, up front, at
    application startup rather than on whichever request happens to arrive
    first. Never raises - a warmup failure is logged and the engine will
    simply retry lazily on first use.
    """
    try:
        _get_ocr(settings)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("paddleocr_warmup_failed", extra={"error": str(exc)})
        return False


def is_ready() -> bool:
    return _ocr_instance is not None


def extract_text(image: Image.Image, settings: Settings) -> str:
    """
    Extract raw OCR text.

    Returns:
        Raw OCR text or empty string.
    """

    temp_path: str | None = None
    try:
        ocr = _get_ocr(settings)
        logger.warning("OCR_ENGINE_VERSION_20260731_V3")
        semaphore = _get_semaphore(settings)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            temp_path = tmp.name

        with semaphore:
            result = ocr.ocr(temp_path, cls=False)

        logger.info(
            "ocr_raw_result",
            extra={
                "result": str(result)[:3000],
            },
        )

        if not result:
            return ""

        texts: list[str] = []

        for page in result or []:
            if not page:
                continue

            for line in page:
                try:
                    if (
                        isinstance(line, (list, tuple))
                        and len(line) >= 2
                        and isinstance(line[1], (list, tuple))
                        and len(line[1]) >= 1
                    ):
                        text = str(line[1][0]).strip()
                        if text:
                            texts.append(text)
                except Exception:
                    continue

        return "\n".join(texts).strip()

    except Exception:  # noqa: BLE001
        logger.exception("ocr_extraction_failed")

        print("=" * 80)
        print(traceback.format_exc())
        print("=" * 80)

        raise
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)