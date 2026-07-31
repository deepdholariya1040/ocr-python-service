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

                # _ocr_instance = PaddleOCR(
                #     lang=settings.OCR_LANGUAGE,
                #     use_textline_orientation=False,
                # )

                _ocr_instance = PaddleOCR(
                    lang=settings.OCR_LANGUAGE,

                    # Use lightweight detection model
                    text_detection_model_name="PP-OCRv5_mobile_det",

                    # Use lightweight recognition model
                    text_recognition_model_name="en_PP-OCRv5_mobile_rec",

                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )

                logger.warning(
                    "PADDLE_DEBUG",
                    extra={
                        "version": __import__("paddleocr").__version__,
                        "has_ocr": hasattr(_ocr_instance, "ocr"),
                        "has_predict": hasattr(_ocr_instance, "predict"),
                        "type": str(type(_ocr_instance)),
                    },
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
            result = ocr.predict(temp_path)

        logger.info(
            "ocr_raw_result",
            extra={
                "result": str(result)[:3000],
            },
        )

        if not result:
            return ""

        texts: list[str] = []

        for res in result or []:
            try:
                rec_texts = res["rec_texts"] if isinstance(res, dict) else res.get("rec_texts")
            except Exception:
                rec_texts = getattr(res, "rec_texts", None)

            if not rec_texts:
                continue

            for text in rec_texts:
                text = str(text).strip()
                if text:
                    texts.append(text)

        raw_text = "\n".join(texts).strip()

        print("=" * 100)
        print("OCR RAW TEXT START")
        print(raw_text)
        print("OCR RAW TEXT END")
        print("=" * 100)

        logger.info(
            "OCR_TEXT_LENGTH",
            extra={
                "length": len(raw_text)
            },
        )

        return raw_text

    except Exception:  # noqa: BLE001
        logger.exception("ocr_extraction_failed")

        print("=" * 80)
        print(traceback.format_exc())
        print("=" * 80)

        return ""
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)