"""
=============================================================================
Configuration
=============================================================================
Single source of truth for every environment-driven setting used by the
service. Nothing below should ever be hardcoded elsewhere - if a module
needs a tunable value, it belongs here.

All values have safe, production-sane defaults EXCEPT the Gemini API key,
which is required for AI parsing to run (the service still boots and still
serves requests without it - it just falls back to regex parsing for every
request and logs a warning once at startup).
=============================================================================
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Service ---------------------------------------------------------
    SERVICE_NAME: str = "python-ocr-service"
    ENVIRONMENT: str = "development"  # development | production
    HOST: str = "0.0.0.0"
    PORT: int = 5001
    LOG_LEVEL: str = "INFO"

    # CORS - only the Node backend needs to call this service directly,
    # but kept open/configurable in case of local tooling / direct testing.
    CORS_ALLOWED_ORIGINS: str = "*"

    # --- Gemini AI ---------------------------------------------------------
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    GEMINI_TIMEOUT_SECONDS: float = 25.0
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_RETRY_MIN_WAIT_SECONDS: float = 1.0
    GEMINI_RETRY_MAX_WAIT_SECONDS: float = 6.0
    GEMINI_TEMPERATURE: float = 0.1

    # --- OCR (PaddleOCR) ---------------------------------------------------
    OCR_LANGUAGE: str = "en"
    OCR_USE_GPU: bool = False
    OCR_MAX_IMAGE_DIMENSION: int = 2200  # px, upper bound after preprocessing
    OCR_MIN_IMAGE_DIMENSION: int = 800  # px, upscale threshold for small images

    # Below this many "meaningful" characters (alnum, post-OCR), the text is
    # treated as noise, not content - Gemini and the regex parser are both
    # skipped. Prevents a stray speck the OCR engine misreads as one or two
    # characters from triggering a full AI parsing call.
    OCR_MIN_MEANINGFUL_CHARS: int = 6

    # Max number of PaddleOCR `.predict()` calls allowed to run at the same
    # time. PaddleOCR's inference session is not guaranteed safe under
    # unlimited concurrent access from multiple threads; this bounds it via
    # a semaphore rather than a hard lock so it can be raised later if the
    # underlying engine is confirmed safe at higher concurrency.
    OCR_MAX_CONCURRENCY: int = 1

    # --- Uploads / request limits -------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_CONTENT_TYPES: str = (
        "image/jpeg,image/png,image/webp,image/bmp,image/tiff,image/heic,image/heif"
    )

    # Hard ceiling on total time spent processing one /api/ocr/process
    # request (image validation is already done by this point). Guards
    # against a hung PaddleOCR call or a Gemini retry loop that ignores its
    # own per-call timeout from ever holding a request open indefinitely.
    REQUEST_TIMEOUT_SECONDS: float = 55.0

    # --- QR / Barcode -------------------------------------------------------
    BARCODE_TRY_ROTATIONS: bool = True  # try 0/90/180/270 if nothing found upright

    @property
    def cors_origins_list(self) -> List[str]:
        if self.CORS_ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.CORS_ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def allowed_content_types_list(self) -> List[str]:
        return [c.strip() for c in self.ALLOWED_CONTENT_TYPES.split(",") if c.strip()]

@lru_cache
def get_settings() -> Settings:
    return Settings()
