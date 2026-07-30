"""
=============================================================================
Exception hierarchy
=============================================================================
Only truly unrecoverable, client-facing problems raise these (bad/missing
images, unsupported formats, no input at all). Everything downstream of
"we have a valid image" - OCR failure, Gemini failure/timeout, parsing
weirdness - is handled internally with graceful fallbacks and NEVER raises;
that is what "never crash because Gemini fails" means in practice.
=============================================================================
"""

from __future__ import annotations


class OCRServiceError(Exception):
    """Base class for all deliberately-raised, client-facing errors."""

    status_code: int = 500

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class InvalidRequestError(OCRServiceError):
    """No image supplied, or the request itself is malformed."""

    status_code = 400


class UnsupportedImageError(OCRServiceError):
    """File extension / content-type is not one we accept."""

    status_code = 415


class CorruptImageError(OCRServiceError):
    """File claims to be an image but cannot actually be decoded."""

    status_code = 422


class PayloadTooLargeError(OCRServiceError):
    status_code = 413


class ProcessingTimeoutError(OCRServiceError):
    """The pipeline (OCR + parsing) exceeded REQUEST_TIMEOUT_SECONDS."""

    status_code = 504
