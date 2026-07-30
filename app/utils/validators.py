"""
=============================================================================
Image validation
=============================================================================
Runs BEFORE anything touches OCR/Gemini/QR pipelines. Two jobs:

  1. Reject obviously-bad input fast (empty file, oversized file,
     content-type we don't support) without ever loading it into memory
     twice.
  2. Confirm the bytes we DID receive actually decode as an image (catches
     corrupt/truncated uploads, renamed non-image files, etc.) and hand
     back a normalized, RGB Pillow Image ready for the rest of the
     pipeline.
=============================================================================
"""

from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.utils.exceptions import CorruptImageError, PayloadTooLargeError, UnsupportedImageError


def validate_and_load_image(raw_bytes: bytes, filename: str, settings: Settings) -> Image.Image:
    if not raw_bytes:
        raise CorruptImageError(f"'{filename}' is empty.")

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(raw_bytes) > max_bytes:
        raise PayloadTooLargeError(
            f"'{filename}' exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB upload limit."
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        image.load()  # force full decode now, not lazily later mid-pipeline
    except UnidentifiedImageError as exc:
        raise CorruptImageError(f"'{filename}' could not be decoded as an image.") from exc
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any decode failure -> 422
        raise CorruptImageError(f"'{filename}' is corrupt or unreadable.") from exc

    if image.format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF", "HEIC", "HEIF", "MPO"}:
        raise UnsupportedImageError(
            f"'{filename}' has unsupported format '{image.format}'."
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image
