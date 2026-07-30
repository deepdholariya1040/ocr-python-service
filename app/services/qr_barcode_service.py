"""
=============================================================================
QR Code & Barcode detection
=============================================================================
Uses `zxing-cpp` (native ZXing C++ port with Python bindings) rather than
`pyzbar`/libzbar, because zxing-cpp natively covers every format this
service is required to support out of the box - QR, Code128, Code39,
EAN-13/8, UPC-A/E, PDF417, Data Matrix, Aztec, and more - in one pass, with
no extra system package to apt-install (the wheel ships its own compiled
native code), which keeps the Docker image both simpler and smaller.

Detection runs against the ORIGINAL (non-grayscaled/non-sharpened) image,
not the OCR-preprocessed one - barcode readers want the real pixel data,
not text-oriented contrast/sharpen filters, and zxing-cpp already handles
its own internal binarization.

If nothing is found upright and BARCODE_TRY_ROTATIONS is enabled, the image
is retried at 90/180/270 degrees before giving up - business card photos
are frequently captured with the card rotated relative to the phone.
=============================================================================
"""

from __future__ import annotations

from typing import List

import zxingcpp
from PIL import Image

from app.config import Settings
from app.logging_config import get_logger
from app.models.schemas import BarcodeItem, QRCodeItem

logger = get_logger(__name__)

# zxingcpp reports its own format enum; QR/Aztec/DataMatrix/PDF417 are all
# genuinely 2D "matrix" symbologies -> qrCodes bucket (schema calls the
# bucket "qrCodes" but the backend's own model comment describes it as
# generic 2D code storage with a `type` field, so we keep the *type* value
# accurate and just route 2D formats there, everything else to barcodes).
_MATRIX_FORMATS = {"QRCode", "Aztec", "DataMatrix", "PDF417", "MicroQRCode", "RMQRCode"}


def detect_codes(image: Image.Image, settings: Settings) -> tuple[List[QRCodeItem], List[BarcodeItem]]:
    results = _read_with_rotations(image, settings)

    qr_codes: List[QRCodeItem] = []
    barcodes: List[BarcodeItem] = []
    seen: set[tuple[str, str]] = set()

    for result in results:
        if not result.valid:
            continue

        format_name = _format_to_str(result.format)
        content = result.text or ""
        dedupe_key = (format_name, content)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        if format_name in _MATRIX_FORMATS:
            qr_codes.append(
                QRCodeItem(
                    type=format_name if format_name != "QRCode" else "QR_CODE",
                    dataType=_infer_data_type(content),
                    content=content,
                    raw=content,
                )
            )
        else:
            barcodes.append(
                BarcodeItem(
                    type=format_name or "UNKNOWN",
                    content=content,
                    raw=content,
                )
            )

    return qr_codes, barcodes


def _format_to_str(fmt) -> str:
    """
    zxingcpp.BarcodeFormat is a compiled enum whose str() representation
    has looked like "BarcodeFormat.QRCode" across the versions this
    service has been tested against. Prefer the enum's own `.name`
    attribute when present (version-proof), and only fall back to
    string-splitting the repr if it isn't.
    """
    name = getattr(fmt, "name", None)
    if isinstance(name, str) and name:
        return name
    text = str(fmt)
    return text.rsplit(".", 1)[-1]


def _read_with_rotations(image: Image.Image, settings: Settings) -> list:
    try:
        results = zxingcpp.read_barcodes(image)
    except Exception as exc:  # noqa: BLE001 - detection must never crash the request
        logger.warning("qr_barcode_detection_failed", extra={"error": str(exc)})
        results = []

    if results or not settings.BARCODE_TRY_ROTATIONS:
        return results

    for angle in (90, 180, 270):
        try:
            rotated = image.rotate(angle, expand=True)
            results = zxingcpp.read_barcodes(rotated)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "qr_barcode_rotation_attempt_failed",
                extra={"angle": angle, "error": str(exc)},
            )
            results = []
        if results:
            break

    return results


def _infer_data_type(content: str) -> str:
    lowered = content.strip().lower()
    if lowered.startswith(("http://", "https://")):
        return "URL"
    if lowered.startswith("mailto:"):
        return "EMAIL"
    if lowered.startswith(("tel:", "sms:")):
        return "PHONE"
    if lowered.startswith("wifi:"):
        return "WIFI"
    if lowered.startswith("begin:vcard"):
        return "VCARD"
    return "TEXT"
