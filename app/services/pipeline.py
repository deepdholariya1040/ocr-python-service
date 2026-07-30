# """
# =============================================================================
# OCR pipeline orchestrator
# =============================================================================
# Implements the full required flow end-to-end for one scan request:

#   Business Card Image(s)
#       -> Image Processing
#       -> OCR
#       -> Raw OCR Text
#       -> Gemini AI
#       -> Structured Parsed Data

#   ...running QR/barcode detection on each image in the same pass.

# This module owns request-level error handling: it validates images first
# (raises -> 4xx, handled by the router), then treats every downstream step
# (OCR, QR/barcode, Gemini) as best-effort - none of them may raise past this
# function. If OCR finds nothing, if Gemini is down, if no codes are found:
# the response is still 200, just with empty fields where data legitimately
# isn't available.
# =============================================================================
# """

# from __future__ import annotations

# from typing import Dict, Optional

# from PIL import Image

# from app.config import Settings
# from app.logging_config import get_logger
# from app.models.schemas import BarcodeItem, OCRProcessResponse, QRCodeItem
# from app.services import ocr_engine, parser, qr_barcode_service
# from app.services.image_processing import assess_image_quality, preprocess_for_ocr
# from app.utils.timing import stage_timer

# logger = get_logger(__name__)


# def run_pipeline(
#     *,
#     front_image: Optional[Image.Image],
#     back_image: Optional[Image.Image],
#     settings: Settings,
#     request_id: str,
# ) -> OCRProcessResponse:
#     timings: Dict[str, float] = {}
#     image_quality: Dict[str, Dict[str, object]] = {}

#     front_text = ""
#     back_text = ""
#     qr_codes: list[QRCodeItem] = []
#     barcodes: list[BarcodeItem] = []

#     if front_image is not None:
#         with stage_timer(timings, "front_ocr_ms"):
#             front_text = ocr_engine.extract_text(preprocess_for_ocr(front_image, settings), settings)
#         with stage_timer(timings, "front_qr_barcode_ms"):
#             front_qr, front_barcodes = qr_barcode_service.detect_codes(front_image, settings)
#         qr_codes.extend(front_qr)
#         barcodes.extend(front_barcodes)
#         image_quality["front"] = assess_image_quality(front_image)

#     if back_image is not None:
#         with stage_timer(timings, "back_ocr_ms"):
#             back_text = ocr_engine.extract_text(preprocess_for_ocr(back_image, settings), settings)
#         with stage_timer(timings, "back_qr_barcode_ms"):
#             back_qr, back_barcodes = qr_barcode_service.detect_codes(back_image, settings)
#         qr_codes.extend(back_qr)
#         barcodes.extend(back_barcodes)
#         image_quality["back"] = assess_image_quality(back_image)

#     merged_text = "\n".join(t for t in (front_text, back_text) if t)

#     with stage_timer(timings, "parsing_ms"):
#         parsed_data, provider_label = parser.build_parsed_data(merged_text, settings)

#     # "empty" only tells us Gemini/regex were skipped for lack of
#     # meaningful text - if a code was still detected, make that visible in
#     # diagnostics rather than reporting a bare "empty" scan.
#     if provider_label == "empty" and (qr_codes or barcodes):
#         provider_label = "qr-barcode-only"

#     logger.info(
#         "ocr_pipeline_completed",
#         extra={
#             "requestId": request_id,
#             "provider": provider_label,
#             "frontTextLength": len(front_text),
#             "backTextLength": len(back_text),
#             "qrCodesFound": len(qr_codes),
#             "barcodesFound": len(barcodes),
#             **timings,
#         },
#     )

#     return OCRProcessResponse(
#         success=True,
#         requestId=request_id,
#         provider=f"gemini-python-ocr-service:{provider_label}",
#         frontOCRText=front_text,
#         backOCRText=back_text,
#         mergedOCRText=merged_text,
#         parsedData=parsed_data,
#         qrCodes=qr_codes,
#         barcodes=barcodes,
#         meta={"timings": timings, "imageQuality": image_quality},
#     )


# ================================

"""
=============================================================================
OCR pipeline orchestrator
=============================================================================
"""

from __future__ import annotations

import traceback
from typing import Dict, Optional

from PIL import Image

from app.config import Settings
from app.logging_config import get_logger
from app.models.schemas import BarcodeItem, OCRProcessResponse, QRCodeItem
from app.services import ocr_engine, parser, qr_barcode_service
from app.services.image_processing import assess_image_quality, preprocess_for_ocr
from app.utils.timing import stage_timer

logger = get_logger(__name__)


def run_pipeline(
    *,
    front_image: Optional[Image.Image],
    back_image: Optional[Image.Image],
    settings: Settings,
    request_id: str,
) -> OCRProcessResponse:

    logger.info(f"[{request_id}] PIPELINE START")

    try:
        timings: Dict[str, float] = {}
        image_quality: Dict[str, Dict[str, object]] = {}

        front_text = ""
        back_text = ""
        qr_codes: list[QRCodeItem] = []
        barcodes: list[BarcodeItem] = []

        # ------------------------------------------------------------------
        # FRONT IMAGE
        # ------------------------------------------------------------------
        if front_image is not None:
            logger.info(f"[{request_id}] STEP 1 - Front preprocessing")

            processed = preprocess_for_ocr(front_image, settings)

            logger.info(f"[{request_id}] STEP 2 - Front OCR")

            with stage_timer(timings, "front_ocr_ms"):
                front_text = ocr_engine.extract_text(processed, settings)

            logger.info(
                f"[{request_id}] STEP 3 - Front OCR completed ({len(front_text)} chars)"
            )

            logger.info(f"[{request_id}] STEP 4 - Front QR/Barcode")

            with stage_timer(timings, "front_qr_barcode_ms"):
                front_qr, front_barcodes = qr_barcode_service.detect_codes(
                    front_image,
                    settings,
                )

            qr_codes.extend(front_qr)
            barcodes.extend(front_barcodes)

            logger.info(
                f"[{request_id}] STEP 5 - Front QR done "
                f"(QR={len(front_qr)}, Barcode={len(front_barcodes)})"
            )

            image_quality["front"] = assess_image_quality(front_image)

        # ------------------------------------------------------------------
        # BACK IMAGE
        # ------------------------------------------------------------------
        if back_image is not None:
            logger.info(f"[{request_id}] STEP 6 - Back preprocessing")

            processed = preprocess_for_ocr(back_image, settings)

            logger.info(f"[{request_id}] STEP 7 - Back OCR")

            with stage_timer(timings, "back_ocr_ms"):
                back_text = ocr_engine.extract_text(processed, settings)

            logger.info(
                f"[{request_id}] STEP 8 - Back OCR completed ({len(back_text)} chars)"
            )

            logger.info(f"[{request_id}] STEP 9 - Back QR/Barcode")

            with stage_timer(timings, "back_qr_barcode_ms"):
                back_qr, back_barcodes = qr_barcode_service.detect_codes(
                    back_image,
                    settings,
                )

            qr_codes.extend(back_qr)
            barcodes.extend(back_barcodes)

            logger.info(
                f"[{request_id}] STEP 10 - Back QR done "
                f"(QR={len(back_qr)}, Barcode={len(back_barcodes)})"
            )

            image_quality["back"] = assess_image_quality(back_image)

        merged_text = "\n".join(t for t in (front_text, back_text) if t)

        logger.info(
            f"[{request_id}] STEP 11 - Parser (merged text length={len(merged_text)})"
        )

        with stage_timer(timings, "parsing_ms"):
            parsed_data, provider_label = parser.build_parsed_data(
                merged_text,
                settings,
            )

        logger.info(f"[{request_id}] STEP 12 - Parser completed")

        if provider_label == "empty" and (qr_codes or barcodes):
            provider_label = "qr-barcode-only"

        logger.info(
            "ocr_pipeline_completed",
            extra={
                "requestId": request_id,
                "provider": provider_label,
                "frontTextLength": len(front_text),
                "backTextLength": len(back_text),
                "qrCodesFound": len(qr_codes),
                "barcodesFound": len(barcodes),
                **timings,
            },
        )

        return OCRProcessResponse(
            success=True,
            requestId=request_id,
            provider=f"gemini-python-ocr-service:{provider_label}",
            frontOCRText=front_text,
            backOCRText=back_text,
            mergedOCRText=merged_text,
            parsedData=parsed_data,
            qrCodes=qr_codes,
            barcodes=barcodes,
            meta={
                "timings": timings,
                "imageQuality": image_quality,
            },
        )

    except Exception:
        logger.error(
            f"[{request_id}] PIPELINE FAILED\n{traceback.format_exc()}"
        )
        raise