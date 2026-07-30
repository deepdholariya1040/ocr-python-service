"""
=============================================================================
Response / internal schemas
=============================================================================
These models mirror, field for field, what the Node backend already knows
how to consume:

  - src/modules/ocr/scan.service.js     -> POSTs frontImage/backImage,
                                            returns response.data verbatim.
  - src/modules/ocr/ocr.service.js      -> reads:
                                              ocrPayload.frontOCRText
                                              ocrPayload.backOCRText
                                              ocrPayload.mergedOCRText
                                              ocrPayload.parsedData
                                              ocrPayload.provider
                                              result.qrCodes / .barcodes
  - src/modules/business-cards/
      businessCard.model.js             -> parsedData sub-schema only keeps
                                            {name, designation, company,
                                             email, phones[], website,
                                             address}. Anything else placed
                                            on parsedData (emails[],
                                            websites[], dynamicFields) is
                                            read by ocr.service.js BEFORE
                                            Mongoose strips unknown keys,
                                            and re-homed onto the
                                            BusinessCard's own top-level
                                            emails/websites/dynamicFields
                                            fields - which is exactly why
                                            those keys exist here.
                                          -> qrCodes[]: {type, dataType,
                                             content, raw}
                                          -> barcodes[]: {type, content, raw}

No field here is invented for its own sake - every one of them is read by
existing Node code. Do not rename any of these keys without also updating
ocr.service.js.
=============================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class QRCodeItem(BaseModel):
    type: str = "QR_CODE"
    dataType: str = "TEXT"
    content: str = ""
    raw: str = ""


class BarcodeItem(BaseModel):
    type: str = "UNKNOWN"
    content: str = ""
    raw: str = ""


class ParsedData(BaseModel):
    # --- Fields the Mongoose sub-schema persists directly -----------------
    name: str = ""
    designation: str = ""
    company: str = ""
    email: str = ""
    phones: List[str] = Field(default_factory=list)
    website: str = ""
    address: str = ""

    # --- Additive fields, read by ocr.service.js then re-homed ------------
    emails: List[str] = Field(default_factory=list)
    websites: List[str] = Field(default_factory=list)
    dynamicFields: Dict[str, Any] = Field(default_factory=dict)


class OCRProcessResponse(BaseModel):
    success: bool = True
    requestId: str = ""
    provider: str = "gemini-python-ocr-service"

    frontOCRText: str = ""
    backOCRText: str = ""
    mergedOCRText: str = ""

    parsedData: ParsedData = Field(default_factory=ParsedData)

    qrCodes: List[QRCodeItem] = Field(default_factory=list)
    barcodes: List[BarcodeItem] = Field(default_factory=list)

    # Non-authoritative diagnostics. Node does not read this, but it is
    # invaluable when debugging a specific scan from the Python side.
    meta: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """
    Mirrors the Node ApiError/ApiResponse envelope
    ({ success, statusCode, message }) so the two services are
    consistent to work with, even though today's Node code path
    (axios throwing on non-2xx) does not parse this body.
    """

    success: bool = False
    statusCode: int
    message: str
    requestId: str = ""
