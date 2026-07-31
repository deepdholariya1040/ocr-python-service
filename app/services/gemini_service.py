"""
=============================================================================
Gemini AI parsing service
=============================================================================
This is the "same approach" as the working reference file provided
alongside the backend (test_gemini.py): raw OCR text goes in, one
`client.models.generate_content(...)` call comes back with strict JSON out.
The prompt below is the same one proven to work, extended only with the
extra fields (department, mobiles, faxNumbers, socials, notes, GST/PAN/
bank/certification fields, etc.) the backend needs and documented as
"examples include but are not limited to" in the requirements.

Production hardening on top of the reference script:
  - API key loaded from env (never hardcoded).
  - Bounded timeout per call (GEMINI_TIMEOUT_SECONDS).
  - Retry with exponential backoff on transient failures
    (GEMINI_MAX_RETRIES), via tenacity.
  - Strict response validation: must be valid JSON AND pass through the
    GeminiCardExtraction pydantic model before being trusted. A response
    that fails validation after all retries is treated as a Gemini failure
    -> caller falls back to regex parsing. Gemini is NEVER allowed to
    crash the request.
  - Markdown code-fence stripping, since models frequently wrap JSON in
    ```json ... ``` even when told not to.
  - Any field Gemini returns that is NOT part of the known schema (e.g.
    "VAT Number", "FSSAI", "ISO Certification") is preserved automatically
    inside dynamicFields instead of being silently dropped, via
    model_config = ConfigDict(extra="allow").
=============================================================================
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Dict, Optional

from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.logging_config import get_logger
from app.models.schemas import ParsedData

logger = get_logger(__name__)

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class _GeminiTransientError(Exception):
    """Raised internally to trigger a tenacity retry."""


class _GeminiAddress(BaseModel):
    full: str = ""
    street: str = ""
    area: str = ""
    city: str = ""
    state: str = ""
    country: str = ""
    postalCode: str = ""


class _GeminiSocials(BaseModel):
    linkedin: str = ""
    facebook: str = ""
    instagram: str = ""
    twitter: str = ""
    whatsapp: str = ""
    telegram: str = ""
    youtube: str = ""


class GeminiCardExtraction(BaseModel):
    """Validates the exact JSON shape we ask Gemini for.

    extra="allow" is required so that any field Gemini returns which is
    NOT part of this known schema (VAT Number, Tax ID, FSSAI, Import
    Export License, Registration Number, Awards, ISO Certification, etc.)
    is kept on the model (accessible via `model_extra`) instead of being
    dropped by pydantic. `_to_parsed_data()` below folds those extras into
    dynamicFields automatically.
    """

    model_config = ConfigDict(extra="allow")

    name: str = ""
    firstName: str = ""
    lastName: str = ""
    designation: str = ""
    department: str = ""
    company: str = ""
    brand: str = ""

    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    mobiles: list[str] = Field(default_factory=list)
    faxNumbers: list[str] = Field(default_factory=list)

    website: str = ""
    websites: list[str] = Field(default_factory=list)

    address: _GeminiAddress = Field(default_factory=_GeminiAddress)

    socials: _GeminiSocials = Field(default_factory=_GeminiSocials)

    services: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)

    notes: str = ""

    gstNumber: str = ""
    panNumber: str = ""
    cin: str = ""
    msme: str = ""
    iec: str = ""
    udyam: str = ""

    upi: str = ""

    bankName: str = ""
    accountNumber: str = ""
    ifsc: str = ""
    branch: str = ""

    certifications: list[str] = Field(default_factory=list)
    licenses: list[str] = Field(default_factory=list)


_RESPONSE_SCHEMA_EXAMPLE = """{
  "name": "",
  "firstName": "",
  "lastName": "",
  "designation": "",
  "department": "",
  "company": "",
  "brand": "",

  "emails": [],
  "phones": [],
  "mobiles": [],
  "faxNumbers": [],

  "website": "",
  "websites": [],

  "address": {
      "full": "",
      "street": "",
      "area": "",
      "city": "",
      "state": "",
      "country": "",
      "postalCode": ""
  },

  "socials": {
      "linkedin": "",
      "facebook": "",
      "instagram": "",
      "twitter": "",
      "whatsapp": "",
      "telegram": "",
      "youtube": ""
  },

  "services": [],
  "products": [],

  "gstNumber": "",
  "panNumber": "",
  "cin": "",
  "msme": "",
  "iec": "",
  "udyam": "",

  "upi": "",

  "bankName": "",
  "accountNumber": "",
  "ifsc": "",
  "branch": "",

  "certifications": [],
  "licenses": [],

  "notes": ""
}"""


def _build_prompt(ocr_text: str) -> str:
    return f"""You are an expert AI Business Card Parser.

Extract all information from the OCR text below, which may combine the
front and back of one business card.

Rules:

- Return ONLY valid JSON.
- Do not return markdown, code fences, comments, explanations, or any text outside the JSON.
- If a field is missing, return an empty string ("") or empty array ([]). Never omit a key.
- Never invent information that is not present or strongly implied by the OCR text.
- Correct only obvious OCR mistakes when you are highly confident (for example "0" vs "O", "1" vs "l").
- Correct obvious OCR word splits where a phrase has been broken across multiple lines.
  Example:
  "Move-In/Move"
  "Out Cleaning"
  should become
  "Move-In/Move-Out Cleaning".
- Remove duplicate values from every array.
- Preserve the original capitalization of names, companies, and brands whenever possible.

Contact Information

- Extract every email address.
- Extract every phone number.
- Separate Mobile numbers from Office/Landline/Fax when labels exist.
- If a phone number has no label, place it inside "phones".
- Extract every website.
- Extract complete postal addresses.

Social Media

Extract every social profile if available:

- LinkedIn
- Facebook
- Instagram
- Twitter/X
- WhatsApp
- Telegram
- YouTube

Business Information

Extract:

- Name
- First Name
- Last Name
- Designation
- Department
- Company
- Brand

Also extract:

- Services
- Products
- Certifications
- Licenses

Business / Financial Identifiers

Extract whenever present:

- GST Number
- PAN Number
- CIN
- MSME
- Udyam
- IEC
- UPI
- Bank Name
- Account Number
- IFSC
- Branch

Normalize Common Labels

Always normalize these labels before returning JSON.

Examples:

GSTIN
GST No
GST Registration Number
→ gstNumber

PAN
PAN No
Permanent Account Number
→ panNumber

MSME Registration
Udyam Registration
→ udyam

IEC Code
→ iec

IFSC Code
→ ifsc

A/C No
Account No
Account Number
→ accountNumber

Bank
Bank Name
→ bankName

Additional Fields

If the business card contains any information that is NOT already part of the schema below, DO NOT discard it.

Examples include (but are not limited to):

- VAT Number
- Tax ID
- FSSAI Number
- Drug License
- Food License
- Import Export License
- Registration Number
- Membership Number
- ISO Certification
- Awards
- Distributor Code
- Dealer Code
- Employee ID
- Vendor ID
- Franchise Code
- QR Payment ID
- Swift Code
- IBAN
- Routing Number
- Company Registration Number
- Any other business-related information

Return these as additional top-level JSON keys using:

- short
- consistent
- human-readable
- camelCase names

Do not discard any business information.

Notes

Only use "notes" for information that genuinely does not belong to any structured field.

IMPORTANT:
Do NOT discard any OCR text.

If any line of OCR text cannot be mapped to a structured field or an additional top-level field, append it to the "notes" field exactly as it appears.

Every OCR line must appear somewhere in the output:
- structured fields,
- additional top-level fields,
- or notes.

Never lose any OCR text.

Return JSON exactly matching the schema below.
Additional top-level keys are allowed whenever the business card contains extra information.

{_RESPONSE_SCHEMA_EXAMPLE}

OCR TEXT

{ocr_text}
"""


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type(_GeminiTransientError),
)
def _call_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
    timeout_seconds: float,
    temperature: float,
) -> str:
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                http_options=genai_types.HttpOptions(
                    timeout=int(timeout_seconds * 1000)
                ),
            ),
        )

        text = (response.text or "").strip()

        # Debug-only, and truncated - the raw text may contain the card
        # holder's personal contact details, so it never goes to stdout at
        # INFO level or above, and never in full even at DEBUG.
        logger.debug(
            "gemini_response_received",
            extra={"responseLength": len(text), "preview": text[:120]},
        )

    except Exception as exc:
        message = str(exc).lower()
        transient_markers = (
            "timeout", "deadline", "429", "500", "502", "503", "504", "unavailable"
        )
        if any(marker in message for marker in transient_markers):
            raise _GeminiTransientError(str(exc)) from exc
        raise

    if not text:
        raise _GeminiTransientError("Gemini returned an empty response.")

    return text


@lru_cache(maxsize=4)
def _get_client(api_key: str) -> genai.Client:
    """
    genai.Client is safe to reuse across requests and constructing one
    isn't free (it sets up HTTP connection pooling internally), so it's
    cached per API key instead of rebuilt on every single scan.
    """
    return genai.Client(api_key=api_key)


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE_RE.sub("", text).strip()


def _to_parsed_data(extraction: GeminiCardExtraction) -> ParsedData:
    dynamic_fields: Dict[str, Any] = {}

    if extraction.department:
        dynamic_fields["Department"] = extraction.department
    if extraction.brand:
        dynamic_fields["Brand"] = extraction.brand
    if extraction.firstName:
        dynamic_fields["FirstName"] = extraction.firstName
    if extraction.lastName:
        dynamic_fields["LastName"] = extraction.lastName
    if extraction.mobiles:
        dynamic_fields["Mobiles"] = extraction.mobiles
    if extraction.faxNumbers:
        dynamic_fields["FaxNumbers"] = extraction.faxNumbers
    if extraction.services:
        dynamic_fields["Services"] = extraction.services
    if extraction.products:
        dynamic_fields["Products"] = extraction.products

    # "notes" is explicitly the catch-all for text that doesn't belong to
    # any structured field (see the prompt) - slogans, taglines, stray
    # descriptions. It belongs in uncategorizedText, not as its own
    # first-class dynamic field.
    uncategorized_text: list[str] = []
    if extraction.notes:
        uncategorized_text.append(extraction.notes)

    if extraction.gstNumber:
        dynamic_fields["GST Number"] = extraction.gstNumber

    if extraction.panNumber:
        dynamic_fields["PAN Number"] = extraction.panNumber

    if extraction.cin:
        dynamic_fields["CIN"] = extraction.cin

    if extraction.msme:
        dynamic_fields["MSME"] = extraction.msme

    if extraction.iec:
        dynamic_fields["IEC"] = extraction.iec

    if extraction.udyam:
        dynamic_fields["Udyam"] = extraction.udyam

    if extraction.upi:
        dynamic_fields["UPI"] = extraction.upi

    if extraction.bankName:
        dynamic_fields["Bank Name"] = extraction.bankName

    if extraction.accountNumber:
        dynamic_fields["Account Number"] = extraction.accountNumber

    if extraction.ifsc:
        dynamic_fields["IFSC"] = extraction.ifsc

    if extraction.branch:
        dynamic_fields["Branch"] = extraction.branch

    if extraction.certifications:
        dynamic_fields["Certifications"] = extraction.certifications

    if extraction.licenses:
        dynamic_fields["Licenses"] = extraction.licenses

    address_parts = extraction.address
    if address_parts.street:
        dynamic_fields["Street"] = address_parts.street
    if address_parts.area:
        dynamic_fields["Area"] = address_parts.area
    if address_parts.city:
        dynamic_fields["City"] = address_parts.city
    if address_parts.state:
        dynamic_fields["State"] = address_parts.state
    if address_parts.country:
        dynamic_fields["Country"] = address_parts.country
    if address_parts.postalCode:
        dynamic_fields["PostalCode"] = address_parts.postalCode

    socials = extraction.socials
    if socials.linkedin:
        dynamic_fields["LinkedIn"] = socials.linkedin
    if socials.facebook:
        dynamic_fields["Facebook"] = socials.facebook
    if socials.instagram:
        dynamic_fields["Instagram"] = socials.instagram
    if socials.twitter:
        dynamic_fields["Twitter"] = socials.twitter
    if socials.whatsapp:
        dynamic_fields["WhatsApp"] = socials.whatsapp
    if socials.telegram:
        dynamic_fields["Telegram"] = socials.telegram
    if socials.youtube:
        dynamic_fields["YouTube"] = socials.youtube

    # Anything Gemini returned that isn't part of the known schema above
    # (VAT Number, Tax ID, FSSAI, Import Export License, Registration
    # Number, Awards, ISO Certification, etc.) is preserved here instead
    # of being discarded. `model_extra` is populated because
    # GeminiCardExtraction uses ConfigDict(extra="allow").
    extra_fields = extraction.model_extra or {}
    for key, value in extra_fields.items():
        if value in (None, "", [], {}):
            continue
        if key not in dynamic_fields:
            dynamic_fields[key] = value

    if uncategorized_text:
        dynamic_fields["uncategorizedText"] = uncategorized_text

    all_phones = extraction.phones + extraction.mobiles

    return ParsedData(
        name=extraction.name,
        designation=extraction.designation,
        company=extraction.company,
        email=extraction.emails[0] if extraction.emails else "",
        phones=_dedupe_preserve_order(all_phones),
        website=extraction.website or (extraction.websites[0] if extraction.websites else ""),
        address=address_parts.full,
        emails=_dedupe_preserve_order(extraction.emails),
        websites=_dedupe_preserve_order(extraction.websites or ([extraction.website] if extraction.website else [])),
        dynamicFields=dynamic_fields,
    )


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen


def parse_with_gemini(ocr_text: str, settings: Settings) -> Optional[ParsedData]:
    """
    Returns a validated ParsedData on success, or None if Gemini is not
    configured / failed after all retries / returned something that could
    not be validated. Callers MUST treat None as "fall back to regex" -
    this function never raises.
    """

    if not settings.GEMINI_API_KEY:
        logger.warning("gemini_not_configured_falling_back")
        return None

    if not ocr_text or not ocr_text.strip():
        # Nothing for Gemini to work with - skip the network call entirely.
        return None

    try:
        client = _get_client(settings.GEMINI_API_KEY)
        prompt = _build_prompt(ocr_text)
        raw_text = _call_gemini(
            client=client,
            model=settings.GEMINI_MODEL,
            prompt=prompt,
            timeout_seconds=settings.GEMINI_TIMEOUT_SECONDS,
            temperature=settings.GEMINI_TEMPERATURE,
        )
    except Exception as exc:  # noqa: BLE001 - never let Gemini crash the request
        logger.error("gemini_call_failed_after_retries", extra={"error": str(exc)})
        return None

    cleaned = _strip_code_fences(raw_text)


    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("gemini_response_not_json", extra={"error": str(exc), "raw_preview": cleaned[:200]})
        return None

    try:
        extraction = GeminiCardExtraction.model_validate(payload)
    except ValidationError as exc:
        logger.error("gemini_response_failed_validation", extra={"error": str(exc)})
        return None

    return _to_parsed_data(extraction)