"""
=============================================================================
Parser orchestration
=============================================================================
Ties together: merged OCR text -> Gemini (primary) -> regex (fallback) ->
normalized ParsedData ready to hand back to Node.

  Raw OCR Text -> Gemini AI -> Structured Parsed Data

Regex is explicitly a fallback ONLY - it runs when Gemini is unconfigured,
unreachable, or returns something invalid, per the requirement "Regex may
only be used as a fallback if Gemini fails."
=============================================================================
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

from app.config import Settings
from app.logging_config import get_logger
from app.models.schemas import ParsedData
from app.services.gemini_service import parse_with_gemini
from app.services.regex_fallback import parse_with_regex

logger = get_logger(__name__)

_ALNUM_RE = re.compile(r"[A-Za-z0-9]")


def has_meaningful_text(text: str, min_chars: int) -> bool:
    """
    True once the text contains at least `min_chars` alphanumeric
    characters. Filters out OCR noise (a misread speck, stray punctuation)
    so it can't trigger a full Gemini call on its own.
    """
    if not text:
        return False
    return len(_ALNUM_RE.findall(text)) >= min_chars


def build_parsed_data(merged_text: str, settings: Settings) -> Tuple[ParsedData, str]:
    """
    Returns (parsed_data, provider_label).
    provider_label is one of: "gemini", "regex-fallback", "empty".

    Gemini (and the regex fallback) are both skipped entirely - no network
    call, no parsing effort - whenever the OCR text has no meaningful
    content. This is what makes it safe for the pipeline to call this
    unconditionally even for a QR-only or barcode-only card: there's simply
    nothing here for either engine to work with.
    """

    if not has_meaningful_text(merged_text, settings.OCR_MIN_MEANINGFUL_CHARS):
        return ParsedData(), "empty"

    gemini_result = parse_with_gemini(merged_text, settings)
    if gemini_result is not None:
        return _normalize(gemini_result), "gemini"

    logger.info("falling_back_to_regex_parser")
    regex_result = parse_with_regex(merged_text)
    return _normalize(regex_result), "regex-fallback"


def _normalize(parsed: ParsedData) -> ParsedData:
    """
    Final cleanup pass shared by both engines: trims whitespace, dedupes
    arrays once more (defense in depth), and drops obviously-junk single
    characters that occasionally slip through OCR noise.
    """

    parsed.name = parsed.name.strip()
    parsed.designation = parsed.designation.strip()
    parsed.company = parsed.company.strip()
    parsed.email = parsed.email.strip()
    parsed.website = _normalize_website(parsed.website.strip())
    parsed.address = " ".join(parsed.address.split())

    parsed.phones = _dedupe([p.strip() for p in parsed.phones if p and p.strip()])
    parsed.emails = _dedupe([e.strip() for e in parsed.emails if e and e.strip()])
    parsed.websites = _dedupe(
        [_normalize_website(w.strip()) for w in parsed.websites if w and w.strip()]
    )

    cleaned_dynamic: Dict[str, object] = {}
    for key, value in parsed.dynamicFields.items():
        if isinstance(value, list):
            deduped = _dedupe([str(v).strip() for v in value if str(v).strip()])
            if deduped:
                cleaned_dynamic[key] = deduped
        elif isinstance(value, str):
            if value.strip():
                cleaned_dynamic[key] = value.strip()
        elif value not in (None, "", []):
            cleaned_dynamic[key] = value

    parsed.dynamicFields = _dedupe_against_structured(parsed, cleaned_dynamic)

    return parsed


def _structured_value_set(parsed: ParsedData) -> set[str]:
    """
    Every value already living in a first-class structured field,
    normalized for comparison (casefolded, whitespace-collapsed).
    """
    values: list[str] = [
        parsed.name, parsed.designation, parsed.company, parsed.email,
        parsed.website, parsed.address,
    ]
    values.extend(parsed.phones)
    values.extend(parsed.emails)
    values.extend(parsed.websites)
    return {_normalize_for_compare(v) for v in values if v and v.strip()}


def _normalize_for_compare(value: str) -> str:
    return " ".join(str(value).split()).strip().casefold()


def _dedupe_against_structured(parsed: ParsedData, dynamic_fields: Dict[str, object]) -> Dict[str, object]:
    """
    Ensures a value that already exists in a structured field (Name,
    Company, Phone, Email, Website, Address, ...) never also appears
    inside dynamicFields, and that the same value never appears twice
    across two different dynamicFields keys either. Every extracted value
    should exist exactly once in the final JSON.
    """
    structured_values = _structured_value_set(parsed)
    seen_dynamic_values: set[str] = set()
    result: Dict[str, object] = {}

    for key, value in dynamic_fields.items():
        if isinstance(value, list):
            kept: list[str] = []
            for item in value:
                normalized = _normalize_for_compare(item)
                if not normalized or normalized in structured_values or normalized in seen_dynamic_values:
                    continue
                seen_dynamic_values.add(normalized)
                kept.append(item)
            if kept:
                result[key] = kept
        else:
            normalized = _normalize_for_compare(value) if isinstance(value, str) else ""
            if isinstance(value, str):
                if not normalized or normalized in structured_values or normalized in seen_dynamic_values:
                    continue
                seen_dynamic_values.add(normalized)
            result[key] = value

    return result


def _normalize_website(url: str) -> str:
    if not url:
        return url
    return url.rstrip("/.,")


def _dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen
