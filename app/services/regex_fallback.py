"""
=============================================================================
Regex fallback parser
=============================================================================
Used ONLY when Gemini is unreachable, misconfigured, or returns something
that fails validation after every retry. This is intentionally a Python
mirror of the field-detection conventions already established in
src/modules/ocr/ocrFieldExtractor.js (labeled-line parsing, phone/email/url
regexes, PAN/GST/UPI detection, social-domain detection, dynamicFields for
anything unrecognized) so that whichever engine actually produced a given
card's data, the *shape* of dynamicFields keys stays consistent
(Services, Products, LinkedIn, Instagram, Facebook, Twitter, WhatsApp,
Telegram, PAN, GST, UPI, ...).

Never raises. Worst case, it returns an almost-empty ParsedData.
=============================================================================
"""

from __future__ import annotations

import re
from typing import Dict, List

from app.models.schemas import ParsedData

PHONE_REGEX = re.compile(r"\+?\d[\d\s().-]{5,16}\d")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
URL_REGEX = re.compile(
    r"\b((?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?(?:/[^\s,]*)?)\b"
)
PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
GST_REGEX = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z0-9]{1}Z[A-Z0-9]{1}\b")

UPI_HANDLES = {
    "okaxis", "okhdfcbank", "okicici", "oksbi", "okbizaxis",
    "ybl", "ibl", "axl", "paytm", "apl", "upi", "fbl", "idfcbank",
    "airtel", "jio", "freecharge", "yesbank", "rbl", "kotak",
}

SOCIAL_DOMAINS = [
    ("LinkedIn", re.compile(r"linkedin\.com/[^\s,]+", re.I)),
    ("Instagram", re.compile(r"instagram\.com/[^\s,]+", re.I)),
    ("Facebook", re.compile(r"facebook\.com/[^\s,]+", re.I)),
    ("Twitter", re.compile(r"(?:twitter\.com|x\.com)/[^\s,]+", re.I)),
    ("WhatsApp", re.compile(r"(?:wa\.me|api\.whatsapp\.com)/[^\s,]+", re.I)),
    ("Telegram", re.compile(r"(?:t\.me|telegram\.me)/[^\s,]+", re.I)),
]

STRUCTURED_LABELS = {
    "pan": "PAN", "pan no": "PAN", "pan number": "PAN",
    "gst": "GST", "gstin": "GST", "gst no": "GST", "gst number": "GST",
    "services": "Services", "service": "Services",
    "products": "Products", "product": "Products",
    "upi": "UPI", "upi id": "UPI",
    "whatsapp": "WhatsApp", "telegram": "Telegram",
    "instagram": "Instagram", "facebook": "Facebook",
    "twitter": "Twitter", "linkedin": "LinkedIn",
    "website": "__skip_website", "email": "__skip_email",
    "phone": "__skip_phone", "mobile": "__skip_phone",
    "name": "__skip_name", "designation": "__skip_designation",
    "company": "__skip_company", "address": "__skip_address",
    "department": "__skip_department",
}

LIST_FIELDS = {"Services", "Products"}

LABEL_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /_-]{1,30})\s*[:\-]\s*(.+)$")


def _dedupe(items: List[str]) -> List[str]:
    seen: List[str] = []
    for item in items:
        v = (item or "").strip()
        if v and v not in seen:
            seen.append(v)
    return seen


def _split_list(value: str) -> List[str]:
    return [v.strip() for v in re.split(r"[,|/]|(?:\s{2,})", value) if v.strip()]


def parse_with_regex(raw_text: str) -> ParsedData:
    text = raw_text or ""
    lines = text.splitlines()

    structured: Dict[str, str] = {}
    dynamic_fields: Dict[str, object] = {}
    # Lines that don't match a "Label: value" shape at all - these are the
    # ones genuinely at risk of being silently dropped (slogans, taglines,
    # stray descriptions). Entity-bearing lines (an email/phone/URL/PAN/GST
    # on its own line) are filtered back out below once those extractors
    # run, since that content is already captured in a structured field.
    unclassified_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        match = LABEL_LINE_RE.match(line)
        if not match:
            unclassified_lines.append(stripped)
            continue

        raw_label, value = match.group(1).strip(), match.group(2).strip()
        if not value:
            continue

        mapped = STRUCTURED_LABELS.get(raw_label.lower())
        if mapped and mapped.startswith("__skip_"):
            structured[mapped.replace("__skip_", "")] = value
            continue
        if mapped:
            if mapped in LIST_FIELDS:
                dynamic_fields[mapped] = _dedupe(
                    [*dynamic_fields.get(mapped, []), *_split_list(value)]
                )
            else:
                dynamic_fields[mapped] = value
            continue

        dynamic_fields[raw_label] = value

    # --- Emails / UPI -------------------------------------------------------
    all_matches = _dedupe(EMAIL_REGEX.findall(text))
    emails: List[str] = []
    upi_ids: List[str] = []
    for candidate in all_matches:
        domain = candidate.split("@")[-1].lower() if "@" in candidate else ""
        if domain in UPI_HANDLES:
            upi_ids.append(candidate)
        else:
            emails.append(candidate)

    if upi_ids:
        dynamic_fields["UPI"] = ", ".join(_dedupe(upi_ids))

    text_without_emails = text
    for token in [*emails, *upi_ids]:
        text_without_emails = text_without_emails.replace(token, " ")

    # --- Social links ---------------------------------------------------------
    social_urls_found: List[str] = []
    for key, pattern in SOCIAL_DOMAINS:
        m = pattern.search(text)
        if m:
            social_urls_found.append(m.group(0))
            dynamic_fields.setdefault(key, m.group(0))

    # --- Phones ---------------------------------------------------------------
    phone_candidates = []
    for p in PHONE_REGEX.findall(text):
        cleaned = re.sub(r"[\s.]", "", p.strip())
        digits = re.sub(r"\D", "", cleaned)
        if 7 <= len(digits) <= 13:
            phone_candidates.append(cleaned)

    seen_last10 = set()
    phones: List[str] = []
    for p in phone_candidates:
        digits = re.sub(r"\D", "", p)
        last10 = digits[-10:]
        if last10 in seen_last10:
            continue
        seen_last10.add(last10)
        phones.append(p)

    # --- Websites --------------------------------------------------------------
    website_candidates = [
        url for url in URL_REGEX.findall(text_without_emails)
        if "@" not in url and not any(s in url or url in s for s in social_urls_found)
    ]
    websites = _dedupe(website_candidates)

    # --- PAN / GST ---------------------------------------------------------------
    pan_matches = _dedupe(PAN_REGEX.findall(text))
    gst_matches = _dedupe(GST_REGEX.findall(text))
    gst_embedded_pans = {g[2:11] for g in gst_matches}
    standalone_pans = [p for p in pan_matches if p not in gst_embedded_pans]

    if standalone_pans and "PAN" not in dynamic_fields:
        dynamic_fields["PAN"] = standalone_pans[0]
    if gst_matches and "GST" not in dynamic_fields:
        dynamic_fields["GST"] = gst_matches[0]

    if "department" in structured:
        dynamic_fields.setdefault("Department", structured.pop("department"))

    # A line only counts as genuinely "uncategorized" if it isn't just an
    # email/phone/URL/PAN/GST that's already sitting in a structured field
    # above - otherwise every contact-detail line would be duplicated here.
    # Phone numbers are compared by digits only, since the captured form
    # has whitespace/punctuation stripped out but the original line won't.
    phone_digit_sets = [re.sub(r"\D", "", p) for p in phones]
    other_tokens = [*emails, *upi_ids, *websites, *pan_matches, *gst_matches]

    def _already_captured(line: str) -> bool:
        if any(token and token in line for token in other_tokens):
            return True
        line_digits = re.sub(r"\D", "", line)
        return any(digits and digits in line_digits for digits in phone_digit_sets)

    uncategorized_text = _dedupe(
        [line for line in unclassified_lines if not _already_captured(line)]
    )
    if uncategorized_text:
        dynamic_fields["uncategorizedText"] = uncategorized_text

    return ParsedData(
        name=structured.get("name", ""),
        designation=structured.get("designation", ""),
        company=structured.get("company", ""),
        email=emails[0] if emails else structured.get("email", ""),
        phones=phones,
        website=websites[0] if websites else structured.get("website", ""),
        address=structured.get("address", ""),
        emails=emails,
        websites=websites,
        dynamicFields=dynamic_fields,
    )
