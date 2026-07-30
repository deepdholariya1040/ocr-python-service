# Business Card OCR Microservice (Python / FastAPI / PaddleOCR)

Production OCR + Gemini AI parsing + QR/Barcode detection service, built to
sit behind the existing Node.js `ocr-saas-backend` with **zero changes**
to that backend. It implements exactly the contract `src/modules/ocr/scan.service.js`
and `src/modules/ocr/ocr.service.js` already expect.

## Pipeline

```
Business Card Image(s) (front and/or back)
        │
        ▼
Image Processing   (EXIF fix, resize, denoise, grayscale, contrast, sharpen,
        │            + non-blocking blur/contrast quality check)
        ▼
PaddleOCR                                                    ──┐
        │                                                      │  QR / Barcode
        ▼                                                      │  detection runs
Raw OCR Text                                                   │  in parallel on
        │                                                      │  the original
        ▼                                                      │  (non-preprocessed)
Smart routing: meaningful text? ── no ──► skip AI, return       │  images via
        │                          QR/barcode data only         │  zxing-cpp
       yes
        ▼
Gemini AI           (structured extraction, retried, validated)  │
        │ (falls back to regex parser if Gemini fails)          │
        ▼                                                       │
Structured Parsed Data  ◄──────────────────────────────────────┘
        │
        ▼
JSON response → Node.js backend → MongoDB
```

## Why these choices

- **PaddleOCR** — a learned-model OCR engine, kept exactly as-is per the
  project requirement. The model is loaded once at process startup
  (`app/services/ocr_engine.py::warmup`) and its weights are baked into
  the Docker image at *build* time (not downloaded at runtime), so a cold
  deploy is ready the moment its health check passes.
- **zxing-cpp, not pyzbar** — one native wheel (no `apt install libzbar0`
  needed) natively covers every required format: QR, Aztec, DataMatrix,
  PDF417, Code128, Code39, EAN-13/8, UPC-A/E, and more, in a single pass.
- **Smart AI routing** — Gemini is only ever called when the OCR text
  contains meaningful content (`OCR_MIN_MEANINGFUL_CHARS`, default 6
  alphanumeric characters). A QR-only or barcode-only card skips Gemini
  and the regex parser entirely and returns the detected code data alone -
  no wasted latency, no wasted API cost. Empty/noise OCR output does the
  same.
- **Gemini first, regex only as fallback** — matches the requirement
  exactly. The regex fallback (`app/services/regex_fallback.py`)
  deliberately mirrors the field-detection conventions already established
  in `src/modules/ocr/ocrFieldExtractor.js` (dynamicFields keys like
  `Services`, `LinkedIn`, `PAN`, `GST`, etc.), so results stay consistent
  regardless of which engine produced them.
- **No duplicate data, nothing lost** — a final normalization pass
  (`app/services/parser.py::_dedupe_against_structured`) strips any value
  out of `dynamicFields` that already exists in a structured field (name,
  phones, emails, website, address, ...), and de-duplicates across
  `dynamicFields` keys too. Anything that can't be confidently mapped to a
  known field (slogans, taglines, stray descriptions) is preserved in
  `dynamicFields.uncategorizedText` instead of being discarded.
- **No backend changes** — `ocr.service.js` already treats the Python
  response generically (`ocrPayload.parsedData`, `.dynamicFields`,
  `.qrCodes`, `.barcodes`, `.provider`, with safe fallbacks throughout).
  This service returns exactly that shape. See "Backend compatibility"
  below for the full field-by-field mapping.

## Project layout

```
app/
  main.py                    FastAPI app, middleware, global exception handling,
                             PaddleOCR startup warmup
  config.py                  All settings, env-driven (pydantic-settings)
  logging_config.py          Structured JSON logging
  api/
    routes_ocr.py            POST /api/ocr/process (runs the pipeline off the
                             event loop thread, with a bounded timeout)
    routes_health.py         GET  /health (liveness + OCR-engine readiness)
  models/
    schemas.py                Pydantic models = the exact backend contract
  services/
    image_processing.py       Preprocessing + non-blocking quality assessment
    ocr_engine.py              PaddleOCR text extraction, startup warmup,
                               bounded concurrency
    qr_barcode_service.py      zxing-cpp QR/barcode detection
    gemini_service.py          Gemini call: retry, timeout, validation
    regex_fallback.py          Fallback parser (Gemini-failure only)
    parser.py                  Orchestrates smart routing → Gemini → fallback
                               → normalize → dedupe
    pipeline.py                Top-level per-request pipeline
  utils/
    exceptions.py              Client-facing error types
    validators.py               Image validation (format/size/corruption)
    timing.py                   Stage-duration helper for logging
  core/
    request_context.py          Request-id middleware
requirements.txt
Dockerfile                     Multi-stage build, PaddleOCR model pre-baked in
railway.json
nixpacks.toml                  (fallback build config, see below)
Procfile
.env.example
.gitignore
```

## Backend compatibility (field-by-field)

The Node side (`ocr.service.js`) reads:

| Node reads                                              | This service returns                          |
|-----------------------------------------------------------|-------------------------------------------------|
| `ocrPayload.frontOCRText` / `.backOCRText` / `.mergedOCRText` | same keys, top-level                          |
| `ocrPayload.parsedData`                                    | `parsedData` object                              |
| `parsedData.phones`                                        | array (merged mobile + landline)                 |
| `parsedData.allEmails \|\| parsedData.emails \|\| [parsedData.email]` | `parsedData.emails` (array) + `.email` (first)   |
| `parsedData.allWebsites \|\| parsedData.websites \|\| [parsedData.website]` | `parsedData.websites` (array) + `.website` (first) |
| `parsedData.dynamicFields`                                  | everything that doesn't map onto a first-class Mongoose field (Department, Services, Products, LinkedIn, Instagram, Facebook, Twitter, WhatsApp, Telegram, YouTube, GST/PAN/CIN/MSME/IEC/Udyam/UPI/bank fields, Street/Area/City/State/Country/PostalCode, `uncategorizedText[]`, ...) |
| `result.qrCodes` / `ocrPayload.qrCodes`                     | `qrCodes: [{type, dataType, content, raw}]`      |
| `result.barcodes` / `ocrPayload.barcodes`                   | `barcodes: [{type, content, raw}]`               |
| `ocrPayload.provider \|\| result.provider`                   | `provider` (e.g. `"gemini-python-ocr-service:gemini"`, `"...:regex-fallback"`, `"...:qr-barcode-only"`, `"...:empty"`) |

`parsedData.name`, `.designation`, `.company`, `.address` map directly onto
the Mongoose sub-schema fields of the same name. Every value in
`dynamicFields` is guaranteed not to duplicate a value already present in a
structured field.

**No backend file needs to change.** `PYTHON_SERVICE_URL` in the existing
`.env` (`http://localhost:5001/api/ocr/process` locally) already points at
exactly the path this service exposes.

## Local setup

Requirements: Python 3.12. No system OCR engine to install - PaddleOCR ships
its models as a Python dependency (they download automatically on first
use if not already cached).

```bash
cd python-ocr-service
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# edit .env and set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)

# Run
uvicorn app.main:app --reload --port 5001
```

Verify it's up:

```bash
curl http://localhost:5001/health
```

Point the Node backend at it — in the backend's own `.env`:

```
PYTHON_SERVICE_URL=http://localhost:5001/api/ocr/process
```

Test a scan directly against the Python service:

```bash
curl -X POST http://localhost:5001/api/ocr/process \
  -F "frontImage=@/path/to/card-front.jpg" \
  -F "backImage=@/path/to/card-back.jpg"
```

## Railway deployment

1. **Push this directory** (`python-ocr-service/`) to its own GitHub repo,
   or as a subdirectory of your monorepo with Railway's "Root Directory"
   setting pointed at it.
2. **Create a new Railway service** from that repo. Railway will detect
   `railway.json`, which pins the builder to `DOCKERFILE` — this is what
   guarantees the PaddleOCR model weights get baked into the image at
   build time (the `nixpacks.toml` in this repo is only a fallback in
   case the builder is ever switched away from Dockerfile, and does not
   pre-bake models).
3. **Set environment variables** on the Railway service (Variables tab),
   using `.env.example` as the reference. At minimum:
   - `GEMINI_API_KEY`
   - `ENVIRONMENT=production`
   Railway sets `PORT` automatically — the Dockerfile's `CMD` already
   reads `$PORT`, don't hardcode it. It also runs automatically via the
   container's `CMD` — no manual `.venv` activation needed in production;
   that's strictly a local-development convenience.
4. **Deploy.** Railway will build the Docker image (including the model
   pre-download step) and run the health check against `/health`
   (configured in `railway.json`, with a generous `start_period`/timeout
   to allow for that build-time cost rather than runtime cost).
5. **Copy the generated public URL** (Settings → Networking → Generate
   Domain, or use Railway's private networking if the Node backend is
   also on Railway) and set it as `PYTHON_SERVICE_URL` on the **Node**
   backend's Railway service:
   ```
   PYTHON_SERVICE_URL=https://<your-python-service>.up.railway.app/api/ocr/process
   ```
6. **Redeploy the Node backend** so it picks up the new env var. No code
   changes needed on that side.

### Expected deployment size

`python:3.12-slim` + PaddleOCR's runtime libs (`libgl1`/`libglib2.0-0`/etc.)
+ Python deps (paddlepaddle, paddleocr, FastAPI, Pillow, google-genai,
zxing-cpp) lands in the ~1.3-1.8GB range. This is meaningfully larger than
a classical-OCR-engine image would be - there's no realistic way to keep
paddlepaddle's ML runtime and land under ~1GB, and PaddleOCR is a firm
requirement for this service.

## Error handling summary

| Situation | Behavior |
|---|---|
| No image supplied at all | `400` |
| Unsupported file type | `415` |
| Corrupt / unreadable image | `422` |
| File exceeds size limit | `413` |
| Pipeline exceeds `REQUEST_TIMEOUT_SECONDS` | `504` |
| OCR finds no text | `200`, empty OCR text fields (not an error) |
| No QR/barcode present | `200`, empty `qrCodes`/`barcodes` arrays |
| Card is QR/barcode-only (no meaningful OCR text) | `200`, `provider` ends in `:qr-barcode-only`, Gemini/regex both skipped |
| Gemini unconfigured, times out, or returns invalid JSON | `200`, `parsedData` from the regex fallback, `provider` reflects `:regex-fallback` |
| Any other unexpected exception | `500`, generic message only — full detail goes to structured logs, never to the client |

## Logging

Every log line is single-line JSON (`app/logging_config.py`) with a
`requestId` correlating every stage of one scan (echoed back to the caller
via the `X-Request-Id` response header). Per-request logs include stage
durations (`front_ocr_ms`, `back_ocr_ms`, `*_qr_barcode_ms`, `parsing_ms`,
total request time) and counts (`qrCodesFound`, `barcodesFound`,
`frontTextLength`, `backTextLength`). API keys, image bytes, and full
Gemini responses are never logged.
