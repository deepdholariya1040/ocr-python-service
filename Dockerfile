# =============================================================================
# Business Card OCR Microservice - production image (PaddleOCR)
# =============================================================================
# Multi-stage build:
#   1. "builder" installs build-time-only tooling and resolves/installs
#      every Python dependency into an isolated virtualenv.
#   2. The final stage copies just that virtualenv + the app code onto a
#      clean slim base with only the *runtime* system libraries PaddleOCR
#      actually needs - keeping the final image meaningfully smaller than
#      if build tooling shipped in the runtime layer too.
#
# PaddleOCR's model weights are downloaded and baked into the image at BUILD
# time (see the RUN step near the bottom, before USER switches to appuser).
# This is deliberate: without it, the FIRST request after every deploy would
# have to download the detection/recognition models over the network before
# it could return - unpredictable latency, and a hard failure if the
# container's egress happens to be restricted at runtime. Baking them in
# means the container is fully self-contained and ready the instant it
# passes its health check.
#
# Expect ~1.3-1.8GB for the final image (paddlepaddle + its ML runtime are
# meaningfully heavier than a classical OCR engine - there is no realistic
# path to a sub-1GB image while keeping PaddleOCR, which is a firm
# requirement here).
# =============================================================================

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-time-only system deps: gcc/g++ cover the rare case a dependency has
# no prebuilt wheel for this exact Python/arch and needs to compile from
# source. None of this ships in the final image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

# Runtime system libraries actually required by paddlepaddle / the OpenCV
# backend PaddleOCR uses internally, plus Pillow's JPEG/PNG codecs.
# No Tesseract here - PaddleOCR is a self-contained deep-learning OCR
# engine and does not use or need it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libjpeg62-turbo \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY app ./app

# Runs as a non-root user, standard hardening. Created before the model
# pre-download step below so the downloaded weights land under this user's
# home directory with the right ownership from the start (no runtime chown
# needed).
RUN useradd --create-home --shell /bin/false appuser
ENV HOME=/home/appuser
USER appuser

# Bakes the PaddleOCR model weights for the default language into the
# image at build time (see rationale at the top of this file). If
# OCR_LANGUAGE is overridden to something else at runtime, that language's
# models will instead download on first use - normal PaddleOCR behavior,
# just no longer the default path.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', use_textline_orientation=False)"

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",5001)}/health', timeout=5)" || exit 1

# Railway injects $PORT at runtime; default to 5001 for local `docker run`.
# WEB_CONCURRENCY controls uvicorn worker count - defaults to 1 because the
# PaddleOCR model is loaded once per process; raising it multiplies memory
# use by that many copies of the model, so only raise it alongside more
# available memory.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-5001} --workers ${WEB_CONCURRENCY:-1}"]
