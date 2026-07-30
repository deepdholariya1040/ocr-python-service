"""
=============================================================================
Structured logging
=============================================================================
Emits single-line JSON logs (Railway, Docker, and every common log
aggregator parse these cleanly). Every log call goes through the standard
`logging` module so third-party libraries (uvicorn, PIL, etc.) are captured
too.

Nothing in this module ever logs an API key, request body, or raw image
bytes - only metadata (request id, durations, sizes, counts, error
messages).
=============================================================================
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Dict

from app.config import get_settings

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    settings = get_settings()

    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root.handlers = [handler]

    # Quiet down noisy third-party loggers unless we're debugging.
    for noisy in ("uvicorn.access", "PIL", "httpx", "google_genai"):
        logging.getLogger(noisy).setLevel(
            settings.LOG_LEVEL.upper() if settings.LOG_LEVEL.upper() == "DEBUG" else "WARNING"
        )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
