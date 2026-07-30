"""
Small helper for measuring how long each pipeline stage took, in
milliseconds, without littering the pipeline code with time.time() calls.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator


@contextmanager
def stage_timer(bucket: Dict[str, float], key: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        bucket[key] = round((time.perf_counter() - start) * 1000, 2)
