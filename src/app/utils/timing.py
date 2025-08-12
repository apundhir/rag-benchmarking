from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager that yields a dict where 'elapsed_ms' is set on exit."""
    data: dict[str, float] = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield data
    finally:
        end = time.perf_counter()
        data["elapsed_ms"] = (end - start) * 1000.0


