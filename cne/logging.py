"""Structured, timestamped logging for pipeline runs.

CNE deliberately avoids bare ``print`` in library code: runs are long, they are
restarted, and an operator needs to be able to tell which stage produced which
number.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def configure(level: str = "INFO", stream=None) -> None:
    """Install the CNE root handler exactly once."""
    global _CONFIGURED
    root = logging.getLogger("cne")
    if _CONFIGURED:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(f"cne.{name}")


@contextmanager
def stage(name: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    """Time a pipeline stage and log a one-line summary on exit."""
    log = logger or get_logger("stage")
    log.info("stage=%s status=start", name)
    started = time.perf_counter()
    info: dict = {"name": name}
    try:
        yield info
    except Exception as exc:  # pragma: no cover - defensive
        log.error("stage=%s status=failed elapsed=%.1fs error=%s", name, time.perf_counter() - started, exc)
        raise
    else:
        log.info("stage=%s status=ok elapsed=%.1fs %s", name, time.perf_counter() - started, info.get("summary", ""))
