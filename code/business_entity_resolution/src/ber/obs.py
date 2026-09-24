"""Observability: key=value logging and a ``stage`` context manager timing work and tracking RSS."""

from __future__ import annotations

import logging
import resource
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_MB = 1024 * 1024
log = logging.getLogger("ber")


def kv(**fields: Any) -> str:
    """Render fields as ``key=value`` pairs (values with spaces are repr-quoted)."""
    parts = []
    for key, val in fields.items():
        if isinstance(val, float):
            text = f"{val:.4g}"
        else:
            text = str(val)
            if not text or any(c.isspace() for c in text):
                text = repr(text)
        parts.append(f"{key}={text}")
    return " ".join(parts)


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> None:
    """Configure the ``ber`` logger to write to stderr and, optionally, to ``log_file`` (appended)."""
    root = logging.getLogger("ber")
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    root.propagate = False


def rss_mb() -> float:
    """Return this process's current resident set size in MB."""
    return psutil.Process().memory_info().rss / _MB


def max_rss_mb() -> float:
    """Return this process's lifetime peak RSS in MB (Linux ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class _RssSampler(threading.Thread):
    """Background thread that samples RSS to find the peak within a stage."""

    def __init__(self, interval: float = 0.2) -> None:
        """Create a daemon sampler polling every ``interval`` seconds."""
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = rss_mb()
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Poll RSS until stopped, keeping the maximum."""
        proc = psutil.Process()
        while not self._stop_event.wait(self.interval):
            self.peak = max(self.peak, proc.memory_info().rss / _MB)

    def stop(self) -> float:
        """Stop sampling and return the peak RSS in MB seen during the stage."""
        self._stop_event.set()
        self.join()
        self.peak = max(self.peak, rss_mb())
        return self.peak


@contextmanager
def stage(name: str, logger: logging.Logger | None = None, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log start/end of a pipeline stage with duration and memory.

    Yields a dict; keys the caller adds (e.g. ``rows``) are logged on completion. On exception logs
    ``status=error`` and re-raises. End line fields: ``seconds``, ``rss_start_mb``, ``rss_end_mb``,
    ``stage_peak_rss_mb`` (sampled), ``peak_rss_delta_mb`` (growth of the process high-water mark).
    """
    lg = logger or log
    extra: dict[str, Any] = {}
    lg.info(kv(event="stage_start", stage=name, **fields))
    rss0, maxrss0 = rss_mb(), max_rss_mb()
    sampler = _RssSampler()
    sampler.start()
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield extra
    except BaseException as exc:
        status = "error"
        extra["error"] = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        peak = sampler.stop()
        lg.log(
            logging.INFO if status == "ok" else logging.ERROR,
            kv(
                event="stage_end",
                stage=name,
                status=status,
                seconds=round(time.perf_counter() - t0, 3),
                rss_start_mb=round(rss0, 1),
                rss_end_mb=round(rss_mb(), 1),
                stage_peak_rss_mb=round(peak, 1),
                peak_rss_delta_mb=round(max_rss_mb() - maxrss0, 1),
                **fields,
                **extra,
            ),
        )
