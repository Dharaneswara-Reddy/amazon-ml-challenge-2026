"""The single place DuckDB connections are created, with memory/thread/spill limits from config."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
from pathlib import Path

import duckdb

from ber.config import Config
from ber.obs import kv

log = logging.getLogger("ber.duck")


def _spill_dir(tmp_root: Path) -> Path:
    """Create a per-process spill dir under ``tmp_root`` (removed at exit) so processes never collide."""
    tmp_root.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=f"duckdb-{os.getpid()}-", dir=tmp_root))
    atexit.register(shutil.rmtree, path, True)
    return path


def connect(cfg: Config, database: str | Path = ":memory:") -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with memory_limit, threads, temp_directory and no insertion-order guarantee.

    Values come from ``cfg.engine.duckdb_memory_limit`` / ``duckdb_threads`` and ``cfg.paths.tmp_dir``.
    """
    spill = _spill_dir(Path(cfg.paths.tmp_dir))
    con = duckdb.connect(str(database))
    con.execute(f"SET memory_limit = '{cfg.engine.duckdb_memory_limit}'")
    con.execute(f"SET threads = {int(cfg.engine.duckdb_threads)}")
    con.execute(f"SET temp_directory = '{spill}'")
    con.execute("SET preserve_insertion_order = false")
    log.debug(
        kv(
            event="duckdb_connect",
            memory_limit=cfg.engine.duckdb_memory_limit,
            threads=cfg.engine.duckdb_threads,
            temp_directory=spill,
        )
    )
    return con
