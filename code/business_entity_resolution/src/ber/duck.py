"""The single place DuckDB connections are created, with memory/thread/spill limits from config."""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import duckdb
import pyarrow as pa

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


TableLike = str | duckdb.DuckDBPyRelation | pa.Table
"""Accepted table inputs: a table/view name in the connection, a relation of the same connection, or a
pyarrow Table (registered zero-copy)."""


def as_view(con: duckdb.DuckDBPyConnection, obj: TableLike, columns: tuple[str, ...]) -> str:
    """Expose ``obj`` as a temporary view named ``_ber_<hex>`` and return that name.

    Raises ValueError if any of ``columns`` is missing. Call ``drop_views`` when done.
    """
    name = f"_ber_{uuid.uuid4().hex[:12]}"
    if isinstance(obj, str):
        con.execute(f'CREATE TEMP VIEW {name} AS SELECT * FROM "{obj.replace(chr(34), chr(34) * 2)}"')
    elif isinstance(obj, duckdb.DuckDBPyRelation):
        obj.create_view(name, replace=True)
    elif isinstance(obj, pa.Table):
        con.register(name, obj)
    else:
        raise TypeError(
            f"unsupported table input {type(obj).__name__}; use a name, relation or pyarrow Table"
        )
    have = {r[0] for r in con.execute(f"DESCRIBE {name}").fetchall()}
    missing = [c for c in columns if c not in have]
    if missing:
        drop_views(con, name)
        raise ValueError(f"input is missing columns {missing}; has {sorted(have)}")
    return name


def drop_views(con: duckdb.DuckDBPyConnection, *names: str) -> None:
    """Drop temporary views/registrations created by ``as_view`` (ignores already-dropped ones)."""
    for name in names:
        try:
            con.unregister(name)
        except duckdb.Error:
            pass
        con.execute(f"DROP VIEW IF EXISTS {name}")
