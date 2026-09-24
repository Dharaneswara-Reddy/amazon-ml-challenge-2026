"""Shared fixtures: a throwaway repo root with a tiny config (never touches the real dataset)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest
import yaml

from ber.config import Config, load_config
from ber.duck import connect

SOURCE_HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def write_text(path: Path, text: str) -> Path:
    """Write UTF-8 ``text`` to ``path`` byte-exactly (no newline translation) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A fake repo root containing dataset/ and code/ plus a small base config."""
    (tmp_path / "dataset" / "train").mkdir(parents=True)
    (tmp_path / "dataset" / "test").mkdir(parents=True)
    (tmp_path / "code").mkdir()
    cfg = {
        "seed": 1,
        "paths": {
            "data_dir": "dataset",
            "cache_dir": "artifacts/cache",
            "runs_dir": "artifacts/runs",
            "tmp_dir": "artifacts/tmp",
            "output_dir": "output",
            "validator": "student_resource/utils/validate_submission.py",
        },
        "engine": {"duckdb_memory_limit": "256MB", "duckdb_threads": 1},
    }
    write_text(tmp_path / "config.yaml", yaml.safe_dump(cfg))
    return tmp_path


@pytest.fixture
def cfg(root: Path) -> Config:
    """Config loaded from the fake root."""
    return load_config(root / "config.yaml", root=root)


@pytest.fixture
def con(cfg: Config) -> duckdb.DuckDBPyConnection:
    """A limited DuckDB connection, closed after the test."""
    c = connect(cfg)
    yield c
    c.close()
