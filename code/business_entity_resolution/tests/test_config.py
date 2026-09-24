"""Tests for ber.config."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_text

from ber.config import DEFAULT_CONFIG, config_hash, deep_merge, find_repo_root, load_config


def test_paths_resolved_against_root(cfg, root: Path) -> None:
    """Relative paths become absolute under the root; other sections stay as written."""
    assert cfg.paths.cache_dir == (root / "artifacts" / "cache").resolve()
    assert cfg.engine.duckdb_threads == 1
    assert cfg["engine"]["duckdb_memory_limit"] == "256MB"
    assert cfg.root == root.resolve()


def test_frozen(cfg) -> None:
    """Config and nested nodes reject mutation."""
    with pytest.raises(AttributeError):
        cfg.seed = 2
    with pytest.raises(AttributeError):
        cfg.engine.duckdb_threads = 8
    with pytest.raises(TypeError):
        cfg.engine["duckdb_threads"] = 8  # type: ignore[index]


def test_override_deep_merge_and_hash(root: Path) -> None:
    """Override merges leaves recursively and changes the hash; the hash is 12 hex chars and stable."""
    base = load_config(root / "config.yaml", root=root)
    write_text(root / "over.yaml", "engine:\n  duckdb_threads: 3\nnew_key: [1, 2]\n")
    over = load_config(root / "config.yaml", root / "over.yaml", root=root)
    assert over.engine.duckdb_threads == 3
    assert over.engine.duckdb_memory_limit == "256MB"
    assert over.new_key == (1, 2)
    h = config_hash(base)
    assert len(h) == 12 and int(h, 16) >= 0
    assert h == config_hash(load_config(root / "config.yaml", root=root))
    assert h != config_hash(over)


def test_hash_independent_of_root(root: Path, tmp_path_factory) -> None:
    """The hash fingerprints unresolved YAML, so another checkout location gives the same hash."""
    other = tmp_path_factory.mktemp("other")
    assert config_hash(load_config(root / "config.yaml", root=root)) == config_hash(
        load_config(root / "config.yaml", root=other)
    )


def test_deep_merge_does_not_mutate() -> None:
    """deep_merge returns a new dict and leaves inputs untouched."""
    a = {"x": {"y": 1, "z": 2}}
    b = {"x": {"y": 5}}
    assert deep_merge(a, b) == {"x": {"y": 5, "z": 2}}
    assert a == {"x": {"y": 1, "z": 2}}


def test_find_repo_root(root: Path) -> None:
    """Walking up from a nested dir finds the dir holding dataset/ and code/."""
    nested = root / "code" / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == root.resolve()


def test_default_config_loads(root: Path) -> None:
    """The package's configs/base.yaml parses and has the sections the pipeline needs."""
    cfg = load_config(DEFAULT_CONFIG, root=root)
    assert {"paths", "engine"} <= set(cfg)
    assert cfg.paths.tmp_dir.is_absolute()
