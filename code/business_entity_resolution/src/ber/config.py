"""Configuration loading: base YAML + optional override, frozen and attribute-accessible."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "base.yaml"
PATHS_SECTION = "paths"


class FrozenNode(Mapping[str, Any]):
    """Read-only mapping whose keys are also readable as attributes; nested dicts become FrozenNodes."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        """Wrap ``data`` (deep-frozen: nested mappings wrapped, lists turned into tuples)."""
        object.__setattr__(self, "_data", {k: _freeze(v) for k, v in data.items()})

    def __getitem__(self, key: str) -> Any:
        """Return the value stored under ``key``."""
        return self._data[key]

    def __getattr__(self, key: str) -> Any:
        """Return the value stored under ``key`` as an attribute; AttributeError if absent."""
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"config has no key {key!r} (keys: {sorted(self._data)})") from None

    def __setattr__(self, key: str, value: Any) -> None:
        """Reject mutation."""
        raise AttributeError("config is frozen")

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of keys."""
        return len(self._data)

    def __repr__(self) -> str:
        """Return a dict-like representation."""
        return f"{type(self).__name__}({self.to_dict()!r})"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, mutable deep copy (FrozenNode -> dict, tuple -> list, Path -> str)."""
        return {k: _thaw(v) for k, v in self._data.items()}


class Config(FrozenNode):
    """Frozen pipeline config. ``paths.*`` values are absolute Paths resolved against ``root``.

    ``raw`` holds the merged YAML before path resolution; it is what ``config_hash`` fingerprints,
    so the hash does not depend on the machine or checkout location.
    """

    __slots__ = ("_raw", "_root")

    def __init__(self, raw: Mapping[str, Any], root: Path) -> None:
        """Build a config from the merged YAML dict ``raw`` and resolve its ``paths`` against ``root``."""
        resolved = copy.deepcopy(dict(raw))
        paths = resolved.get(PATHS_SECTION) or {}
        resolved[PATHS_SECTION] = {k: _resolve(root, v) for k, v in paths.items()}
        super().__init__(resolved)
        object.__setattr__(self, "_raw", FrozenNode(raw))
        object.__setattr__(self, "_root", root)

    @property
    def root(self) -> Path:
        """Repository root that relative paths were resolved against."""
        return self._root

    @property
    def raw(self) -> dict[str, Any]:
        """Merged YAML (unresolved paths) as a plain dict."""
        return self._raw.to_dict()


def _freeze(value: Any) -> Any:
    """Recursively convert dicts to FrozenNode and lists to tuples."""
    if isinstance(value, FrozenNode):
        return value
    if isinstance(value, Mapping):
        return FrozenNode(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Inverse of ``_freeze`` producing JSON-friendly plain Python objects."""
    if isinstance(value, FrozenNode):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve(root: Path, value: Any) -> Any:
    """Resolve a path-like string against ``root``; non-strings pass through."""
    if not isinstance(value, str | Path):
        return value
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict: ``override`` merged into ``base`` recursively (override wins on leaves)."""
    out = copy.deepcopy(dict(base))
    for key, val in override.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: cwd, then this file) to the dir with ``dataset/`` and ``code/``."""
    starts = [start] if start is not None else [Path.cwd(), Path(__file__).resolve().parent]
    for s in starts:
        for d in [s.resolve(), *s.resolve().parents]:
            if (d / "dataset").is_dir() and (d / "code").is_dir():
                return d
    raise FileNotFoundError(f"no repo root (dir containing dataset/ and code/) above {starts}")


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file that must contain a mapping (empty file -> {})."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def load_config(
    path: Path | str | None = None,
    override: Path | str | None = None,
    root: Path | str | None = None,
) -> Config:
    """Load ``path`` (default: package configs/base.yaml), deep-merge ``override`` if given, resolve paths.

    ``root`` defaults to the repo root found by ``find_repo_root``. Returns a frozen ``Config``.
    """
    raw = _read_yaml(Path(path) if path else DEFAULT_CONFIG)
    if override:
        raw = deep_merge(raw, _read_yaml(Path(override)))
    root_path = Path(root).resolve() if root else find_repo_root()
    return Config(raw, root_path)


def config_hash(cfg: Config | Mapping[str, Any]) -> str:
    """Return the first 12 hex chars of sha256 over the canonical JSON of the (unresolved) config."""
    if isinstance(cfg, Config):
        data: Any = cfg.raw
    elif isinstance(cfg, FrozenNode):
        data = cfg.to_dict()
    else:
        data = cfg
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]
