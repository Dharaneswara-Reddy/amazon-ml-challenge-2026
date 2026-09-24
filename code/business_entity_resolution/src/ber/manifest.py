"""Run directories and JSON manifests recording code, config, environment and data fingerprints."""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from ber.config import Config, config_hash
from ber.io import input_fingerprint

PACKAGES = ("ber", "duckdb", "pyarrow", "numpy", "rapidfuzz", "lightgbm", "PyYAML", "psutil")


def utc_stamp() -> str:
    """Return the current UTC time as ``yyyymmddTHHMMSS``."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def git_info(root: Path | None = None) -> dict[str, Any]:
    """Return ``{"sha", "sha7", "dirty"}`` for the repo at ``root``; sha "nogit" if git is unavailable."""

    def run(*args: str) -> str:
        """Run a git command in ``root`` and return stripped stdout."""
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()

    try:
        sha = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return {"sha": "nogit", "sha7": "nogit", "dirty": None}
    return {"sha": sha, "sha7": sha[:7], "dirty": dirty}


def new_run_dir(cfg: Config, name: str) -> Path:
    """Create and return ``<runs_dir>/<UTC stamp>_<gitsha7>_<cfghash8>_<name>``."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "run"
    sha7 = git_info(cfg.root)["sha7"]
    run_dir = Path(cfg.paths.runs_dir) / f"{utc_stamp()}_{sha7}_{config_hash(cfg)[:8]}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def package_versions() -> dict[str, str | None]:
    """Return installed versions of the pipeline's key packages (None if not installed)."""
    out: dict[str, str | None] = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def data_fingerprints(data_dir: Path) -> dict[str, Any]:
    """Return size + mtime for every organizer TSV under ``data_dir/{train,test}`` (no content hashing)."""
    return {
        str(p.relative_to(data_dir)): input_fingerprint(p) for p in sorted(Path(data_dir).glob("*/*.tsv"))
    }


def write_manifest(run_dir: Path, cfg: Config | None = None, **fields: Any) -> Path:
    """Write ``run_dir/manifest.json``: environment, git, config (+hash), data fingerprints and ``fields``.

    Returns the manifest path. Non-JSON values (e.g. Paths) are stringified.
    """
    manifest: dict[str, Any] = {
        "written_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
        "packages": package_versions(),
        "git": git_info(cfg.root if cfg else None),
    }
    if cfg is not None:
        manifest["config"] = cfg.raw
        manifest["config_hash"] = config_hash(cfg)
        manifest["root"] = str(cfg.root)
        manifest["data"] = data_fingerprints(Path(cfg.paths.data_dir))
    manifest.update(fields)
    path = Path(run_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    return path
