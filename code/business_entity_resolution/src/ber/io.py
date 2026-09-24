"""Canonical TSV reading and the zstd Parquet cache every later stage reads from.

TSV semantics (verified on the organizer files): lines are split on TAB only (DuckDB ``quote=''``,
``escape=''``) so a stray quote can never merge or drop lines. Fields that contain a double quote were
written by the organizer with CSV QUOTE_MINIMAL style (3 quotes+abc+1 quote = quote+abc); after splitting,
any field that *starts* with ``"`` must be a well-formed quoted field and is unquoted (outer quotes
removed, ``""`` -> ``"``); a malformed one raises. Quotes elsewhere are kept literally (same as pandas).
Empty fields become ``''`` (never NULL). Row counts are checked against the file's newline count.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import duckdb

from ber.config import Config
from ber.obs import kv, stage

log = logging.getLogger("ber.io")

IO_FORMAT_VERSION = 1
SPLITS = ("train", "test")
SOURCES = (1, 2, 3)
SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GT_COLUMNS = ("source1_entity_id", "matched_entity_ids")
PARQUET_OPTS = "FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 122880"
_CHUNK = 8 * 1024 * 1024


def sql_str(value: str | Path) -> str:
    """Return ``value`` as a single-quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def count_data_rows(path: Path) -> int:
    """Count data lines (all lines minus the header) by streaming bytes; a missing final newline counts."""
    lines, last = 0, b"\n"
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        lines += 1
    return max(lines - 1, 0)


def read_header(path: Path) -> list[str]:
    """Return the header fields of a TSV (first line split on TAB, newline stripped)."""
    with open(path, encoding="utf-8") as f:
        return f.readline().rstrip("\n").rstrip("\r").split("\t")


def check_header(path: Path, expected: tuple[str, ...]) -> None:
    """Raise ValueError unless the TSV header equals ``expected`` exactly."""
    got = read_header(path)
    if tuple(got) != expected:
        raise ValueError(f"{path}: header {got} != expected {list(expected)}")


def decode_field(col: str) -> str:
    """SQL mapping a raw TAB-split field to its value (NULL->'', CSV-unquote, raise if malformed)."""
    x = f"coalesce({col}, '')"
    return (
        f"CASE WHEN starts_with({x}, '\"') THEN "
        f'CASE WHEN regexp_full_match({x}, \'"([^"]|"")*"\') '
        f"THEN replace(substr({x}, 2, length({x}) - 2), '\"\"', '\"') "
        f"ELSE error('malformed quoted field in column {col}: ' || {x}) END "
        f"ELSE {x} END"
    )


def read_tsv_sql(path: Path, columns: tuple[str, ...]) -> str:
    """SQL ``read_csv(...)`` call reading ``path`` as strict TAB-only TSV with VARCHAR ``columns``."""
    cols = ", ".join(f"'{c}': 'VARCHAR'" for c in columns)
    return (
        f"read_csv({sql_str(path)}, delim='\\t', header=true, quote='', escape='', new_line='\\n', "
        f"auto_detect=false, strict_mode=true, null_padding=false, ignore_errors=false, "
        f"columns={{{cols}}})"
    )


def source_select_sql(path: Path, source: int) -> str:
    """SELECT producing the canonical source table (4 decoded columns + ``source`` TINYINT) from a TSV."""
    fields = ", ".join(f"{decode_field(c)} AS {c}" for c in SOURCE_COLUMNS)
    return (
        f"SELECT {fields}, CAST({int(source)} AS TINYINT) AS source FROM {read_tsv_sql(path, SOURCE_COLUMNS)}"
    )


def id_list_select_sql(path: Path) -> str:
    """SELECT (s1_id, ids) from any 2-column id-list TSV (ground truth, matching or candidate file)."""
    return (
        f"SELECT {decode_field('a')} AS s1_id, {decode_field('b')} AS ids "
        f"FROM {read_tsv_sql(path, ('a', 'b'))}"
    )


def explode_ids_sql(inner: str) -> str:
    """SELECT (s1_id, match_id) exploding the comma list ``ids`` of ``inner``; empty lists yield no rows."""
    return f"SELECT s1_id, unnest(string_split(ids, ',')) AS match_id FROM ({inner}) WHERE ids <> ''"


def input_fingerprint(path: Path) -> dict[str, Any]:
    """Return a cheap fingerprint (path, size, mtime_ns) of an input file — never hashes contents."""
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _sidecar(out: Path) -> Path:
    """Path of the JSON sidecar describing how ``out`` was produced."""
    return out.with_name(out.name + ".meta.json")


def is_fresh(out: Path, src: Path) -> bool:
    """True when ``out`` exists and its sidecar matches the current input fingerprint and format version."""
    side = _sidecar(out)
    if not (out.exists() and side.exists()):
        return False
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    fp = input_fingerprint(src)
    return (
        meta.get("format_version") == IO_FORMAT_VERSION
        and meta.get("input", {}).get("size") == fp["size"]
        and meta.get("input", {}).get("mtime_ns") == fp["mtime_ns"]
    )


def _write_sidecar(out: Path, src: Path, rows: int, extra: dict[str, Any] | None = None) -> None:
    """Write the sidecar JSON for ``out`` recording input fingerprint, row count and extras."""
    meta = {
        "format_version": IO_FORMAT_VERSION,
        "input": input_fingerprint(src),
        "rows": rows,
        **(extra or {}),
    }
    _sidecar(out).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _copy_parquet(con: duckdb.DuckDBPyConnection, select_sql: str, out: Path) -> None:
    """COPY ``select_sql`` to ``out`` as zstd Parquet atomically (write to .tmp then rename)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con.execute(f"COPY ({select_sql}) TO {sql_str(tmp)} ({PARQUET_OPTS})")
    os.replace(tmp, out)


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> Any:
    """Run ``sql`` and return the first column of the first row."""
    row = con.execute(sql).fetchone()
    return None if row is None else row[0]


def _require(ok: bool, message: str) -> None:
    """Raise ValueError(message) unless ``ok``."""
    if not ok:
        raise ValueError(message)


def convert_source(con: duckdb.DuckDBPyConnection, src: Path, out: Path, source: int) -> dict[str, Any]:
    """Convert one source TSV to canonical Parquet and verify rows, ids and prefixes; return stats.

    Raises ValueError on header mismatch, row-count mismatch, empty ids or wrong ``S{source}-`` prefix.
    """
    check_header(src, SOURCE_COLUMNS)
    expected = count_data_rows(src)
    _copy_parquet(con, source_select_sql(src, source), out)
    pq = f"read_parquet({sql_str(out)})"
    rows, bad_prefix, n_distinct, empty_addr, empty_name = con.execute(
        f"SELECT count(*), count_if(NOT starts_with(entity_id, 'S{source}-')), count(DISTINCT entity_id), "
        f"count_if(business_address = ''), count_if(business_name = '') FROM {pq}"
    ).fetchone()
    try:
        _require(rows == expected, f"{src}: parquet has {rows} rows but file has {expected} data lines")
        _require(bad_prefix == 0, f"{src}: {bad_prefix} entity_ids without prefix S{source}-")
    except ValueError:
        out.unlink(missing_ok=True)
        raise
    stats = {
        "rows": rows,
        "expected_rows": expected,
        "duplicate_ids": rows - n_distinct,
        "empty_address": empty_addr,
        "empty_name": empty_name,
    }
    if rows != n_distinct:
        log.warning(kv(event="duplicate_entity_ids", file=src.name, n=rows - n_distinct))
    return stats


def convert_ground_truth(
    con: duckdb.DuckDBPyConnection, src: Path, out_pairs: Path, out_s1: Path
) -> dict[str, Any]:
    """Convert the GT TSV to (s1_id, match_id) pairs and (s1_id, n_matches) Parquet; verify and return stats.

    Raises ValueError on header/row-count mismatch, non-S1 keys, empty list tokens or non-S2/S3 matches.
    """
    check_header(src, GT_COLUMNS)
    expected = count_data_rows(src)
    base = id_list_select_sql(src)
    _copy_parquet(
        con,
        f"SELECT s1_id, CAST(CASE WHEN ids = '' THEN 0 ELSE len(string_split(ids, ',')) END AS INTEGER) "
        f"AS n_matches FROM ({base})",
        out_s1,
    )
    _copy_parquet(con, explode_ids_sql(base), out_pairs)
    s1 = f"read_parquet({sql_str(out_s1)})"
    pairs = f"read_parquet({sql_str(out_pairs)})"
    rows, bad_s1, n_distinct_s1, n_singletons, n_listed = con.execute(
        f"SELECT count(*), count_if(NOT starts_with(s1_id, 'S1-')), count(DISTINCT s1_id), "
        f"count_if(n_matches = 0), sum(n_matches) FROM {s1}"
    ).fetchone()
    n_pairs, bad_match, n_distinct_pairs, n_distinct_match = con.execute(
        f"SELECT count(*), count_if(NOT regexp_full_match(match_id, 'S[23]-[^,\\s]+')), "
        f"count(DISTINCT (s1_id, match_id)), count(DISTINCT match_id) FROM {pairs}"
    ).fetchone()
    try:
        _require(rows == expected, f"{src}: parquet has {rows} rows but file has {expected} data lines")
        _require(bad_s1 == 0, f"{src}: {bad_s1} source1_entity_id values without S1- prefix")
        _require(n_distinct_s1 == rows, f"{src}: {rows - n_distinct_s1} duplicate source1_entity_id rows")
        _require(n_pairs == (n_listed or 0), f"{src}: exploded {n_pairs} pairs but lists hold {n_listed}")
        _require(bad_match == 0, f"{src}: {bad_match} matched ids that are empty or not S2-/S3- ids")
    except ValueError:
        out_s1.unlink(missing_ok=True)
        out_pairs.unlink(missing_ok=True)
        raise
    return {
        "rows": rows,
        "expected_rows": expected,
        "singletons": n_singletons,
        "pairs": n_pairs,
        "duplicate_pairs": n_pairs - n_distinct_pairs,
        "distinct_match_ids": n_distinct_match,
    }


def cache_dir(cfg: Config) -> Path:
    """Return the Parquet cache directory from config."""
    return Path(cfg.paths.cache_dir)


def raw_tsv_path(cfg: Config, split: str, source: int) -> Path:
    """Return the organizer TSV path for ``split`` in {train,test} and ``source`` in {1,2,3}."""
    return Path(cfg.paths.data_dir) / split / f"{split}_source{source}.tsv"


def gt_tsv_path(cfg: Config) -> Path:
    """Return the organizer training ground-truth TSV path."""
    return Path(cfg.paths.data_dir) / "train" / "train_ground_truth.tsv"


def raw_parquet_path(cfg: Config, split: str, source: int) -> Path:
    """Return the cached Parquet path ``raw_{split}_s{source}.parquet``."""
    return cache_dir(cfg) / f"raw_{split}_s{source}.parquet"


def gt_pairs_path(cfg: Config) -> Path:
    """Return the cached GT pairs Parquet path (columns s1_id, match_id)."""
    return cache_dir(cfg) / "train_gt_pairs.parquet"


def gt_s1_path(cfg: Config) -> Path:
    """Return the cached GT per-S1 Parquet path (columns s1_id, n_matches)."""
    return cache_dir(cfg) / "train_gt_s1.parquet"


def prepare(cfg: Config, force: bool = False, con: duckdb.DuckDBPyConnection | None = None) -> dict[str, Any]:
    """Convert all 7 organizer TSVs into the Parquet cache (idempotent; ``force`` rebuilds everything).

    Returns ``{output_name: stats}`` where stats include rows, seconds and ``skipped``.
    """
    from ber.duck import connect

    own = con is None
    con = con or connect(cfg)
    results: dict[str, Any] = {}
    try:
        for split in SPLITS:
            for source in SOURCES:
                src, out = raw_tsv_path(cfg, split, source), raw_parquet_path(cfg, split, source)
                results[out.name] = _run_step(
                    out, src, force, lambda s=src, o=out, n=source: convert_source(con, s, o, n)
                )
        src = gt_tsv_path(cfg)
        pairs_out, s1_out = gt_pairs_path(cfg), gt_s1_path(cfg)
        if not force and is_fresh(pairs_out, src) and is_fresh(s1_out, src):
            results[pairs_out.name] = {"skipped": True}
            log.info(kv(event="prepare_skip", output=pairs_out.name, reason="fresh"))
        else:
            with stage("prepare.file", output=pairs_out.name) as st:
                t0 = time.perf_counter()
                stats = convert_ground_truth(con, src, pairs_out, s1_out)
                stats["seconds"] = round(time.perf_counter() - t0, 2)
                st.update(stats)
            _write_sidecar(s1_out, src, stats["rows"], {"stats": stats})
            _write_sidecar(pairs_out, src, stats["pairs"], {"stats": stats})
            results[pairs_out.name] = {**stats, "skipped": False}
    finally:
        if own:
            con.close()
    return results


def _run_step(out: Path, src: Path, force: bool, fn: Any) -> dict[str, Any]:
    """Run one conversion ``fn`` unless ``out`` is fresh; log via ``stage`` and write the sidecar."""
    if not force and is_fresh(out, src):
        log.info(kv(event="prepare_skip", output=out.name, reason="fresh"))
        return {"skipped": True}
    with stage("prepare.file", output=out.name) as st:
        t0 = time.perf_counter()
        stats = fn()
        stats["seconds"] = round(time.perf_counter() - t0, 2)
        st.update(stats)
    _write_sidecar(out, src, stats["rows"], {"stats": stats})
    return {**stats, "skipped": False}


def open_parquet(con: duckdb.DuckDBPyConnection, path: Path) -> duckdb.DuckDBPyRelation:
    """Open a Parquet file as a DuckDB relation; FileNotFoundError with a hint if missing."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `ber prepare` first")
    return con.read_parquet(str(path))


def open_source(
    con: duckdb.DuckDBPyConnection, cfg: Config, split: str, source: int
) -> duckdb.DuckDBPyRelation:
    """Relation over ``raw_{split}_s{source}.parquet``.

    Columns: entity_id, business_name, business_address, country, source.
    """
    return open_parquet(con, raw_parquet_path(cfg, split, source))


def open_right(con: duckdb.DuckDBPyConnection, cfg: Config, split: str) -> duckdb.DuckDBPyRelation:
    """Relation over the union of Source 2 and Source 3 for ``split`` (same columns as ``open_source``)."""
    paths = [str(raw_parquet_path(cfg, split, s)) for s in (2, 3)]
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} missing — run `ber prepare` first")
    return con.read_parquet(paths)


def open_gt_pairs(con: duckdb.DuckDBPyConnection, cfg: Config) -> duckdb.DuckDBPyRelation:
    """Relation over train GT pairs (s1_id, match_id)."""
    return open_parquet(con, gt_pairs_path(cfg))


def open_gt_s1(con: duckdb.DuckDBPyConnection, cfg: Config) -> duckdb.DuckDBPyRelation:
    """Relation over train GT per-S1 rows (s1_id, n_matches); n_matches = 0 marks a singleton."""
    return open_parquet(con, gt_s1_path(cfg))


def read_id_list_file(
    con: duckdb.DuckDBPyConnection, path: Path
) -> tuple[duckdb.DuckDBPyRelation, duckdb.DuckDBPyRelation]:
    """Read a 2-column id-list TSV (GT/matching/candidate) or GT-pairs Parquet.

    Returns ``(pairs, s1_ids)`` relations: pairs has (s1_id, match_id), s1_ids has (s1_id). For Parquet
    input (already exploded pairs) s1_ids are the distinct s1_id values present.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        pairs = con.read_parquet(str(path)).select("s1_id, match_id")
        return pairs, pairs.select("s1_id").distinct()
    base = id_list_select_sql(path)
    return con.sql(explode_ids_sql(base)), con.sql(f"SELECT s1_id FROM ({base})")
