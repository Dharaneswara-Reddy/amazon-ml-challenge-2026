"""Pair features for candidate (left, right) pairs.

Two layers:
  * SQL layer (DuckDB): joins normalized fields onto the `candidates` table and adds group-context features
    (rank of the pair among a right record's candidates and among a left record's candidates, gap to the best,
    candidate counts, name frequency as a chain-store signal, token Jaccard).
  * Python layer (rapidfuzz, streamed Arrow batches): string similarities on names and addresses, postal and
    house-number agreement.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

log = logging.getLogger(__name__)

FEATURE_SET_VERSION = "v1b"

SQL_FEATURES = [
    "in_tok",
    "in_sn_name",
    "in_sn_name_addr",
    "in_sn_addr_name",
    "in_sn_skel_nums",
    "in_sn_nums_addr",
    "in_sn_addr_skel",
    "tok_score",
    "jw_core",
    "jw_addr",
    "jacc_addr",
    "num_match",
    "block_score",
    "r_is_s3",
    "r_n_cand",
    "r_rank",
    "r_gap",
    "l_n_cand",
    "l_rank",
    "l_gap",
    "l_name_freq",
    "r_name_freq",
    "tok_jacc_name",
    "tok_jacc_addr",
    "len_ratio_name",
    "len_ratio_addr",
    "l_addr_empty",
    "r_addr_empty",
    "l_script_latin",
    "r_script_latin",
]
PY_FEATURES = [
    "name_ratio",
    "name_jw",
    "core_ratio",
    "core_token_set",
    "core_token_sort",
    "core_partial",
    "core_jw",
    "addr_ratio",
    "addr_token_set",
    "addr_token_sort",
    "addr_partial",
    "addr_jw",
    "postal_state",
    "nums_shared",
    "nums_conflict",
    "first_tok_eq",
]
FEATURES = SQL_FEATURES + PY_FEATURES


def _jacc(a: str, b: str) -> str:
    """Return SQL for token-set Jaccard between two space-separated string columns."""
    sa, sb = f"list_distinct(string_split({a}, ' '))", f"list_distinct(string_split({b}, ' '))"
    return f"len(list_intersect({sa}, {sb}))::FLOAT / greatest(len(list_distinct(list_concat({sa}, {sb}))), 1)"


def pair_frame_sql(label_sql: str | None = None, where_sql: str | None = None) -> str:
    """Return SQL producing one row per candidate with ids, raw normalized fields, SQL features, and optional label.

    Expects table `candidates` (from ber.blocking) and views L, R with normalized columns. If label_sql is given it
    must yield (s1_id, match_id) and adds column `label` (1 if the pair is a true match). where_sql filters rows
    AFTER the group-context window features are computed over all candidates (e.g. to keep a sample of S1 ids).
    """
    label_join = f"LEFT JOIN ({label_sql}) g ON g.s1_id = c.lid AND g.match_id = c.rid" if label_sql else ""
    label_col = ", (g.s1_id IS NOT NULL)::TINYINT AS label" if label_sql else ""
    return f"""
    WITH lf AS (SELECT name_core, count(*) AS f FROM L GROUP BY 1),
         rf AS (SELECT name_core, count(*) AS f FROM R GROUP BY 1),
         c AS (
           SELECT *,
             count(*) OVER (PARTITION BY rid) AS r_n_cand,
             rank() OVER (PARTITION BY rid ORDER BY block_score DESC) AS r_rank,
             max(block_score) OVER (PARTITION BY rid) - block_score AS r_gap,
             count(*) OVER (PARTITION BY lid) AS l_n_cand,
             rank() OVER (PARTITION BY lid ORDER BY block_score DESC) AS l_rank,
             max(block_score) OVER (PARTITION BY lid) - block_score AS l_gap
           FROM candidates)
    SELECT c.lid, c.rid,
      l.name_norm AS l_name_norm, r.name_norm AS r_name_norm,
      l.name_core AS l_name_core, r.name_core AS r_name_core,
      l.name_skel AS l_name_skel, r.name_skel AS r_name_skel,
      l.addr_norm AS l_addr_norm, r.addr_norm AS r_addr_norm,
      l.postal AS l_postal, r.postal AS r_postal,
      l.addr_nums AS l_addr_nums, r.addr_nums AS r_addr_nums,
      c.in_tok::TINYINT AS in_tok, c.in_sn_name::TINYINT AS in_sn_name,
      c.in_sn_name_addr::TINYINT AS in_sn_name_addr, c.in_sn_addr_name::TINYINT AS in_sn_addr_name,
      c.in_sn_skel_nums::TINYINT AS in_sn_skel_nums, c.in_sn_nums_addr::TINYINT AS in_sn_nums_addr,
      c.in_sn_addr_skel::TINYINT AS in_sn_addr_skel,
      coalesce(c.tok_score, 0) AS tok_score, c.jw_core, c.jw_addr, c.jacc_addr, c.num_match, c.block_score,
      (c.rid LIKE 'S3-%')::TINYINT AS r_is_s3,
      c.r_n_cand, c.r_rank, c.r_gap, c.l_n_cand, c.l_rank, c.l_gap,
      lf.f AS l_name_freq, coalesce(rf.f, 0) AS r_name_freq,
      {_jacc("l.name_core", "r.name_core")} AS tok_jacc_name,
      {_jacc("l.addr_norm", "r.addr_norm")} AS tok_jacc_addr,
      least(length(l.name_norm), length(r.name_norm))::FLOAT
        / greatest(length(l.name_norm), length(r.name_norm), 1) AS len_ratio_name,
      least(length(l.addr_norm), length(r.addr_norm))::FLOAT
        / greatest(length(l.addr_norm), length(r.addr_norm), 1) AS len_ratio_addr,
      (l.addr_norm = '')::TINYINT AS l_addr_empty, (r.addr_norm = '')::TINYINT AS r_addr_empty,
      (l.script = 'latin')::TINYINT AS l_script_latin, (r.script = 'latin')::TINYINT AS r_script_latin
      {label_col}
    FROM c
    JOIN L l ON l.entity_id = c.lid
    JOIN R r ON r.entity_id = c.rid
    LEFT JOIN lf ON lf.name_core = l.name_core
    LEFT JOIN rf ON rf.name_core = r.name_core
    {label_join}
    {f"WHERE {where_sql}" if where_sql else ""}
    """


def _cp(a: list[str], b: list[str], scorer, workers: int) -> np.ndarray:
    """Element-wise rapidfuzz scores for two equal-length string lists as float32 in [0, 1]."""
    out = process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)
    return out / 100.0 if scorer is not JaroWinkler.normalized_similarity else out


def _postal_state(a: list[str], b: list[str]) -> np.ndarray:
    """1 if both postal codes present and equal, -1 if both present and different, 0 if either missing."""
    return np.fromiter(
        ((0 if not x or not y else (1 if x == y else -1)) for x, y in zip(a, b, strict=True)),
        dtype=np.int8,
        count=len(a),
    )


def _nums(a: list[str], b: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Count shared house/unit numbers and flag pairs where both sides have numbers but share none."""
    shared = np.zeros(len(a), dtype=np.int8)
    conflict = np.zeros(len(a), dtype=np.int8)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x and y:
            n = len(set(x.split()) & set(y.split()))
            shared[i] = min(n, 127)
            conflict[i] = n == 0
    return shared, conflict


def python_features(batch: pa.RecordBatch, workers: int) -> dict[str, np.ndarray]:
    """Compute rapidfuzz/string features for one batch of joined pairs."""
    col = {
        n: batch.column(n).to_pylist()
        for n in (
            "l_name_norm",
            "r_name_norm",
            "l_name_core",
            "r_name_core",
            "l_name_skel",
            "r_name_skel",
            "l_addr_norm",
            "r_addr_norm",
            "l_postal",
            "r_postal",
            "l_addr_nums",
            "r_addr_nums",
        )
    }
    jw = JaroWinkler.normalized_similarity
    f = {
        "name_ratio": _cp(col["l_name_norm"], col["r_name_norm"], fuzz.ratio, workers),
        "name_jw": _cp(col["l_name_norm"], col["r_name_norm"], jw, workers),
        "core_ratio": _cp(col["l_name_core"], col["r_name_core"], fuzz.ratio, workers),
        "core_token_set": _cp(col["l_name_core"], col["r_name_core"], fuzz.token_set_ratio, workers),
        "core_token_sort": _cp(col["l_name_core"], col["r_name_core"], fuzz.token_sort_ratio, workers),
        "core_partial": _cp(col["l_name_core"], col["r_name_core"], fuzz.partial_ratio, workers),
        "core_jw": _cp(col["l_name_core"], col["r_name_core"], jw, workers),
        "addr_ratio": _cp(col["l_addr_norm"], col["r_addr_norm"], fuzz.ratio, workers),
        "addr_token_set": _cp(col["l_addr_norm"], col["r_addr_norm"], fuzz.token_set_ratio, workers),
        "addr_token_sort": _cp(col["l_addr_norm"], col["r_addr_norm"], fuzz.token_sort_ratio, workers),
        "addr_partial": _cp(col["l_addr_norm"], col["r_addr_norm"], fuzz.partial_ratio, workers),
        "addr_jw": _cp(col["l_addr_norm"], col["r_addr_norm"], jw, workers),
        "postal_state": _postal_state(col["l_postal"], col["r_postal"]),
        "skel_ratio": _cp(col["l_name_skel"], col["r_name_skel"], fuzz.ratio, workers),
        "skel_jw": _cp(col["l_name_skel"], col["r_name_skel"], jw, workers),
    }
    f["nums_shared"], f["nums_conflict"] = _nums(col["l_addr_nums"], col["r_addr_nums"])
    f["first_tok_eq"] = np.fromiter(
        (
            (x.split(" ", 1)[0] == y.split(" ", 1)[0]) if x and y else False
            for x, y in zip(col["l_name_core"], col["r_name_core"], strict=True)
        ),
        dtype=np.int8,
        count=batch.num_rows,
    )
    return f


def iter_feature_batches(
    con: duckdb.DuckDBPyConnection,
    label_sql: str | None,
    batch_rows: int,
    workers: int,
    where_sql: str | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream candidate pairs through both feature layers, yielding batches with ids, FEATURES, and label if any."""
    reader = con.execute(pair_frame_sql(label_sql, where_sql)).to_arrow_reader(batch_rows)
    keep = ["lid", "rid", *SQL_FEATURES] + (["label"] if label_sql else [])
    for batch in reader:
        py = python_features(batch, workers)
        arrays = [batch.column(n) for n in keep] + [pa.array(py[n]) for n in PY_FEATURES]
        yield pa.RecordBatch.from_arrays(arrays, names=keep + PY_FEATURES)


def write_features(
    con: duckdb.DuckDBPyConnection,
    out_path: Path,
    label_sql: str | None,
    batch_rows: int,
    workers: int,
    where_sql: str | None = None,
) -> int:
    """Write the feature matrix for table `candidates` (optionally filtered) to zstd parquet; return the row count."""
    writer = None
    n = 0
    try:
        for batch in iter_feature_batches(con, label_sql, batch_rows, workers, where_sql):
            if writer is None:
                writer = pq.ParquetWriter(out_path, batch.schema, compression="zstd")
            writer.write_batch(batch)
            n += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    log.info("features written path=%s rows=%d version=%s", out_path, n, FEATURE_SET_VERSION)
    return n
