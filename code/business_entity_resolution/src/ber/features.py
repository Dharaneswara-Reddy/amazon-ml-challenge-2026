"""Pair features for candidate (left = Source 1, right = Source 2/3) pairs — feature set v2 (DESIGN_V2 §S3).

Scale design (decision D-019): nothing runs a global window over all candidates. Entity-side context (rank / gap /
count among an entity's candidates) is computed per hash bucket of the left id into a narrow table; the main pass
then runs per hash bucket of the right id, streaming that bucket's candidates from parquet, computing right-side
context inline, joining normalized strings, and sending Arrow batches to a spawn-based process pool for the
Python string features.

Feature families:
  * blocking evidence: which passes found the pair, blocking similarities, learned/legacy block score;
  * context: rank / gap / candidate count of the pair within its right record and within its entity;
  * v1 string similarities on name, core name, consonant skeleton, address (rapidfuzz);
  * v2 components: legal-form agreement, business-type-word overlap, street / city / admin agreement, unit presence;
  * v2 text differences (``ber.textdiff``): token-alignment classes (typo / abbreviation / transliteration /
    legal / unexplained), IDF mass and learned log-odds of unexplained tokens, house-number relation classes.
Output rows carry ``in_group`` (1 when the pair's entity belongs to the requested group, 0 for rival-context rows
that are kept only so stage 2 and the one-to-one step see every rival; never train on in_group = 0).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections import deque
from collections.abc import Iterator, Mapping
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from ber import signature, textdiff
from ber.block3 import V3_CAND_COLS

log = logging.getLogger(__name__)

FEATURE_SET_VERSION = "v3"

PASS_FLAGS = [
    "in_tok",
    "in_q3",
    "in_sn_name",
    "in_sn_name_addr",
    "in_sn_addr_name",
    "in_sn_skel_nums",
    "in_sn_nums_addr",
    "in_sn_addr_skel",
    "in_nameonly",
]
BLOCK_COLS = ["tok_score", "q3_score", "jw_core", "jw_addr", "jacc_addr", "num_match", "block_score"]
CONTEXT = ["r_n_cand", "r_rank", "r_gap", "l_n_cand", "l_rank", "l_gap"]
SQL_OTHER = [
    "r_is_s3",
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
    "legal_eq",
    "legal_l_only",
    "legal_r_only",
    "type_jacc",
    "type_diff",
    "city_eq",
    "admin_eq",
    "admin_diff",
    "l_unit",
    "r_unit",
]
# v3 candidate evidence (compound-key passes, cut model) and crowded-name context (D-022, D-023)
V3_BLOCK = list(V3_CAND_COLS)
CROWD = ["r_core_l_freq", "r_same_name_cands"]
SQL_FEATURES = PASS_FLAGS + BLOCK_COLS + CONTEXT + SQL_OTHER + V3_BLOCK + CROWD
PY_STRING = [
    "name_ratio",
    "name_jw",
    "core_ratio",
    "core_token_set",
    "core_token_sort",
    "core_partial",
    "core_jw",
    "skel_ratio",
    "skel_jw",
    "addr_ratio",
    "addr_token_set",
    "addr_token_sort",
    "addr_partial",
    "addr_jw",
    "street_jw",
    "street_token_set",
    "postal_state",
    "nums_shared",
    "nums_conflict",
    "first_tok_eq",
]
TEXTDIFF_NAME = list(textdiff.name_diff_features([], [], None, None).keys())
TEXTDIFF_NUM = list(textdiff.number_relation_features([], []).keys())
PY_FEATURES = PY_STRING + TEXTDIFF_NAME + TEXTDIFF_NUM + signature.SIG_FEATURES
FEATURES = SQL_FEATURES + PY_FEATURES

STRING_COLS = (
    "name_norm",
    "name_core",
    "name_skel",
    "addr_norm",
    "postal",
    "addr_nums",
    "addr_street",
    "addr_house",
)


def _jacc(a: str, b: str) -> str:
    """Return SQL for token-set Jaccard between two space-separated string expressions."""
    sa, sb = f"list_distinct(string_split({a}, ' '))", f"list_distinct(string_split({b}, ' '))"
    return f"len(list_intersect({sa}, {sb}))::FLOAT / greatest(len(list_distinct(list_concat({sa}, {sb}))), 1)"


def _col_or(cols: set[str], name: str, default: str, cast: str = "") -> str:
    """SQL for candidate column ``name`` (optionally cast) or a default literal when the column is absent."""
    return f"c.{name}{cast} AS {name}" if name in cols else f"{default} AS {name}"


def _norm_col(norm_cols: set[str], side: str, name: str) -> str:
    """SQL for a normalized column of L or R, '' when an older normalized file lacks it."""
    return f"{side}.{name}" if name in norm_cols else "''"


def build_entity_context(con: duckdb.DuckDBPyConnection, buckets: int) -> None:
    """Create table lctx(lid, rid, l_n_cand, l_rank, l_gap), computed per hash bucket of the entity id."""
    con.execute("DROP TABLE IF EXISTS lctx")
    for b in range(buckets):
        body = f"""SELECT lid, rid,
                      count(*) OVER (PARTITION BY lid) AS l_n_cand,
                      rank() OVER (PARTITION BY lid ORDER BY block_score DESC) AS l_rank,
                      max(block_score) OVER (PARTITION BY lid) - block_score AS l_gap
                   FROM candidates WHERE hash(lid) % {buckets} = {b}"""
        con.execute(f"CREATE TABLE lctx AS {body}" if b == 0 else f"INSERT INTO lctx {body}")


def bucket_frame_sql(
    cand_cols: set[str],
    norm_cols: set[str],
    b: int,
    buckets: int,
    label_sql: str | None,
    group_sql: str | None,
    context_rids: bool,
) -> str:
    """SQL for one right-id bucket: candidates of that bucket with strings, SQL features, in_group, optional label.

    group_sql is a predicate on ``c.lid`` selecting the requested entities; with context_rids, all candidates of any
    right record that has a candidate in the group are kept too (in_group = 0) so rivals are complete.
    """
    rb = f"hash(rid) % {buckets} = {b}"
    if group_sql and context_rids:
        keep = f"(c0.rid IN (SELECT rid FROM ctx_rids WHERE {rb}))"
    elif group_sql:
        keep = group_sql.replace("c.lid", "c0.lid")
    else:
        keep = "TRUE"
    in_group = f"({group_sql})::TINYINT" if group_sql else "1::TINYINT"
    label_join = f"LEFT JOIN ({label_sql}) g ON g.s1_id = c.lid AND g.match_id = c.rid" if label_sql else ""
    label_col = ", (g.s1_id IS NOT NULL)::TINYINT AS label" if label_sql else ""
    flags = ",\n      ".join(_col_or(cand_cols, f, "0::TINYINT", "::TINYINT") for f in PASS_FLAGS)
    blocks = ",\n      ".join(
        f"coalesce(c.{f}, 0)::FLOAT AS {f}" if f in cand_cols else f"0::FLOAT AS {f}" for f in BLOCK_COLS + V3_BLOCK
    )
    strings = ",\n      ".join(
        f"{_norm_col(norm_cols, 'l', s)} AS l_{s}, {_norm_col(norm_cols, 'r', s)} AS r_{s}" for s in STRING_COLS
    )
    lf, rf = _norm_col(norm_cols, "l", "legal_form"), _norm_col(norm_cols, "r", "legal_form")
    lt, rt = _norm_col(norm_cols, "l", "type_words"), _norm_col(norm_cols, "r", "type_words")
    lcity, rcity = _norm_col(norm_cols, "l", "addr_city"), _norm_col(norm_cols, "r", "addr_city")
    ladm, radm = _norm_col(norm_cols, "l", "addr_admin"), _norm_col(norm_cols, "r", "addr_admin")
    lunit, runit = _norm_col(norm_cols, "l", "addr_unit"), _norm_col(norm_cols, "r", "addr_unit")
    return f"""
    WITH c AS (
      SELECT c0.*,
        count(*) OVER (PARTITION BY c0.rid) AS r_n_cand,
        rank() OVER (PARTITION BY c0.rid ORDER BY c0.block_score DESC) AS r_rank,
        max(c0.block_score) OVER (PARTITION BY c0.rid) - c0.block_score AS r_gap
      FROM (SELECT * FROM candidates WHERE {rb}) c0
      WHERE {keep})
    SELECT c.lid, c.rid, {in_group} AS in_group,
      {strings},
      {flags},
      {blocks},
      c.r_n_cand, c.r_rank, c.r_gap, x.l_n_cand, x.l_rank, x.l_gap,
      (c.rid LIKE 'S3-%')::TINYINT AS r_is_s3,
      lfq.f AS l_name_freq, coalesce(rfq.f, 0) AS r_name_freq,
      {_jacc("l.name_core", "r.name_core")} AS tok_jacc_name,
      {_jacc("l.addr_norm", "r.addr_norm")} AS tok_jacc_addr,
      least(length(l.name_norm), length(r.name_norm))::FLOAT
        / greatest(length(l.name_norm), length(r.name_norm), 1) AS len_ratio_name,
      least(length(l.addr_norm), length(r.addr_norm))::FLOAT
        / greatest(length(l.addr_norm), length(r.addr_norm), 1) AS len_ratio_addr,
      (l.addr_norm = '')::TINYINT AS l_addr_empty, (r.addr_norm = '')::TINYINT AS r_addr_empty,
      (l.script = 'latin')::TINYINT AS l_script_latin, (r.script = 'latin')::TINYINT AS r_script_latin,
      ({lf} <> '' AND {lf} = {rf})::TINYINT AS legal_eq,
      ({lf} <> '' AND {rf} = '')::TINYINT AS legal_l_only,
      ({lf} = '' AND {rf} <> '')::TINYINT AS legal_r_only,
      {_jacc(lt, rt)} AS type_jacc,
      len(list_filter(list_concat(list_distinct(string_split({lt}, ' ')), list_distinct(string_split({rt}, ' '))),
          x -> x <> '')) - 2 * len(list_filter(list_intersect(string_split({lt}, ' '), string_split({rt}, ' ')),
          x -> x <> '')) AS type_diff,
      ({lcity} <> '' AND {lcity} = {rcity})::TINYINT AS city_eq,
      ({ladm} <> '' AND {ladm} = {radm})::TINYINT AS admin_eq,
      ({ladm} <> '' AND {radm} <> '' AND {ladm} <> {radm})::TINYINT AS admin_diff,
      ({lunit} <> '')::TINYINT AS l_unit, ({runit} <> '')::TINYINT AS r_unit,
      coalesce(rlf.f, 0) AS r_core_l_freq,
      sum((l.name_core = r.name_core)::INT) OVER (PARTITION BY c.rid) AS r_same_name_cands
      {label_col}
    FROM c
    JOIN (SELECT * FROM lctx WHERE {rb}) x ON x.lid = c.lid AND x.rid = c.rid
    JOIN L l ON l.entity_id = c.lid
    JOIN (SELECT * FROM R WHERE hash(entity_id) % {buckets} = {b}) r ON r.entity_id = c.rid
    LEFT JOIN lname_freq lfq ON lfq.name_core = l.name_core
    LEFT JOIN rname_freq rfq ON rfq.name_core = r.name_core
    LEFT JOIN lname_freq rlf ON rlf.name_core = r.name_core
    {label_join}
    """


def _cp(a: list[str], b: list[str], scorer) -> np.ndarray:
    """Element-wise rapidfuzz scores for two equal-length string lists as float32 in [0, 1] (single thread)."""
    out = process.cpdist(a, b, scorer=scorer, workers=1, dtype=np.float32)
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


_IDF: Mapping[str, float] | None = None
_LOGODDS: Mapping[str, float] | None = None
_SIGVOCAB: Mapping[str, Mapping[str, float]] | None = None


def _init_worker(
    idf: Mapping[str, float] | None,
    logodds: Mapping[str, float] | None,
    sigvocab: Mapping[str, Mapping[str, float]] | None = None,
) -> None:
    """Process-pool initializer: install the IDF, token log-odds and signature vocabularies used by the features."""
    global _IDF, _LOGODDS, _SIGVOCAB
    _IDF, _LOGODDS, _SIGVOCAB = idf, logodds, sigvocab


def python_features(batch: pa.RecordBatch) -> dict[str, np.ndarray]:
    """Compute rapidfuzz string features and text-difference features for one batch of joined pairs."""
    col = {f"{s}_{c}": batch.column(f"{s}_{c}").to_pylist() for s in ("l", "r") for c in STRING_COLS}
    jw = JaroWinkler.normalized_similarity
    pairs = {
        "name_ratio": ("name_norm", fuzz.ratio),
        "name_jw": ("name_norm", jw),
        "core_ratio": ("name_core", fuzz.ratio),
        "core_token_set": ("name_core", fuzz.token_set_ratio),
        "core_token_sort": ("name_core", fuzz.token_sort_ratio),
        "core_partial": ("name_core", fuzz.partial_ratio),
        "core_jw": ("name_core", jw),
        "skel_ratio": ("name_skel", fuzz.ratio),
        "skel_jw": ("name_skel", jw),
        "addr_ratio": ("addr_norm", fuzz.ratio),
        "addr_token_set": ("addr_norm", fuzz.token_set_ratio),
        "addr_token_sort": ("addr_norm", fuzz.token_sort_ratio),
        "addr_partial": ("addr_norm", fuzz.partial_ratio),
        "addr_jw": ("addr_norm", jw),
        "street_jw": ("addr_street", jw),
        "street_token_set": ("addr_street", fuzz.token_set_ratio),
    }
    f = {name: _cp(col[f"l_{c}"], col[f"r_{c}"], scorer) for name, (c, scorer) in pairs.items()}
    f["postal_state"] = _postal_state(col["l_postal"], col["r_postal"])
    f["nums_shared"], f["nums_conflict"] = _nums(col["l_addr_nums"], col["r_addr_nums"])
    f["first_tok_eq"] = np.fromiter(
        (
            (x.split(" ", 1)[0] == y.split(" ", 1)[0]) if x and y else False
            for x, y in zip(col["l_name_core"], col["r_name_core"], strict=True)
        ),
        dtype=np.int8,
        count=batch.num_rows,
    )
    f.update(textdiff.name_diff_features(col["l_name_core"], col["r_name_core"], _IDF, _LOGODDS))
    houses_l = [h or n for h, n in zip(col["l_addr_house"], col["l_addr_nums"], strict=True)]
    houses_r = [h or n for h, n in zip(col["r_addr_house"], col["r_addr_nums"], strict=True)]
    f.update(textdiff.number_relation_features(houses_l, houses_r))
    f.update(
        signature.signature_features(
            col["l_name_norm"], col["l_addr_norm"], col["r_name_norm"], col["r_addr_norm"], _SIGVOCAB
        )
    )
    return f


def _assemble(batch: pa.RecordBatch, py: dict[str, np.ndarray], with_label: bool) -> pa.RecordBatch:
    """Combine SQL columns and Python features into one output batch with a fixed column order."""
    keep = ["lid", "rid", "in_group", *SQL_FEATURES] + (["label"] if with_label else [])
    arrays = [batch.column(n) for n in keep] + [pa.array(np.asarray(py[n], dtype=np.float32)) for n in PY_FEATURES]
    return pa.RecordBatch.from_arrays(arrays, names=keep + PY_FEATURES)


def _prepare_tables(con: duckdb.DuckDBPyConnection, buckets: int, group_sql: str | None, context_rids: bool) -> None:
    """Name-frequency tables, entity context (lctx), and the rival-context right ids for a group."""
    con.execute("CREATE OR REPLACE TABLE lname_freq AS SELECT name_core, count(*) AS f FROM L GROUP BY 1")
    con.execute("CREATE OR REPLACE TABLE rname_freq AS SELECT name_core, count(*) AS f FROM R GROUP BY 1")
    build_entity_context(con, buckets)
    if group_sql and context_rids:
        con.execute(f"CREATE OR REPLACE TABLE ctx_rids AS SELECT DISTINCT rid FROM candidates c WHERE {group_sql}")


def iter_feature_batches(
    con: duckdb.DuckDBPyConnection,
    label_sql: str | None,
    batch_rows: int,
    workers: int,
    group_sql: str | None = None,
    context_rids: bool = False,
    buckets: int = 16,
    idf: Mapping[str, float] | None = None,
    logodds: Mapping[str, float] | None = None,
    sigvocab: Mapping[str, Mapping[str, float]] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream candidate pairs bucket by bucket through both feature layers; yield assembled feature batches.

    Expects view ``candidates`` (blocking output) and views L, R (normalized). group_sql is a predicate on c.lid.
    """
    cand_cols = {r[0] for r in con.execute("DESCRIBE candidates").fetchall()}
    norm_cols = {r[0] for r in con.execute("DESCRIBE L").fetchall()}
    _prepare_tables(con, buckets, group_sql, context_rids)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(max(1, workers), initializer=_init_worker, initargs=(idf, logodds, sigvocab))
    pending: deque = deque()
    try:
        for b in range(buckets):
            sql = bucket_frame_sql(cand_cols, norm_cols, b, buckets, label_sql, group_sql, context_rids)
            for batch in con.execute(sql).to_arrow_reader(batch_rows):
                pending.append((batch, pool.apply_async(python_features, (batch,))))
                while len(pending) > 2 * workers:
                    done_batch, res = pending.popleft()
                    yield _assemble(done_batch, res.get(), label_sql is not None)
            log.info("features bucket=%d/%d", b + 1, buckets)
        while pending:
            done_batch, res = pending.popleft()
            yield _assemble(done_batch, res.get(), label_sql is not None)
    finally:
        pool.terminate()
        pool.join()


def write_features(
    con: duckdb.DuckDBPyConnection,
    out_path: Path,
    label_sql: str | None,
    batch_rows: int,
    workers: int,
    group_sql: str | None = None,
    context_rids: bool = False,
    buckets: int = 16,
    idf: Mapping[str, float] | None = None,
    logodds: Mapping[str, float] | None = None,
    sigvocab: Mapping[str, Mapping[str, float]] | None = None,
) -> int:
    """Write the feature matrix for view `candidates` (optionally one group + rivals) to zstd parquet; return rows."""
    writer = None
    n = 0
    try:
        for batch in iter_feature_batches(
            con, label_sql, batch_rows, workers, group_sql, context_rids, buckets, idf, logodds, sigvocab
        ):
            if writer is None:
                writer = pq.ParquetWriter(out_path, batch.schema, compression="zstd")
            writer.write_batch(batch)
            n += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    log.info("features written path=%s rows=%d version=%s", out_path, n, FEATURE_SET_VERSION)
    return n


def name_idf(con: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """IDF of name_core tokens over L ∪ R of the current split (unsupervised; no labels)."""
    rows = con.execute(
        """WITH t AS (SELECT DISTINCT entity_id, tok FROM (
                 SELECT entity_id, unnest(string_split(name_core, ' ')) AS tok FROM L
                 UNION ALL SELECT entity_id, unnest(string_split(name_core, ' ')) AS tok FROM R) WHERE tok <> '')
           SELECT tok, ln((SELECT count(*) FROM L) + (SELECT count(*) FROM R)) - ln(count(*)) FROM t GROUP BY tok"""
    ).fetchall()
    return {t: float(v) for t, v in rows}
