"""Blocking v3 (D-022): compound-key IDF passes on top of the v2 candidates, then a learned cut.

v2 candidates reach 94.5% recall because 74% of the misses share no single token rare enough for the token pass,
while *combinations* of common tokens are rare ("1940" + "fisher"). Five compound-key passes (name prefix x address
token, address number x address word, name-word pairs, name word x address word, whole phonetic name) lift the union
to 98.3%; a small GBDT on cheap pair features then keeps "linear top-1 per right OR p >= 1e-3", measured at 98.2%
recall with 7.6 candidates per S1 (v2: 14.1) on the full train split (research_notes/v3/E_proof_end_to_end.md).

Integer surrogate ids, hashed country-scoped keys and right-id buckets keep every step inside ~5 GB of DuckDB memory.
The cut model is trained on FIT-group rows only and reused unchanged for test.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber.blocking import addr_words_sql, load_ranker, ranker_sql

log = logging.getLogger(__name__)

PASSES_V3 = ["px", "hs", "n2", "na", "phx"]
V2_FLAGS = [
    "in_tok", "in_q3", "in_sn_name", "in_sn_name_addr", "in_sn_addr_name", "in_sn_skel_nums", "in_sn_nums_addr",
    "in_sn_addr_skel", "in_nameonly",
]
CUT_FEATURES = [
    "in_v2", "n_pass", "tok_score", "q3_score", *(f"s_{p}" for p in PASSES_V3), "n_pass_new", "jw_core", "jw_addr",
    "jacc_addr", "num_match", "jw_strip", "jw_phn", "jacc_name", "n_num", "r_addr_len", "r_nnum", "block_score",
]
# candidate columns added by v3 (consumed as features by ber.features)
V3_CAND_COLS = [
    *(c for p in PASSES_V3 for c in (f"in_{p}", f"s_{p}")), "n_pass_new", "in_v2", "jw_strip", "jw_phn", "cut_p",
    "lrank",
]

_NUMS = "list_slice(list_reverse(list_distinct(list_filter(string_split(addr_nums, ' '), x -> x <> ''))), 1, 5)"
_AW = ("list_distinct(list_filter(string_split(addr_norm, ' '), "
       "x -> length(x) >= 3 AND NOT regexp_matches(x, '^[0-9]+$')))")
_NW = "list_distinct(list_filter(string_split(name_core, ' '), x -> length(x) >= 2))"
_PFX = "substr(replace(name_core, ' ', ''), 1, 6)"
# pass -> (key SQL over table {T}(iid, c, name_core, addr_norm, addr_nums, phn), left df cap, top-k per right)
KEYS: dict[str, tuple[str, int, int]] = {
    "px": (f"SELECT iid, c, {_PFX} || '#' || x AS key FROM (SELECT iid, c, name_core, list_concat({_NUMS}, {_AW}) xs "
           "FROM {T} WHERE length(replace(name_core, ' ', '')) >= 6), unnest(xs) u(x)", 30, 5),
    "hs": (f"SELECT iid, c, n || '#' || w AS key FROM (SELECT iid, c, unnest({_NUMS}) n, {_AW} aw FROM {{T}}), "
           "unnest(aw) u(w)", 50, 5),
    "n2": (f"SELECT iid, c, least(a, b) || '#' || greatest(a, b) AS key FROM (SELECT iid, c, {_NW} nw FROM {{T}}), "
           "unnest(nw) u(a), unnest(nw) v(b) WHERE a < b", 30, 5),
    "na": (f"SELECT iid, c, w || '#' || a AS key FROM (SELECT iid, c, unnest({_NW}) w, {_AW} aw FROM {{T}}), "
           "unnest(aw) u(a)", 30, 5),
    "phx": ("SELECT iid, c, phn AS key FROM {T} WHERE phn <> ''", 30, 30),
}

_PH = [("ph", "f"), ("sh", "s"), ("ch", "k"), ("th", "t"), ("kh", "k"), ("gh", "k"), ("bh", "b"), ("dh", "d"),
       ("jh", "j"), ("x", "ks"), ("[gqc]", "k"), ("b", "p"), ("d", "t"), ("[vw]", "f"), ("z", "s"), ("j", "s")]
_PHC = [(re.compile(a), b) for a, b in _PH]
_VOWELS = re.compile("[aeiouhy]")
_REPEAT = re.compile(r"(.)\1+")
_PH_CACHE: dict[str, str] = {}


def phonetic_word(w: str) -> str:
    """Consonant skeleton of one word (hand-written rules: ph->f, voiced->unvoiced, vowels after the first dropped)."""
    r = _PH_CACHE.get(w)
    if r is None:
        x = w
        for a, b in _PHC:
            x = a.sub(b, x)
        x = x[:1] + _VOWELS.sub("", x[1:])
        r = _PH_CACHE[w] = _REPEAT.sub(r"\1", x)
    return r


def phonetic_name(name: str | None) -> str:
    """Sorted, de-duplicated phonetic skeletons of the words (len >= 2) of a core name, space-joined."""
    return " ".join(sorted({phonetic_word(w) for w in (name or "").split() if len(w) >= 2}))


def _add_phonetic(con: duckdb.DuckDBPyConnection, side: str) -> None:
    """Add column phn to table ``side`` (computed in Python in 1M-row chunks; the per-word cache makes it fast)."""
    full = con.execute(f"SELECT iid, name_core FROM {side}").to_arrow_table()
    con.execute(f"DROP TABLE IF EXISTS {side}_ph")
    first = True
    for b in full.to_batches(max_chunksize=1_000_000):
        tab = pa.table({"iid": b.column(0), "phn": pa.array([phonetic_name(n) for n in b.column(1).to_pylist()])})
        con.register("ph_tab", tab)
        con.execute(f"CREATE TABLE {side}_ph AS SELECT * FROM ph_tab" if first
                    else f"INSERT INTO {side}_ph SELECT * FROM ph_tab")
        con.unregister("ph_tab")
        first = False
    con.execute(f"CREATE OR REPLACE TABLE {side}2 AS SELECT x.*, p.phn FROM {side} x JOIN {side}_ph p USING (iid)")
    con.execute(f"DROP TABLE {side}; DROP TABLE {side}_ph; ALTER TABLE {side}2 RENAME TO {side}")


def load_tables(
    con: duckdb.DuckDBPyConnection, left: str, rights: list[str], v2_cand: str, gt_pairs: str | None
) -> dict[str, int]:
    """Create L, R (integer ids, phonetic names, md5 bucket of the S1 id), v2 (v2 candidates) and optional gt.

    left / rights: normalized parquet paths; v2_cand: v2 candidate parquet (string ids); gt_pairs: labelled pairs
    parquet (s1_id, match_id) for train, None for test. Returns row counts.
    """
    cols = "entity_id, country_norm AS c, name_core, addr_norm, addr_nums"
    con.execute(f"""CREATE OR REPLACE TABLE L AS SELECT (row_number() OVER (ORDER BY entity_id) - 1)::INT AS iid,
                      {cols}, (md5_number(entity_id) % 100)::TINYINT AS bkt FROM read_parquet('{left}')""")
    rlist = ", ".join(f"'{r}'" for r in rights)
    con.execute(f"""CREATE OR REPLACE TABLE R AS SELECT (row_number() OVER (ORDER BY entity_id) - 1)::INT AS iid,
                      {cols} FROM read_parquet([{rlist}])""")
    _add_phonetic(con, "L")
    _add_phonetic(con, "R")
    con.execute(f"""CREATE OR REPLACE TABLE v2 AS SELECT l.iid AS lid, r.iid AS rid, c.* EXCLUDE (lid, rid)
                    FROM read_parquet('{v2_cand}') c JOIN L l ON l.entity_id = c.lid JOIN R r ON r.entity_id = c.rid""")
    if gt_pairs:
        con.execute(f"""CREATE OR REPLACE TABLE gt AS SELECT l.iid AS lid, r.iid AS rid FROM read_parquet('{gt_pairs}') g
                        JOIN L l ON l.entity_id = g.s1_id JOIN R r ON r.entity_id = g.match_id""")
    n = con.execute("SELECT (SELECT count(*) FROM L), (SELECT count(*) FROM R), (SELECT count(*) FROM v2)").fetchone()
    return {"n_left": n[0], "n_right": n[1], "v2_pairs": n[2]}


def run_passes(con: duckdb.DuckDBPyConnection, passes: list[str], buckets: int = 8) -> dict[str, Any]:
    """Run the compound-key passes: tables cand_<pass>(lid, rid, s) with the top-k left ids per right.

    Keys are country-scoped and hashed; a key is kept when its left document frequency <= the pass cap; a pair's
    score is the summed ln(1 + NL / df) of its shared keys. Right records are processed in ``buckets`` hash buckets.
    """
    nl = con.execute("SELECT count(*) FROM L").fetchone()[0]
    stats: dict[str, Any] = {}
    for name in passes:
        sql, cap, k = KEYS[name]
        t = time.time()
        con.execute(f"""CREATE OR REPLACE TEMP TABLE Lk AS
                        SELECT DISTINCT iid, hash(c || '|' || key) AS k FROM ({sql.format(T='L')})""")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE Lkf AS SELECT l.iid, l.k, d.idf FROM Lk l JOIN (
                          SELECT k, ln(1 + {nl}::DOUBLE / count(*)) AS idf FROM Lk GROUP BY k HAVING count(*) <= {cap}
                        ) d USING (k)""")
        con.execute("DROP TABLE Lk")
        con.execute(f"DROP TABLE IF EXISTS cand_{name}")
        for b in range(buckets):
            right = f"(SELECT * FROM R WHERE iid % {buckets} = {b})"
            body = f"""SELECT lid, rid, s FROM (
                SELECT l.iid AS lid, r.iid AS rid, sum(l.idf) AS s
                FROM (SELECT DISTINCT iid, hash(c || '|' || key) AS k FROM ({sql.format(T=right)})) r
                JOIN Lkf l USING (k) GROUP BY ALL)
                QUALIFY row_number() OVER (PARTITION BY rid ORDER BY s DESC, lid) <= {k}"""
            con.execute(f"CREATE TABLE cand_{name} AS {body}" if b == 0 else f"INSERT INTO cand_{name} {body}")
        con.execute("DROP TABLE Lkf")
        stats[f"{name}_pairs"] = con.execute(f"SELECT count(*) FROM cand_{name}").fetchone()[0]
        stats[f"{name}_s"] = round(time.time() - t, 1)
        log.info("block3 pass=%s pairs=%s seconds=%s", name, stats[f"{name}_pairs"], stats[f"{name}_s"])
    return stats


def build_union(con: duckdb.DuckDBPyConnection, out_dir: Path, ranker_path: str | None, buckets: int = 8) -> int:
    """Union v2 candidates with the new passes per right bucket and add the cheap cut features; write parquet.

    Output files out_dir/b<k>.parquet (integer ids): v2 flags and scores (NULL/false for new pairs), new-pass scores,
    CUT_FEATURES, y (label, when table gt exists) and bkt (S1 md5 bucket). block_score of new pairs is the v2 linear
    ranker score (legacy hand weights when ranker_path is None). Returns total rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    has_gt = bool(con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'gt'").fetchone()[0])
    awl, awr = addr_words_sql("l.addr_norm"), addr_words_sql("r.addr_norm")

    def sl(x: str) -> str:
        """SQL: split a space-separated column into a list."""
        return f"string_split({x}, ' ')"

    def jacc(a: str, b: str) -> str:
        """SQL: token Jaccard of two space-separated columns."""
        return (f"len(list_intersect({sl(a)}, {sl(b)}))::DOUBLE / "
                f"greatest(len(list_distinct(list_concat({sl(a)}, {sl(b)}))), 1)")

    score = ranker_sql(load_ranker(ranker_path))
    sflags = ", ".join(f"max(CASE WHEN p = '{p}' THEN s END) AS s_{p}" for p in PASSES_V3)
    ycol = "(g.lid IS NOT NULL) AS y," if has_gt else ""
    yjoin = "LEFT JOIN gt g ON g.lid = U.lid AND g.rid = U.rid" if has_gt else ""
    total = 0
    for b in range(buckets):
        t = time.time()
        npart = " UNION ALL ".join(
            f"SELECT lid, rid, '{p}' AS p, s FROM cand_{p} WHERE rid % {buckets} = {b}" for p in PASSES_V3
        )
        q = f"""
        WITH N AS (SELECT lid, rid, {sflags}, count(*)::INT AS n_pass_new FROM ({npart}) GROUP BY lid, rid),
             V AS (SELECT * FROM v2 WHERE rid % {buckets} = {b}),
             U AS (SELECT coalesce(V.lid, N.lid) AS lid, coalesce(V.rid, N.rid) AS rid, (V.lid IS NOT NULL) AS in_v2,
                     {', '.join(f'coalesce(V.{f}, false) AS {f}' for f in V2_FLAGS)},
                     V.tok_score, V.q3_score, coalesce(V.n_pass, 0) AS n_pass, V.block_score AS v2_block_score,
                     {', '.join(f'N.s_{p}' for p in PASSES_V3)}, coalesce(N.n_pass_new, 0) AS n_pass_new
                   FROM V FULL OUTER JOIN N ON V.lid = N.lid AND V.rid = N.rid)
        SELECT U.*, {ycol}
          jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core,
          jaro_winkler_similarity({awl}, {awr}) AS jw_addr,
          {jacc('l.addr_norm', 'r.addr_norm')} AS jacc_addr,
          (len(list_intersect({sl('l.addr_nums')}, {sl('r.addr_nums')})) > 0)::INT AS num_match,
          jaro_winkler_similarity(replace(l.name_core, ' ', ''), replace(r.name_core, ' ', '')) AS jw_strip,
          jaro_winkler_similarity(l.phn, r.phn) AS jw_phn,
          {jacc('l.name_core', 'r.name_core')} AS jacc_name,
          len(list_intersect({sl('l.addr_nums')}, {sl('r.addr_nums')})) AS n_num,
          len(list_filter({sl('r.addr_norm')}, x -> x NOT IN ('', ','))) AS r_addr_len,
          len(list_filter({sl('r.addr_nums')}, x -> x <> '')) AS r_nnum,
          l.bkt
        FROM U JOIN L l ON l.iid = U.lid JOIN R r ON r.iid = U.rid {yjoin}"""
        dst = out_dir / f"b{b}.parquet"
        con.execute(f"""COPY (SELECT * EXCLUDE (v2_block_score), coalesce(v2_block_score, {score}) AS block_score
                              FROM ({q})) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)""")
        n = pq.ParquetFile(dst).metadata.num_rows
        total += n
        log.info("block3 union bucket=%d/%d rows=%d seconds=%.1f", b + 1, buckets, n, time.time() - t)
    return total


def _cut_matrix(batch: pa.RecordBatch | pa.Table) -> np.ndarray:
    """CUT_FEATURES of a batch as a float32 matrix with NULL -> -1."""
    return np.column_stack([
        np.nan_to_num(batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32), nan=-1.0)
        for c in CUT_FEATURES
    ])


def train_cut(con: duckdb.DuckDBPyConnection, union_dir: Path, fit_where: str, out_path: Path, seed: int) -> dict:
    """Train the cut GBDT on union rows of FIT-group S1 (a quarter of their right records) and save it.

    fit_where: SQL predicate on the S1 md5 bucket column ``bkt``. Returns training stats.
    """
    sel = ", ".join(f"coalesce({c}::FLOAT, -1) AS {c}" for c in CUT_FEATURES)
    d = con.execute(f"""SELECT y::INT AS y, {sel} FROM read_parquet('{union_dir}/b*.parquet')
                        WHERE {fit_where} AND hash(rid) % 4 = 0 ORDER BY lid, rid""").to_arrow_table()
    x, y = _cut_matrix(d), d.column("y").to_numpy()
    del d
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=50, bagging_fraction=0.5,
                  bagging_freq=1, feature_fraction=0.8, verbose=-1, num_threads=8, seed=seed, deterministic=True,
                  force_row_wise=True)
    booster = lgb.train(params, lgb.Dataset(x, y, feature_name=CUT_FEATURES), num_boost_round=300)
    booster.save_model(str(out_path))
    return {"cut_train_rows": int(len(y)), "cut_train_pos": int(y.sum())}


def apply_cut(
    con: duckdb.DuckDBPyConnection, union_dir: Path, model_path: Path, out_path: Path, top_k: int, threshold: float,
    buckets: int = 8,
) -> dict:
    """Score the union with the cut GBDT, rank by the legacy linear score, keep top_k OR p >= threshold; write parquet.

    Output (string ids): v2-compatible candidate columns (flags, tok/q3 scores, jw/jacc/num, n_pass, block_score)
    plus V3_CAND_COLS, so ber.features can read it like a v2 candidate file.
    """
    booster = lgb.Booster(model_file=str(model_path))
    for b in range(buckets):
        src, dst = union_dir / f"b{b}.parquet", union_dir / f"p{b}.parquet"
        writer = None
        for bt in pq.ParquetFile(src).iter_batches(batch_size=2_000_000, columns=["lid", "rid", *CUT_FEATURES]):
            p = booster.predict(_cut_matrix(bt), num_threads=8).astype(np.float32)
            ob = pa.RecordBatch.from_arrays([bt.column("lid"), bt.column("rid"), pa.array(p)],
                                            names=["lid", "rid", "cut_p"])
            writer = writer or pq.ParquetWriter(dst, ob.schema)
            writer.write_batch(ob)
        if writer:
            writer.close()
        log.info("block3 cut scored bucket=%d/%d", b + 1, buckets)
    lin = ("coalesce(u.jw_core, 0) + coalesce(u.jw_addr, 0) + coalesce(u.jacc_addr, 0) "
           "+ 0.3 * (u.n_num > 0)::INT")
    newcols = ", ".join(f"(u.s_{p} IS NOT NULL)::TINYINT AS in_{p}, coalesce(u.s_{p}, 0)::FLOAT AS s_{p}"
                        for p in PASSES_V3)
    tmp = out_path.with_suffix(".tmp.parquet")
    con.execute(f"""COPY (
        WITH k AS (
          SELECT u.*, p.cut_p, row_number() OVER (PARTITION BY u.rid ORDER BY {lin} DESC, u.lid) AS lrank
          FROM read_parquet('{union_dir}/b*.parquet') u JOIN read_parquet('{union_dir}/p*.parquet') p USING (lid, rid))
        SELECT l.entity_id AS lid, r.entity_id AS rid, {', '.join('u.' + f for f in V2_FLAGS)},
          u.tok_score, u.q3_score, u.jw_core, u.jw_addr, u.jacc_addr, u.num_match, u.n_pass, u.block_score,
          {newcols}, u.n_pass_new, u.in_v2::TINYINT AS in_v2, u.jw_strip, u.jw_phn, u.cut_p, u.lrank
        FROM k u JOIN L l ON l.iid = u.lid JOIN R r ON r.iid = u.rid
        WHERE u.lrank <= {top_k} OR u.cut_p >= {threshold}
      ) TO '{tmp}' (FORMAT parquet, COMPRESSION zstd)""")
    tmp.rename(out_path)
    return {"final_pairs": pq.ParquetFile(out_path).metadata.num_rows}


def recall(con: duckdb.DuckDBPyConnection, cand_path: Path, gt_pairs: str, group_where_sql: str | None = None) -> dict:
    """Share of labelled pairs present in a candidate file (optionally only for S1 matching group_where_sql on s1_id)."""
    where = f"WHERE {group_where_sql}" if group_where_sql else ""
    n, hit = con.execute(f"""SELECT count(*), count(c.lid) FROM (SELECT * FROM read_parquet('{gt_pairs}') {where}) g
                             LEFT JOIN read_parquet('{cand_path}') c ON c.lid = g.s1_id AND c.rid = g.match_id""").fetchone()
    nl = con.execute("SELECT count(*) FROM L").fetchone()[0]
    pairs = pq.ParquetFile(cand_path).metadata.num_rows
    return {"recall": hit / max(n, 1), "gt_pairs": n, "pairs_per_s1": pairs / max(nl, 1)}


def write_stats(path: Path, stats: dict) -> None:
    """Write a stats dict as JSON."""
    path.write_text(json.dumps(stats, indent=2, default=str))
