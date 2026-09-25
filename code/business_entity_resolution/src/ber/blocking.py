"""Candidate generation (blocking) in DuckDB: bounded, multi-pass, country-partitioned — v2 (DESIGN_V2 §S2, D-006).

Left = Source 1 records, Right = Source 2 ∪ Source 3 records. Every true pair in train shares a country and every
right record matches at most one left record, so candidates are generated within country and capped per right
record (top-K left candidates per right record), which bounds the output at K × |Right| pairs.

Business names in this data are highly repetitive (half of Source 1 shares its exact name with another entity) while
name + address is unique, so passes combine name and address evidence:
  tok        IDF-weighted overlap of rare-ish name and address tokens (namespaced n:/a:), top-k per right record.
  q3         IDF-weighted overlap of rare character trigrams of the core name (typos, transliteration), top-k.
  sn_*       sorted neighbourhood on composite keys (name + sorted address words, address words + name, skeleton +
             house numbers, ...): each right record is paired with the w nearest left records within its country.
  nameonly   sorted neighbourhood on the core name with a wider window, only for right records whose address is
             empty or very short (address keys cannot help them).
The union of all passes is ranked per right record by a LINEAR score (weights learned by ``train_ranker`` on a
labelled sample of the uncapped union; legacy hand weights by default) and the top-K kept. A linear score can be
evaluated inside DuckDB, so ranking never leaves the database.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np

log = logging.getLogger(__name__)


def addr_words_sql(col: str = "addr_norm") -> str:
    """SQL for the sorted, space-joined non-numeric address words (order-insensitive address key)."""
    return (
        f"array_to_string(list_sort(list_filter(string_split({col}, ' '), "
        "x -> length(x) >= 2 AND NOT regexp_matches(x, '^[0-9]+$'))), ' ')"
    )


AW = addr_words_sql()
SN_KEYS = {
    "sn_plain": "name_core",
    "sn_rev": "reverse(name_core)",
    "sn_sorted": "array_to_string(list_sort(string_split(name_core, ' ')), ' ')",
    "sn_skel": "name_skel",
    "sn_name": "name_core",
    "sn_name_addr": f"name_core || '|' || {AW}",
    "sn_addr_name": f"{AW} || '|' || name_core",
    "sn_skel_nums": "name_skel || '|' || addr_nums",
    "sn_nums_addr": f"addr_nums || '|' || {AW}",
    "sn_addr_skel": f"{AW} || '|' || name_skel",
}
DEFAULT_SN_KEYS = ("sn_name", "sn_name_addr", "sn_addr_name", "sn_skel_nums", "sn_nums_addr", "sn_addr_skel")
LEGACY_WEIGHTS = {"jw_core": 1.0, "jw_addr": 1.0, "jacc_addr": 1.0, "num_match": 0.3}
RANK_FEATURES = ["jw_core", "jw_addr", "jacc_addr", "num_match", "tok_score", "q3_score", "n_pass"]


@dataclass(frozen=True)
class BlockingConfig:
    """Parameters for all blocking passes and the final ranking/cap."""

    token_min_len: int = 2
    token_fields: tuple[str, ...] = ("name", "addr")
    token_df_cap: int = 100
    token_df_cap_right: int = 500
    token_top_k: int = 5
    q3_enabled: bool = True
    q3_df_cap: int = 300
    q3_df_cap_right: int = 1500
    q3_top_k: int = 5
    sn_window: int = 3
    sn_keys: tuple[str, ...] = DEFAULT_SN_KEYS
    nameonly_enabled: bool = True
    nameonly_window: int = 10
    nameonly_max_addr_tokens: int = 2
    final_top_k: int = 5
    buckets: int = 8
    ranker_path: str | None = None
    dump_union_path: str | None = None
    dump_where: str | None = None
    extra: dict = field(default_factory=dict)


def register_inputs(con: duckdb.DuckDBPyConnection, left_path: str, right_paths: list[str]) -> dict[str, int]:
    """Create views L and R over normalized parquet files and return their row counts.

    Inputs must contain entity_id, name_core, name_skel, addr_norm, addr_nums, country_norm.
    """
    rights = ", ".join(f"'{p}'" for p in right_paths)
    con.execute(f"CREATE OR REPLACE VIEW L AS SELECT * FROM read_parquet('{left_path}')")
    con.execute(f"CREATE OR REPLACE VIEW R AS SELECT * FROM read_parquet([{rights}])")
    n_left = con.execute("SELECT count(*) FROM L").fetchone()[0]
    n_right = con.execute("SELECT count(*) FROM R").fetchone()[0]
    return {"n_left": n_left, "n_right": n_right}


def _tables(name: str) -> tuple[str, str, str]:
    """Table names (left tokens, right tokens, document frequencies) of a token-style pass."""
    return ("Ltok", "Rtok", "tokdf") if name == "tok" else (f"Ltok_{name}", f"Rtok_{name}", f"tokdf_{name}")


def _token_select(side: str, fields: tuple[str, ...], min_len: int) -> str:
    """SQL yielding (entity_id, country_norm, t) with namespaced tokens for one side.

    Fields: "name" (core-name words, n:), "addr" (address words, a:), "q3" (character trigrams of the core name with
    word boundaries marked by '_', q:).
    """
    parts = []
    for f in fields:
        if f in ("name", "addr"):
            prefix, col = ("n:", "name_core") if f == "name" else ("a:", "addr_norm")
            parts.append(
                f"""SELECT entity_id, country_norm, '{prefix}' || t AS t
                    FROM {side}, unnest(string_split({col}, ' ')) u(t) WHERE length(t) >= {min_len}"""
            )
        elif f == "q3":
            parts.append(
                f"""SELECT entity_id, country_norm, 'q:' || substr(s, i, 3) AS t FROM (
                      SELECT entity_id, country_norm, s, unnest(range(1, greatest(length(s) - 1, 1))) AS i
                      FROM (SELECT entity_id, country_norm, '_' || replace(name_core, ' ', '_') || '_' AS s
                            FROM {side} WHERE name_core <> ''))"""
            )
        else:
            raise ValueError(f"unknown token field {f!r}")
    return " UNION ALL ".join(parts)


def build_token_tables(
    con: duckdb.DuckDBPyConnection, cfg: BlockingConfig, name: str = "tok", fields: tuple[str, ...] | None = None
) -> None:
    """Materialize distinct (id, country, token) tables for L and R and per-country token frequencies for a pass."""
    lt, rt, df = _tables(name)
    fields = fields or cfg.token_fields
    for side, table in (("L", lt), ("R", rt)):
        con.execute(
            f"""CREATE OR REPLACE TABLE {table} AS
                SELECT DISTINCT entity_id AS id, country_norm AS c, t
                FROM ({_token_select(side, fields, cfg.token_min_len)})"""
        )
    con.execute(
        f"""CREATE OR REPLACE TABLE {df} AS
            SELECT c, t, coalesce(l.dl, 0) AS dl, coalesce(r.dr, 0) AS dr
            FROM (SELECT c, t, count(*) dl FROM {lt} GROUP BY ALL) l
            FULL JOIN (SELECT c, t, count(*) dr FROM {rt} GROUP BY ALL) r USING (c, t)"""
    )


def estimate_token_pairs(
    con: duckdb.DuckDBPyConnection, caps: list[int], right_factor: int = 5, name: str = "tok"
) -> list[dict]:
    """For each left df cap (right cap = cap*right_factor), return the token-level pair volume sum(dl*dr)."""
    df = _tables(name)[2]
    rows = []
    for cap in caps:
        n_pairs, n_tokens = con.execute(
            f"""SELECT coalesce(sum(dl::HUGEINT * dr), 0), count(*)
                FROM {df} WHERE dl > 0 AND dr > 0 AND dl <= {cap} AND dr <= {cap * right_factor}"""
        ).fetchone()
        rows.append({"cap": cap, "pairs": int(n_pairs), "tokens": n_tokens})
    return rows


def token_pass(
    con: duckdb.DuckDBPyConnection,
    cfg: BlockingConfig,
    name: str = "tok",
    df_cap: int | None = None,
    df_cap_right: int | None = None,
    top_k: int | None = None,
) -> None:
    """Create table cand_<name>(lid, rid, <name>_score, b): top-k left ids per right id by summed token IDF.

    Work is split into hash buckets of right ids (column b) so each GROUP BY stays within the memory limit and the
    union step can read one bucket at a time.
    """
    lt, rt, df = _tables(name)
    cap_l = cfg.token_df_cap if df_cap is None else df_cap
    cap_r = cfg.token_df_cap_right if df_cap_right is None else df_cap_right
    k = cfg.token_top_k if top_k is None else top_k
    con.execute(
        f"""CREATE OR REPLACE TABLE keep_{name} AS
            SELECT c, t, ln(1 + (SELECT count(*) FROM R)::DOUBLE / dr) AS idf
            FROM {df} WHERE dl > 0 AND dr > 0 AND dl <= {cap_l} AND dr <= {cap_r}"""
    )
    con.execute(
        f"CREATE OR REPLACE TABLE cand_{name} (lid VARCHAR, rid VARCHAR, {name}_score DOUBLE, b INTEGER)"
    )
    for b in range(cfg.buckets):
        con.execute(
            f"""INSERT INTO cand_{name}
                SELECT lid, rid, s, {b} FROM (
                  SELECT l.id AS lid, r.id AS rid, sum(k.idf) AS s
                  FROM (SELECT * FROM {rt} WHERE hash(id) % {cfg.buckets} = {b}) r
                  JOIN keep_{name} k USING (c, t)
                  JOIN {lt} l USING (c, t)
                  GROUP BY ALL)
                QUALIFY row_number() OVER (PARTITION BY rid ORDER BY s DESC, lid) <= {k}"""
        )


def sorted_neighbourhood_pass(
    con: duckdb.DuckDBPyConnection,
    name: str,
    cfg: BlockingConfig,
    key: str | None = None,
    window: int | None = None,
    r_where: str | None = None,
) -> None:
    """Create table cand_<name>(lid, rid, sn_dist, b) pairing each right record with the nearest left records.

    Left and right records are sorted together by the pass key within country; a right record sitting after
    `lcum` left records is paired with left ranks lcum-w+1 .. lcum+w. r_where optionally restricts the right side.
    """
    key = key or SN_KEYS[name]
    w = window or cfg.sn_window
    r_filter = f"WHERE {r_where}" if r_where else ""
    con.execute(
        f"""CREATE OR REPLACE TABLE sn AS
            SELECT id, side, c,
                   sum(CASE WHEN side = 0 THEN 1 ELSE 0 END)
                     OVER (PARTITION BY c ORDER BY k, side, id ROWS UNBOUNDED PRECEDING) AS lcum
            FROM (SELECT entity_id AS id, 0 AS side, country_norm AS c, {key} AS k FROM L
                  UNION ALL
                  SELECT entity_id AS id, 1 AS side, country_norm AS c, {key} AS k FROM R {r_filter})"""
    )
    # `target` must be a materialized column: with `l.lcum = r.lcum + off` DuckDB hash-joins on country only and
    # filters afterwards, which is quadratic within a country.
    con.execute(
        f"""CREATE OR REPLACE TABLE cand_{name} AS
            SELECT l.id AS lid, r.id AS rid, min(abs(r.off - 0.5)) AS sn_dist, hash(r.id) % {cfg.buckets} AS b
            FROM (SELECT s.id, s.c, o.off, s.lcum + o.off AS target
                  FROM sn s CROSS JOIN range({-w + 1}, {w + 1}) o(off) WHERE s.side = 1) r
            JOIN (SELECT * FROM sn WHERE side = 0) l ON l.c = r.c AND l.lcum = r.target
            GROUP BY lid, rid, b
            ORDER BY b"""
    )


def _jacc_sql(a: str, b: str) -> str:
    """SQL for token-set Jaccard between two space-separated string expressions."""
    sa, sb = f"string_split({a}, ' ')", f"string_split({b}, ' ')"
    return f"len(list_intersect({sa}, {sb}))::DOUBLE / greatest(len(list_distinct(list_concat({sa}, {sb}))), 1)"


def load_ranker(path: str | None) -> dict:
    """Ranker weights {feature: weight, "bias": b} from JSON, or the legacy hand weights when path is None."""
    if not path:
        return {**LEGACY_WEIGHTS, "bias": 0.0}
    return json.loads(Path(path).read_text())["weights"]


def ranker_sql(weights: dict) -> str:
    """SQL expression of the linear ranking score over the union's cheap features."""
    terms = [f"{float(w)!r} * coalesce({f}, 0)" for f, w in weights.items() if f != "bias" and w != 0]
    return " + ".join([repr(float(weights.get("bias", 0.0))), *terms])


def union_and_rank(con: duckdb.DuckDBPyConnection, cfg: BlockingConfig, passes: list[str]) -> None:
    """Union all pass outputs into table candidates with per-pass flags and cheap similarities; keep top-K per rid.

    block_score is the linear ranker score (legacy: JW(name_core) + JW(sorted address words) + address Jaccard +
    0.3·[shared house number], which kept 95.2% of true pairs at top-5 on a 2% train sample). Runs per hash bucket of
    right ids so the union (~40 pairs per right record) is never held or spilled at once. With cfg.dump_union_path
    the UNCAPPED union rows of right ids matching cfg.dump_where are also written for ranker training.
    """
    for side in ("L", "R"):
        con.execute(
            f"""CREATE OR REPLACE TABLE {side}x AS
                SELECT entity_id, name_core, addr_norm, addr_nums, {AW} AS aw FROM {side}"""
        )
    score_cols = {"tok": "tok_score", "q3": "q3_score"}
    flags = ", ".join(f"bool_or(pass = '{p}') AS in_{p}" for p in passes)
    n_pass = " + ".join(f"in_{p}::INT" for p in passes)
    scores = ", ".join(f"max({c}) AS {c}" for c in score_cols.values())
    score = ranker_sql(load_ranker(cfg.ranker_path))
    con.execute("DROP TABLE IF EXISTS candidates")
    con.execute("DROP TABLE IF EXISTS union_dump")
    for b in range(cfg.buckets):
        selects = []
        for p in passes:
            cols = ", ".join(
                f"{c}" if score_cols.get(p) == c else f"NULL::DOUBLE AS {c}" for c in score_cols.values()
            )
            selects.append(f"SELECT lid, rid, '{p}' AS pass, {cols} FROM cand_{p} WHERE b = {b}")
        scored = f"""
            SELECT u.*, jw_core, jw_addr, jacc_addr, num_match, ({n_pass}) AS n_pass
            FROM (SELECT lid, rid, {flags}, {scores}
                  FROM ({" UNION ALL ".join(selects)}) GROUP BY lid, rid) u
            JOIN Lx l ON l.entity_id = u.lid
            JOIN (SELECT * FROM Rx WHERE hash(entity_id) % {cfg.buckets} = {b}) r ON r.entity_id = u.rid,
            LATERAL (SELECT
              jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core,
              jaro_winkler_similarity(l.aw, r.aw) AS jw_addr,
              {_jacc_sql("l.addr_norm", "r.addr_norm")} AS jacc_addr,
              (len(list_intersect(string_split(l.addr_nums, ' '), string_split(r.addr_nums, ' '))) > 0)::INT
                AS num_match)"""
        con.execute(f"CREATE OR REPLACE TEMP TABLE ub AS SELECT *, {score} AS block_score FROM ({scored})")
        if cfg.dump_union_path:
            dump = f"SELECT * FROM ub WHERE {cfg.dump_where or 'TRUE'}"
            con.execute(f"CREATE TABLE union_dump AS {dump}" if b == 0 else f"INSERT INTO union_dump {dump}")
        body = f"""SELECT * EXCLUDE (rk) FROM (
                     SELECT *, row_number() OVER (PARTITION BY rid ORDER BY block_score DESC, lid) AS rk FROM ub)
                   WHERE rk <= {cfg.final_top_k}"""
        con.execute(f"CREATE TABLE candidates AS {body}" if b == 0 else f"INSERT INTO candidates {body}")
        log.info("blocking union bucket=%d/%d", b + 1, cfg.buckets)
    if cfg.dump_union_path:
        con.execute(f"COPY union_dump TO '{cfg.dump_union_path}' (FORMAT parquet, COMPRESSION zstd)")


def active_passes(cfg: BlockingConfig) -> list[str]:
    """Names of the passes a config runs, in union order."""
    return ["tok", *(["q3"] if cfg.q3_enabled else []), *cfg.sn_keys, *(["nameonly"] if cfg.nameonly_enabled else [])]


def run_blocking(con: duckdb.DuckDBPyConnection, cfg: BlockingConfig) -> dict:
    """Run every configured pass on views L and R and leave the result in table `candidates`; return pass stats."""
    stats: dict = {}
    t0 = time.time()

    def mark(step: str, table: str | None = None) -> None:
        """Record elapsed seconds (and pair count) for a finished step and log it."""
        if table:
            stats[f"{step}_pairs"] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        stats[f"{step}_s"] = round(time.time() - t0, 1)
        log.info("blocking step=%s elapsed_s=%s pairs=%s", step, stats[f"{step}_s"], stats.get(f"{step}_pairs"))

    build_token_tables(con, cfg)
    mark("token_tables")
    token_pass(con, cfg)
    mark("tok", "cand_tok")
    if cfg.q3_enabled:
        build_token_tables(con, cfg, name="q3", fields=("q3",))
        token_pass(con, cfg, name="q3", df_cap=cfg.q3_df_cap, df_cap_right=cfg.q3_df_cap_right, top_k=cfg.q3_top_k)
        mark("q3", "cand_q3")
    for name in cfg.sn_keys:
        sorted_neighbourhood_pass(con, name, cfg)
        mark(name, f"cand_{name}")
    if cfg.nameonly_enabled:
        short = (
            "len(list_filter(string_split(addr_norm, ' '), x -> x NOT IN ('', ','))) "
            f"<= {cfg.nameonly_max_addr_tokens}"
        )
        sorted_neighbourhood_pass(con, "nameonly", cfg, key="name_core", window=cfg.nameonly_window, r_where=short)
        mark("nameonly", "cand_nameonly")
    union_and_rank(con, cfg, active_passes(cfg))
    mark("final", "candidates")
    log.info("blocking done %s", " ".join(f"{k}={v}" for k, v in stats.items()))
    return stats


def recall_by_pass(con: duckdb.DuckDBPyConnection, gt_pairs_sql: str, passes: list[str]) -> dict:
    """Pair completeness overall and per pass flag (after the top-K cut), against GT pairs (s1_id, match_id)."""
    flag_cols = ", ".join(f"avg(coalesce(c.in_{p}, false)::INT) AS recall_{p}" for p in passes)
    row = con.execute(
        f"""SELECT count(*) AS gt_pairs, avg((c.lid IS NOT NULL)::INT) AS recall_final, {flag_cols}
            FROM ({gt_pairs_sql}) g
            LEFT JOIN candidates c ON c.lid = g.s1_id AND c.rid = g.match_id"""
    ).fetchone()
    cols = ["gt_pairs", "recall_final", *[f"recall_{p}" for p in passes]]
    return dict(zip(cols, row, strict=True))


def _logistic_fit(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 25) -> tuple[np.ndarray, float]:
    """L2-regularized logistic regression by Newton-IRLS on standardized features; returns (weights, bias) raw scale."""
    mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-9
    z = (x - mu) / sd
    zb = np.column_stack([z, np.ones(len(z))])
    beta = np.zeros(zb.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(zb @ beta)))
        w = p * (1 - p) + 1e-9
        grad = zb.T @ (p - y) + l2 * np.r_[beta[:-1], 0.0]
        hess = (zb * w[:, None]).T @ zb + l2 * np.diag(np.r_[np.ones(len(beta) - 1), 0.0])
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    weights = beta[:-1] / sd
    bias = float(beta[-1] - (beta[:-1] * mu / sd).sum())
    return weights, bias


def recall_at_k(rid_codes: np.ndarray, score: np.ndarray, label: np.ndarray, n_true_pairs: int, k_max: int) -> list:
    """Recall of true pairs kept when each right record keeps its top-k by score, for k = 1..k_max."""
    order = np.lexsort((-score, rid_codes))
    rid_sorted = rid_codes[order]
    start = np.r_[0, np.flatnonzero(rid_sorted[1:] != rid_sorted[:-1]) + 1]
    rank = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    lab = label[order]
    return [float(lab[rank < k].sum() / max(n_true_pairs, 1)) for k in range(1, k_max + 1)]


def train_ranker(
    con: duckdb.DuckDBPyConnection, dump_path: str, gt_pairs_sql: str, out_path: str, k_max: int = 15
) -> dict:
    """Fit the linear blocking ranker on a labelled sample of the uncapped union and choose the cap K.

    Label = pair is a true match. K = smallest cap whose recall is within 0.002 of the recall at k_max (the plateau);
    the legacy hand-weighted ranking's curve is stored alongside for comparison. Writes JSON {weights, k, curves}.
    """
    feats = ", ".join(f"coalesce({f}, 0)::DOUBLE AS {f}" for f in RANK_FEATURES)
    data = con.execute(
        f"""SELECT dense_rank() OVER (ORDER BY u.rid) AS rc, {feats}, (g.s1_id IS NOT NULL)::INT AS y
            FROM read_parquet('{dump_path}') u
            LEFT JOIN ({gt_pairs_sql}) g ON g.s1_id = u.lid AND g.match_id = u.rid"""
    ).fetchnumpy()
    n_true = int(
        con.execute(
            f"""SELECT count(*) FROM ({gt_pairs_sql}) g
                WHERE g.match_id IN (SELECT DISTINCT rid FROM read_parquet('{dump_path}'))"""
        ).fetchone()[0]
    )
    x = np.column_stack([data[f] for f in RANK_FEATURES])
    y = data["y"].astype(np.float64)
    w, bias = _logistic_fit(x, y)
    weights = {**{f: float(v) for f, v in zip(RANK_FEATURES, w, strict=True)}, "bias": bias}
    rc = data["rc"].astype(np.int64)
    learned = recall_at_k(rc, x @ w + bias, y, n_true, k_max)
    legacy_w = np.array([LEGACY_WEIGHTS.get(f, 0.0) for f in RANK_FEATURES])
    legacy = recall_at_k(rc, x @ legacy_w, y, n_true, k_max)
    plateau = learned[-1]
    k = next(i + 1 for i, r in enumerate(learned) if r >= plateau - 0.002)
    out = {"weights": weights, "k": k, "recall_at_k_learned": learned, "recall_at_k_legacy": legacy,
           "n_pairs": len(y), "n_true_pairs": n_true}
    Path(out_path).write_text(json.dumps(out, indent=2))
    log.info("ranker trained k=%d recall@k=%.4f legacy@k=%.4f plateau=%.4f", k, learned[k - 1], legacy[k - 1], plateau)
    return out
