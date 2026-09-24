"""Candidate generation (blocking) in DuckDB: bounded, multi-pass, country-partitioned.

Left = Source 1 records, Right = Source 2 ∪ Source 3 records. Every true pair in train shares a country and every
right record matches at most one left record, so candidates are generated within country and capped per right
record (top-k left candidates per right record), which bounds the output at k × |Right| pairs.

Passes:
  tok    rare-token overlap on name_core, IDF-weighted; tokens more frequent than a cap are skipped.
  sn_*   sorted neighbourhood on a name key (plain, reversed, token-sorted): each right record is paired with the
         w nearest left records in sort order within its country.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import duckdb

log = logging.getLogger(__name__)

SN_KEYS = {
    "sn_plain": "name_core",
    "sn_rev": "reverse(name_core)",
    "sn_sorted": "array_to_string(list_sort(string_split(name_core, ' ')), ' ')",
    "sn_skel": "name_skel",
}


@dataclass(frozen=True)
class BlockingConfig:
    """Parameters for all blocking passes."""

    token_min_len: int = 2
    token_df_cap: int = 300
    token_top_k: int = 5
    sn_window: int = 3
    sn_keys: tuple[str, ...] = ("sn_plain", "sn_rev", "sn_sorted")
    final_top_k: int = 5
    buckets: int = 8
    extra: dict = field(default_factory=dict)


def register_inputs(con: duckdb.DuckDBPyConnection, left_path: str, right_paths: list[str]) -> dict[str, int]:
    """Create views L and R over normalized parquet files and return their row counts.

    Inputs must contain entity_id, name_core, country_norm (plus other normalized columns used by features).
    """
    rights = ", ".join(f"'{p}'" for p in right_paths)
    con.execute(f"CREATE OR REPLACE VIEW L AS SELECT * FROM read_parquet('{left_path}')")
    con.execute(f"CREATE OR REPLACE VIEW R AS SELECT * FROM read_parquet([{rights}])")
    n_left = con.execute("SELECT count(*) FROM L").fetchone()[0]
    n_right = con.execute("SELECT count(*) FROM R").fetchone()[0]
    return {"n_left": n_left, "n_right": n_right}


def build_token_tables(con: duckdb.DuckDBPyConnection, cfg: BlockingConfig) -> None:
    """Materialize distinct (id, country, token) tables for L and R and per-country token frequencies."""
    for side in ("L", "R"):
        con.execute(
            f"""CREATE OR REPLACE TEMP TABLE {side}tok AS
                SELECT DISTINCT entity_id AS id, country_norm AS c, t
                FROM {side}, unnest(string_split(name_core, ' ')) u(t)
                WHERE length(t) >= {cfg.token_min_len}"""
        )
    con.execute(
        """CREATE OR REPLACE TEMP TABLE tokdf AS
           SELECT c, t, coalesce(l.dl, 0) AS dl, coalesce(r.dr, 0) AS dr
           FROM (SELECT c, t, count(*) dl FROM Ltok GROUP BY ALL) l
           FULL JOIN (SELECT c, t, count(*) dr FROM Rtok GROUP BY ALL) r USING (c, t)"""
    )


def estimate_token_pairs(con: duckdb.DuckDBPyConnection, caps: list[int]) -> list[dict]:
    """For each df cap, return the pair volume sum(dl*dr) the token pass would generate (before top-k)."""
    rows = []
    for cap in caps:
        n_pairs, n_tokens = con.execute(
            f"""SELECT coalesce(sum(dl::HUGEINT * dr), 0), count(*)
                FROM tokdf WHERE dl > 0 AND dr > 0 AND dl <= {cap} AND dr <= {cap}"""
        ).fetchone()
        rows.append({"cap": cap, "pairs": int(n_pairs), "tokens": n_tokens})
    return rows


def token_pass(con: duckdb.DuckDBPyConnection, cfg: BlockingConfig) -> None:
    """Create temp table cand_tok(lid, rid, tok_score): top-k left ids per right id by summed token IDF.

    Work is split into hash buckets of right ids so each GROUP BY stays small enough for the memory limit.
    """
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE tokkeep AS
            SELECT c, t, ln(1 + (SELECT count(*) FROM R)::DOUBLE / dr) AS idf
            FROM tokdf
            WHERE dl > 0 AND dr > 0 AND dl <= {cfg.token_df_cap} AND dr <= {cfg.token_df_cap}"""
    )
    con.execute("CREATE OR REPLACE TEMP TABLE cand_tok (lid VARCHAR, rid VARCHAR, tok_score DOUBLE)")
    for b in range(cfg.buckets):
        con.execute(
            f"""INSERT INTO cand_tok
                SELECT lid, rid, s FROM (
                  SELECT l.id AS lid, r.id AS rid, sum(k.idf) AS s
                  FROM (SELECT * FROM Rtok WHERE hash(id) % {cfg.buckets} = {b}) r
                  JOIN tokkeep k USING (c, t)
                  JOIN Ltok l USING (c, t)
                  GROUP BY ALL)
                QUALIFY row_number() OVER (PARTITION BY rid ORDER BY s DESC, lid) <= {cfg.token_top_k}"""
        )


def sorted_neighbourhood_pass(con: duckdb.DuckDBPyConnection, name: str, cfg: BlockingConfig) -> None:
    """Create temp table cand_<name>(lid, rid, sn_dist) pairing each right record with the nearest left records.

    Left and right records are sorted together by the pass key within country; a right record sitting after
    `lcum` left records is paired with left ranks lcum-w+1 .. lcum+w.
    """
    key = SN_KEYS[name]
    w = cfg.sn_window
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE sn AS
            SELECT id, side, c,
                   sum(CASE WHEN side = 0 THEN 1 ELSE 0 END)
                     OVER (PARTITION BY c ORDER BY k, side, id ROWS UNBOUNDED PRECEDING) AS lcum
            FROM (SELECT entity_id AS id, 0 AS side, country_norm AS c, {key} AS k FROM L
                  UNION ALL
                  SELECT entity_id AS id, 1 AS side, country_norm AS c, {key} AS k FROM R)"""
    )
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE cand_{name} AS
            SELECT l.id AS lid, r.id AS rid, min(abs(o.off - 0.5)) AS sn_dist
            FROM (SELECT * FROM sn WHERE side = 1) r
            CROSS JOIN range({-w + 1}, {w + 1}) o(off)
            JOIN (SELECT * FROM sn WHERE side = 0) l ON l.c = r.c AND l.lcum = r.lcum + o.off
            GROUP BY ALL"""
    )


def union_and_rank(con: duckdb.DuckDBPyConnection, cfg: BlockingConfig, passes: list[str]) -> None:
    """Union all pass outputs into table candidates with per-pass flags and a cheap similarity; keep top-k per rid.

    The cheap score is Jaro-Winkler on name_core plus a bonus for token-pass hits; it only decides which
    candidates survive the per-right-record cap, the matching model re-scores everything.
    """
    selects = []
    for p in passes:
        score_col = "tok_score" if p == "tok" else "NULL::DOUBLE AS tok_score"
        selects.append(f"SELECT lid, rid, '{p}' AS pass, {score_col} FROM cand_{p}")
    flags = ", ".join(f"bool_or(pass = '{p}') AS in_{p}" for p in passes)
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE cand_union AS
            SELECT lid, rid, {flags}, max(tok_score) AS tok_score
            FROM ({" UNION ALL ".join(selects)}) GROUP BY lid, rid"""
    )
    con.execute(
        f"""CREATE OR REPLACE TABLE candidates AS
            SELECT u.*, jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core
            FROM cand_union u
            JOIN L l ON l.entity_id = u.lid
            JOIN R r ON r.entity_id = u.rid
            QUALIFY row_number() OVER (
              PARTITION BY u.rid
              ORDER BY jaro_winkler_similarity(l.name_core, r.name_core)
                       + 0.05 * coalesce(u.tok_score, 0) DESC, u.lid) <= {cfg.final_top_k}"""
    )


def run_blocking(con: duckdb.DuckDBPyConnection, cfg: BlockingConfig) -> dict:
    """Run every configured pass on views L and R and leave the result in table `candidates`; return pair counts."""
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
    for name in cfg.sn_keys:
        sorted_neighbourhood_pass(con, name, cfg)
        mark(name, f"cand_{name}")
    union_and_rank(con, cfg, ["tok", *cfg.sn_keys])
    mark("final", "candidates")
    log.info("blocking done %s", " ".join(f"{k}={v}" for k, v in stats.items()))
    return stats


def recall_by_pass(con: duckdb.DuckDBPyConnection, gt_pairs_sql: str, passes: list[str]) -> dict:
    """Pair completeness overall and per pass flag, against GT pairs given as SQL with columns (s1_id, match_id)."""
    flag_cols = ", ".join(f"avg(coalesce(c.in_{p}, false)::INT) AS recall_{p}" for p in passes)
    row = con.execute(
        f"""SELECT count(*) AS gt_pairs, avg((c.lid IS NOT NULL)::INT) AS recall_final, {flag_cols}
            FROM ({gt_pairs_sql}) g
            LEFT JOIN candidates c ON c.lid = g.s1_id AND c.rid = g.match_id"""
    ).fetchone()
    cols = ["gt_pairs", "recall_final", *[f"recall_{p}" for p in passes]]
    return dict(zip(cols, row, strict=True))
