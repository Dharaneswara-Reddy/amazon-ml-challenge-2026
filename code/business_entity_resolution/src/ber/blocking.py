"""Candidate generation (blocking) in DuckDB: bounded, multi-pass, country-partitioned.

Left = Source 1 records, Right = Source 2 ∪ Source 3 records. Every true pair in train shares a country and every
right record matches at most one left record, so candidates are generated within country and capped per right
record (top-k left candidates per right record), which bounds the output at k × |Right| pairs.

Business names in this data are highly repetitive (half of Source 1 shares its exact name with another entity) while
name + address is unique, so passes combine name and address evidence:
  tok    IDF-weighted overlap of rare-ish name and address tokens (namespaced n:/a:); frequent tokens are skipped.
  sn_*   sorted neighbourhood on composite keys (name + sorted address words, address words + name, skeleton + house
         numbers, ...): each right record is paired with the w nearest left records in sort order within its country.
Survivors are ranked per right record by a cheap name + address similarity and the top-k kept.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import duckdb

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


@dataclass(frozen=True)
class BlockingConfig:
    """Parameters for all blocking passes."""

    token_min_len: int = 2
    token_fields: tuple[str, ...] = ("name", "addr")
    token_df_cap: int = 100
    token_df_cap_right: int = 500
    token_top_k: int = 5
    sn_window: int = 3
    sn_keys: tuple[str, ...] = DEFAULT_SN_KEYS
    final_top_k: int = 5
    buckets: int = 8
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


def _token_select(side: str, cfg: BlockingConfig) -> str:
    """SQL yielding (entity_id, country_norm, t) with field-namespaced tokens for one side."""
    cols = {"name": ("n:", "name_core"), "addr": ("a:", "addr_norm")}
    parts = [
        f"""SELECT entity_id, country_norm, '{prefix}' || t AS t
            FROM {side}, unnest(string_split({col}, ' ')) u(t) WHERE length(t) >= {cfg.token_min_len}"""
        for prefix, col in (cols[f] for f in cfg.token_fields)
    ]
    return " UNION ALL ".join(parts)


def build_token_tables(con: duckdb.DuckDBPyConnection, cfg: BlockingConfig) -> None:
    """Materialize distinct (id, country, token) tables for L and R and per-country token frequencies."""
    for side in ("L", "R"):
        con.execute(
            f"""CREATE OR REPLACE TABLE {side}tok AS
                SELECT DISTINCT entity_id AS id, country_norm AS c, t FROM ({_token_select(side, cfg)})"""
        )
    con.execute(
        """CREATE OR REPLACE TABLE tokdf AS
           SELECT c, t, coalesce(l.dl, 0) AS dl, coalesce(r.dr, 0) AS dr
           FROM (SELECT c, t, count(*) dl FROM Ltok GROUP BY ALL) l
           FULL JOIN (SELECT c, t, count(*) dr FROM Rtok GROUP BY ALL) r USING (c, t)"""
    )


def estimate_token_pairs(con: duckdb.DuckDBPyConnection, caps: list[int], right_factor: int = 5) -> list[dict]:
    """For each left df cap (right cap = cap*right_factor), return the token-level pair volume sum(dl*dr)."""
    rows = []
    for cap in caps:
        n_pairs, n_tokens = con.execute(
            f"""SELECT coalesce(sum(dl::HUGEINT * dr), 0), count(*)
                FROM tokdf WHERE dl > 0 AND dr > 0 AND dl <= {cap} AND dr <= {cap * right_factor}"""
        ).fetchone()
        rows.append({"cap": cap, "pairs": int(n_pairs), "tokens": n_tokens})
    return rows


def token_pass(con: duckdb.DuckDBPyConnection, cfg: BlockingConfig) -> None:
    """Create temp table cand_tok(lid, rid, tok_score): top-k left ids per right id by summed token IDF.

    Work is split into hash buckets of right ids so each GROUP BY stays small enough for the memory limit.
    """
    con.execute(
        f"""CREATE OR REPLACE TABLE tokkeep AS
            SELECT c, t, ln(1 + (SELECT count(*) FROM R)::DOUBLE / dr) AS idf
            FROM tokdf
            WHERE dl > 0 AND dr > 0 AND dl <= {cfg.token_df_cap} AND dr <= {cfg.token_df_cap_right}"""
    )
    con.execute("CREATE OR REPLACE TABLE cand_tok (lid VARCHAR, rid VARCHAR, tok_score DOUBLE, b INTEGER)")
    for b in range(cfg.buckets):
        con.execute(
            f"""INSERT INTO cand_tok
                SELECT lid, rid, s, {b} FROM (
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
        f"""CREATE OR REPLACE TABLE sn AS
            SELECT id, side, c,
                   sum(CASE WHEN side = 0 THEN 1 ELSE 0 END)
                     OVER (PARTITION BY c ORDER BY k, side, id ROWS UNBOUNDED PRECEDING) AS lcum
            FROM (SELECT entity_id AS id, 0 AS side, country_norm AS c, {key} AS k FROM L
                  UNION ALL
                  SELECT entity_id AS id, 1 AS side, country_norm AS c, {key} AS k FROM R)"""
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


def union_and_rank(con: duckdb.DuckDBPyConnection, cfg: BlockingConfig, passes: list[str]) -> None:
    """Union all pass outputs into table candidates with per-pass flags and cheap similarities; keep top-k per rid.

    block_score = JW(name_core) + JW(sorted address words) + token Jaccard(address) + 0.3·[shared house number]. On a
    2% train sample this ranking keeps 95.2% of true pairs in the top 5 per right record (name-only JW: 81.3%).
    Runs per hash bucket of right ids so the union of all passes (~40 pairs per right record) never has to be held
    or spilled at once.
    """
    for side in ("L", "R"):
        con.execute(
            f"""CREATE OR REPLACE TABLE {side}x AS
                SELECT entity_id, name_core, addr_norm, addr_nums, {AW} AS aw FROM {side}"""
        )
    flags = ", ".join(f"bool_or(pass = '{p}') AS in_{p}" for p in passes)
    con.execute("DROP TABLE IF EXISTS candidates")
    for b in range(cfg.buckets):
        where = f"b = {b}"  # pass tables are stored in bucket order, so zone maps skip other buckets
        selects = [
            f"SELECT lid, rid, '{p}' AS pass, {'tok_score' if p == 'tok' else 'NULL::DOUBLE'} AS tok_score "
            f"FROM cand_{p} WHERE {where}"
            for p in passes
        ]
        body = f"""
            SELECT * EXCLUDE (rk) FROM (
              SELECT u.*, jw_core, jw_addr, jacc_addr, num_match,
                     jw_core + jw_addr + jacc_addr + 0.3 * num_match AS block_score,
                     row_number() OVER (
                       PARTITION BY u.rid ORDER BY jw_core + jw_addr + jacc_addr + 0.3 * num_match DESC, u.lid
                     ) AS rk
              FROM (SELECT lid, rid, {flags}, max(tok_score) AS tok_score
                    FROM ({" UNION ALL ".join(selects)}) GROUP BY lid, rid) u
              JOIN Lx l ON l.entity_id = u.lid
              JOIN Rx r ON r.entity_id = u.rid,
              LATERAL (SELECT
                jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core,
                jaro_winkler_similarity(l.aw, r.aw) AS jw_addr,
                {_jacc_sql("l.addr_norm", "r.addr_norm")} AS jacc_addr,
                (len(list_intersect(string_split(l.addr_nums, ' '), string_split(r.addr_nums, ' '))) > 0)::INT
                  AS num_match))
            WHERE rk <= {cfg.final_top_k}"""
        if b == 0:
            con.execute(f"CREATE TABLE candidates AS {body}")
        else:
            con.execute(f"INSERT INTO candidates {body}")
        log.info("blocking union bucket=%d/%d", b + 1, cfg.buckets)


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
    """Pair completeness overall and per pass flag (after the top-k cut), against GT pairs (s1_id, match_id)."""
    flag_cols = ", ".join(f"avg(coalesce(c.in_{p}, false)::INT) AS recall_{p}" for p in passes)
    row = con.execute(
        f"""SELECT count(*) AS gt_pairs, avg((c.lid IS NOT NULL)::INT) AS recall_final, {flag_cols}
            FROM ({gt_pairs_sql}) g
            LEFT JOIN candidates c ON c.lid = g.s1_id AND c.rid = g.match_id"""
    ).fetchone()
    cols = ["gt_pairs", "recall_final", *[f"recall_{p}" for p in passes]]
    return dict(zip(cols, row, strict=True))
