"""Benchmark: is union_and_rank's per-pair cost dominated by repeated string_split/list ops in the LATERAL join,
and does precomputing those columns once (+ integer surrogate ids) speed it up?

Runs on a 10% sample of India-country train records (small enough for the 2.5GB/4-thread/8-min research budget).
Writes timings to stdout; findings are copied into research_notes/v3/C_speed.md by hand.
"""
import time
import duckdb

import sys
sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber.blocking import (
    BlockingConfig, build_token_tables, token_pass, sorted_neighbourhood_pass,
    active_passes, AW, load_ranker, ranker_sql, _jacc_sql,
)

L_PATH = "artifacts/cache/norm_train_s1.parquet"
R_PATHS = ["artifacts/cache/norm_train_s2.parquet", "artifacts/cache/norm_train_s3.parquet"]

def setup(con, buckets):
    rights = ", ".join(f"'{p}'" for p in R_PATHS)
    con.execute(f"""CREATE OR REPLACE VIEW L AS
        SELECT * FROM read_parquet('{L_PATH}')
        WHERE country_norm = 'india' AND hash(entity_id) % 8 = 0""")
    con.execute(f"""CREATE OR REPLACE VIEW R AS
        SELECT * FROM read_parquet([{rights}])
        WHERE country_norm = 'india' AND hash(entity_id) % 8 = 0""")
    nl, nr = con.execute("SELECT count(*) FROM L").fetchone()[0], con.execute("SELECT count(*) FROM R").fetchone()[0]
    print(f"sample: n_left={nl} n_right={nr}")
    return nl, nr

def build_candidates(con, cfg):
    t0 = time.time()
    build_token_tables(con, cfg)
    token_pass(con, cfg)
    build_token_tables(con, cfg, name="q3", fields=("q3",))
    token_pass(con, cfg, name="q3", df_cap=cfg.q3_df_cap, df_cap_right=cfg.q3_df_cap_right, top_k=cfg.q3_top_k)
    for name in cfg.sn_keys:
        sorted_neighbourhood_pass(con, name, cfg)
    short = f"len(list_filter(string_split(addr_norm, ' '), x -> x NOT IN ('', ','))) <= {cfg.nameonly_max_addr_tokens}"
    sorted_neighbourhood_pass(con, "nameonly", cfg, key="name_core", window=cfg.nameonly_window, r_where=short)
    print(f"candidate passes built in {time.time()-t0:.1f}s")

def baseline_union_and_rank(con, cfg, passes, bucket):
    """Exact copy of ber.blocking.union_and_rank body, restricted to one bucket, timed."""
    for side in ("L", "R"):
        con.execute(f"CREATE OR REPLACE TABLE {side}x AS SELECT entity_id, name_core, addr_norm, addr_nums, {AW} AS aw FROM {side}")
    score_cols = {"tok": "tok_score", "q3": "q3_score"}
    flags = ", ".join(f"bool_or(pass = '{p}') AS in_{p}" for p in passes)
    n_pass = " + ".join(f"in_{p}::INT" for p in passes)
    scores = ", ".join(f"max({c}) AS {c}" for c in score_cols.values())
    score = ranker_sql(load_ranker(None))
    selects = []
    for p in passes:
        cols = ", ".join(f"{c}" if score_cols.get(p) == c else f"NULL::DOUBLE AS {c}" for c in score_cols.values())
        selects.append(f"SELECT lid, rid, '{p}' AS pass, {cols} FROM cand_{p} WHERE b = {bucket}")
    scored = f"""
        SELECT u.*, jw_core, jw_addr, jacc_addr, num_match, ({n_pass}) AS n_pass
        FROM (SELECT lid, rid, {flags}, {scores}
              FROM ({" UNION ALL ".join(selects)}) GROUP BY lid, rid) u
        JOIN Lx l ON l.entity_id = u.lid
        JOIN (SELECT * FROM Rx WHERE hash(entity_id) % {cfg.buckets} = {bucket}) r ON r.entity_id = u.rid,
        LATERAL (SELECT
          jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core,
          jaro_winkler_similarity(l.aw, r.aw) AS jw_addr,
          {_jacc_sql("l.addr_norm", "r.addr_norm")} AS jacc_addr,
          (len(list_intersect(string_split(l.addr_nums, ' '), string_split(r.addr_nums, ' '))) > 0)::INT AS num_match)"""
    t0 = time.time()
    con.execute(f"CREATE OR REPLACE TEMP TABLE ub_base AS SELECT *, {score} AS block_score FROM ({scored})")
    n = con.execute("SELECT count(*) FROM ub_base").fetchone()[0]
    con.execute("""CREATE OR REPLACE TEMP TABLE cand_base AS
        SELECT * EXCLUDE (rk) FROM (SELECT *, row_number() OVER (PARTITION BY rid ORDER BY block_score DESC, lid) AS rk FROM ub_base)
        WHERE rk <= 3""")
    dt = time.time() - t0
    print(f"BASELINE bucket={bucket} union_rows={n} time={dt:.2f}s")
    return dt, n

def optimized_union_and_rank(con, cfg, passes, bucket):
    """Same logic, but: (1) precompute split/list columns once in Lx2/Rx2 instead of per-pair in LATERAL,
    (2) integer surrogate ids (dense_rank) for the join instead of VARCHAR entity_id."""
    t_pre = time.time()
    con.execute("""CREATE OR REPLACE TABLE id_map AS
        SELECT entity_id, row_number() OVER () AS iid FROM (
          SELECT DISTINCT entity_id FROM L UNION SELECT DISTINCT entity_id FROM R)""")
    for side in ("L", "R"):
        con.execute(f"""CREATE OR REPLACE TABLE {side}x2 AS
            SELECT m.iid, s.name_core, {AW} AS aw,
                   string_split(s.addr_norm, ' ') AS addr_split,
                   list_distinct(string_split(s.addr_norm, ' ')) AS addr_set,
                   string_split(s.addr_nums, ' ') AS nums_split
            FROM {side} s JOIN id_map m USING (entity_id)""")
    for p in passes:
        con.execute(f"""CREATE OR REPLACE TABLE cand_{p}_i AS
            SELECT ml.iid AS liid, mr.iid AS riid, c.* EXCLUDE (lid, rid)
            FROM cand_{p} c JOIN id_map ml ON ml.entity_id = c.lid JOIN id_map mr ON mr.entity_id = c.rid
            WHERE c.b = {bucket}""")
    pre_dt = time.time() - t_pre
    score_cols = {"tok": "tok_score", "q3": "q3_score"}
    flags = ", ".join(f"bool_or(pass = '{p}') AS in_{p}" for p in passes)
    n_pass = " + ".join(f"in_{p}::INT" for p in passes)
    scores = ", ".join(f"max({c}) AS {c}" for c in score_cols.values())
    score = ranker_sql(load_ranker(None))
    selects = []
    for p in passes:
        cols = ", ".join(f"{c}" if score_cols.get(p) == c else f"NULL::DOUBLE AS {c}" for c in score_cols.values())
        selects.append(f"SELECT liid, riid, '{p}' AS pass, {cols} FROM cand_{p}_i")
    scored = f"""
        SELECT u.*, jw_core, jw_addr, jacc_addr, num_match, ({n_pass}) AS n_pass
        FROM (SELECT liid, riid, {flags}, {scores}
              FROM ({" UNION ALL ".join(selects)}) GROUP BY liid, riid) u
        JOIN Lx2 l ON l.iid = u.liid
        JOIN Rx2 r ON r.iid = u.riid,
        LATERAL (SELECT
          jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core,
          jaro_winkler_similarity(l.aw, r.aw) AS jw_addr,
          len(list_intersect(l.addr_set, r.addr_set))::DOUBLE / greatest(len(list_distinct(list_concat(l.addr_set, r.addr_set))), 1) AS jacc_addr,
          (len(list_intersect(l.nums_split, r.nums_split)) > 0)::INT AS num_match)"""
    t0 = time.time()
    con.execute(f"CREATE OR REPLACE TEMP TABLE ub_opt AS SELECT *, {score} AS block_score FROM ({scored})")
    n = con.execute("SELECT count(*) FROM ub_opt").fetchone()[0]
    con.execute("""CREATE OR REPLACE TEMP TABLE cand_opt AS
        SELECT * EXCLUDE (rk) FROM (SELECT *, row_number() OVER (PARTITION BY riid ORDER BY block_score DESC, liid) AS rk FROM ub_opt)
        WHERE rk <= 3""")
    dt = time.time() - t0
    print(f"OPTIMIZED(core) bucket={bucket} union_rows={n} time={dt:.2f}s  (+ {pre_dt:.2f}s one-time id_map/precompute setup, amortized over all buckets)")
    return dt, n, pre_dt

if __name__ == "__main__":
    import os
    os.makedirs("/tmp/duckdb_spill_bench", exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=3")
    con.execute("PRAGMA memory_limit='1.5GB'")
    con.execute("SET temp_directory='/tmp/duckdb_spill_bench'")
    setup(con, buckets=8)
    # scale df caps down to roughly match the ~12.5% sample fraction so caps stay restrictive (avoids
    # near-cross-join blowup seen with un-scaled caps on a small sample)
    cfg = BlockingConfig(buckets=8, token_df_cap=15, token_df_cap_right=65, q3_df_cap=40, q3_df_cap_right=190)
    build_candidates(con, cfg)
    passes = active_passes(cfg)
    print("passes:", passes)
    baseline_union_and_rank(con, cfg, passes, bucket=0)
    optimized_union_and_rank(con, cfg, passes, bucket=0)
    # second bucket to check optimized amortized cost after id_map exists (won't rebuild id_map, but rebuild ok since cheap)
