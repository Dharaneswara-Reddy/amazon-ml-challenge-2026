"""Isolated benchmark of union_and_rank's per-pair LATERAL scoring cost (jaro_winkler + jaccard + num_match),
using REAL (lid, rid) pairs from the last full run's cand_train.parquet (post-cut, 30.96M rows) as a stand-in
workload, joined against the real full normalized L/R tables. Avoids re-running multi-pass blocking (which OOM'd
on a small sample due to caps not scaling down -> near-cross-join blowup).

Compares:
  A) baseline: VARCHAR entity_id join + string_split/list_distinct/list_intersect recomputed per pair (current code)
  B) optimized: integer surrogate ids (via a join to a small id_map) + split/list columns precomputed ONCE per
     entity in Lx2/Rx2 (not per pair)
on the same 2M-row pair sample, same threads/memory budget.
"""
import time
import duckdb

L_PATH = "artifacts/cache/norm_train_s1.parquet"
R_PATHS = ["artifacts/cache/norm_train_s2.parquet", "artifacts/cache/norm_train_s3.parquet"]
CAND = "artifacts/cache/cand_train.parquet"
AW = "array_to_string(list_sort(list_filter(string_split(addr_norm, ' '), x -> length(x) >= 2 AND NOT regexp_matches(x, '^[0-9]+$'))), ' ')"

def jacc(a, b):
    sa, sb = f"string_split({a}, ' ')", f"string_split({b}, ' ')"
    return f"len(list_intersect({sa}, {sb}))::DOUBLE / greatest(len(list_distinct(list_concat({sa}, {sb}))), 1)"

def main():
    import os
    os.makedirs("/tmp/duckdb_spill_bench2", exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=3")
    con.execute("PRAGMA memory_limit='1.2GB'")
    con.execute("SET temp_directory='/tmp/duckdb_spill_bench2'")
    con.execute("SET preserve_insertion_order=false")

    rights = ", ".join(f"'{p}'" for p in R_PATHS)
    con.execute(f"""CREATE OR REPLACE TABLE pairs AS
        SELECT lid, rid FROM read_parquet('{CAND}') USING SAMPLE 500000 ROWS""")
    n = con.execute("SELECT count(*) FROM pairs").fetchone()[0]
    print(f"pair sample rows: {n}")

    # Only materialize L/R rows actually referenced by the sampled pairs (not the full 12.5M-row tables) --
    # keeps this benchmark's memory small regardless of dataset size; system is under heavy memory pressure
    # from other agents right now (swap ~full), so this matters.
    con.execute(f"""CREATE OR REPLACE TABLE Lx AS
        SELECT entity_id, name_core, addr_norm, addr_nums, {AW} AS aw
        FROM read_parquet('{L_PATH}') WHERE entity_id IN (SELECT DISTINCT lid FROM pairs)""")
    con.execute(f"""CREATE OR REPLACE TABLE Rx AS
        SELECT entity_id, name_core, addr_norm, addr_nums, {AW} AS aw
        FROM read_parquet([{rights}]) WHERE entity_id IN (SELECT DISTINCT rid FROM pairs)""")
    print(f"Lx rows: {con.execute('SELECT count(*) FROM Lx').fetchone()[0]}  Rx rows: {con.execute('SELECT count(*) FROM Rx').fetchone()[0]}")

    # ---- A) baseline: VARCHAR join + string ops recomputed per pair ----
    t0 = time.time()
    con.execute(f"""CREATE OR REPLACE TEMP TABLE scored_base AS
        SELECT p.lid, p.rid,
               jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core,
               jaro_winkler_similarity(l.aw, r.aw) AS jw_addr,
               {jacc("l.addr_norm", "r.addr_norm")} AS jacc_addr,
               (len(list_intersect(string_split(l.addr_nums, ' '), string_split(r.addr_nums, ' '))) > 0)::INT AS num_match
        FROM pairs p JOIN Lx l ON l.entity_id = p.lid JOIN Rx r ON r.entity_id = p.rid""")
    dt_base = time.time() - t0
    print(f"A) baseline VARCHAR+recompute: {dt_base:.2f}s for {n} pairs ({n/dt_base/1000:.0f}k pairs/s)")

    # ---- B) optimized: integer surrogate ids + precomputed split/list columns ----
    t0 = time.time()
    con.execute("""CREATE OR REPLACE TABLE id_map AS
        SELECT entity_id, row_number() OVER () AS iid FROM (
          SELECT entity_id FROM Lx UNION SELECT entity_id FROM Rx)""")
    con.execute("""CREATE OR REPLACE TABLE Lx2 AS
        SELECT m.iid, l.name_core, l.aw, string_split(l.addr_norm, ' ') AS addr_split,
               list_distinct(string_split(l.addr_norm, ' ')) AS addr_set, string_split(l.addr_nums, ' ') AS nums_split
        FROM Lx l JOIN id_map m USING (entity_id)""")
    con.execute("""CREATE OR REPLACE TABLE Rx2 AS
        SELECT m.iid, r.name_core, r.aw, string_split(r.addr_norm, ' ') AS addr_split,
               list_distinct(string_split(r.addr_norm, ' ')) AS addr_set, string_split(r.addr_nums, ' ') AS nums_split
        FROM Rx r JOIN id_map m USING (entity_id)""")
    con.execute("""CREATE OR REPLACE TABLE pairs_i AS
        SELECT ml.iid AS liid, mr.iid AS riid FROM pairs p
        JOIN id_map ml ON ml.entity_id = p.lid JOIN id_map mr ON mr.entity_id = p.rid""")
    setup_dt = time.time() - t0
    print(f"   (one-time id_map + Lx2/Rx2 precompute: {setup_dt:.2f}s -- this is a FIXED per-run cost, not per-bucket)")

    t0 = time.time()
    con.execute("""CREATE OR REPLACE TEMP TABLE scored_opt AS
        SELECT p.liid, p.riid,
               jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core,
               jaro_winkler_similarity(l.aw, r.aw) AS jw_addr,
               len(list_intersect(l.addr_set, r.addr_set))::DOUBLE / greatest(len(list_distinct(list_concat(l.addr_set, r.addr_set))), 1) AS jacc_addr,
               (len(list_intersect(l.nums_split, r.nums_split)) > 0)::INT AS num_match
        FROM pairs_i p JOIN Lx2 l ON l.iid = p.liid JOIN Rx2 r ON r.iid = p.riid""")
    dt_opt = time.time() - t0
    print(f"B) optimized INT+precomputed: {dt_opt:.2f}s for {n} pairs ({n/dt_opt/1000:.0f}k pairs/s)")
    print(f"speedup (scoring step only, excl. one-time setup): {dt_base/dt_opt:.2f}x")
    # setup is a one-time cost amortized over all 8-16 buckets in the real pipeline (id_map/Lx2/Rx2 built once,
    # not per bucket), so for N buckets the fair comparison is dt_base*N vs setup_dt + dt_opt*N
    for nb in (8, 16):
        print(f"  amortized over {nb} buckets: baseline={dt_base*nb:.0f}s  optimized={setup_dt + dt_opt*nb:.0f}s  "
              f"speedup={dt_base*nb/(setup_dt+dt_opt*nb):.2f}x")

if __name__ == "__main__":
    main()
