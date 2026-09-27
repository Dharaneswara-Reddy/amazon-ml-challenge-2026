"""B10: union (v2 K=3 ∪ new passes) on the right subset with cheap ranking features and GT label -> parquet."""
import duckdb, sys, time
S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb"); con.execute(f"SET memory_limit='3GB'; SET threads=4; SET preserve_insertion_order=false; SET temp_directory='{S}/tmp'")
passes = sys.argv[1].split(",")
t = time.time()
AW = "array_to_string(list_sort(list_filter(string_split(addr_norm, ' '), x -> length(x) >= 2 AND NOT regexp_matches(x, '^[0-9]+$'))), ' ')"
sel = ["SELECT lid, rid, 'base' AS p, block_score AS s FROM base"] + [f"SELECT lid, rid, '{p}', s FROM cand_{p}" for p in passes]
flags = ", ".join(f"max(CASE WHEN p = '{p}' THEN s END) AS s_{p}" for p in ["base", *passes])
con.execute(f"""CREATE OR REPLACE TABLE U AS
  SELECT u.*, (g.lid IS NOT NULL)::INT AS y,
    jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core,
    jaro_winkler_similarity(replace(l.name_core,' ',''), replace(r.name_core,' ','')) AS jw_strip,
    jaro_winkler_similarity(lp.phn, rp.phn) AS jw_phn,
    jaro_winkler_similarity({AW.replace('addr_norm','l.addr_norm')}, {AW.replace('addr_norm','r.addr_norm')}) AS jw_addr,
    len(list_intersect(string_split(l.addr_norm,' '), string_split(r.addr_norm,' ')))::DOUBLE / greatest(len(list_distinct(list_concat(string_split(l.addr_norm,' '), string_split(r.addr_norm,' ')))),1) AS jacc_addr,
    len(list_intersect(string_split(l.name_core,' '), string_split(r.name_core,' ')))::DOUBLE / greatest(len(list_distinct(list_concat(string_split(l.name_core,' '), string_split(r.name_core,' ')))),1) AS jacc_name,
    len(list_intersect(string_split(l.addr_nums,' '), string_split(r.addr_nums,' '))) AS n_num,
    len(list_filter(string_split(r.addr_norm,' '), x -> x NOT IN ('', ','))) AS r_addr_len,
    len(string_split(r.addr_nums,' ')) AS r_nnum
  FROM (SELECT lid, rid, {flags}, count(*) AS n_pass FROM ({' UNION ALL '.join(sel)}) GROUP BY lid, rid) u
  JOIN L l ON l.id = u.lid JOIN R r ON r.id = u.rid
  JOIN L_ph lp ON lp.id = u.lid JOIN R_ph rp ON rp.id = u.rid
  LEFT JOIN gt g ON g.lid = u.lid AND g.rid = u.rid""")
con.execute(f"COPY U TO '{S}/union_feat.parquet' (FORMAT parquet)")
print(con.execute("SELECT count(*), sum(y), count(DISTINCT rid) FROM U").fetchall(), time.time() - t)
