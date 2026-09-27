"""Pattern categories for all holdout true pairs; miss rate and FN counts per category (blocking + decision)."""
from pathlib import Path

import duckdb

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
OUT = ROOT / "notebooks/v3_research/out"
C = ROOT / "artifacts/cache"
con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=4")
con.execute(f"""CREATE TEMP TABLE rn AS SELECT entity_id, name_core, name_skel, addr_norm, script, postal, addr_nums
                FROM read_parquet('{C}/norm_train_s2.parquet')
                UNION ALL SELECT entity_id, name_core, name_skel, addr_norm, script, postal, addr_nums
                FROM read_parquet('{C}/norm_train_s3.parquet')""")
con.execute(f"CREATE TEMP TABLE l AS SELECT entity_id, name_core, name_skel, addr_norm, postal, addr_nums FROM read_parquet('{C}/norm_train_s1.parquet')")
con.execute("CREATE TEMP TABLE core_cnt AS SELECT name_core, count(*) n FROM l GROUP BY 1")
con.execute("CREATE TEMP TABLE nosp_cnt AS SELECT replace(name_core,' ','') k, count(*) n FROM l GROUP BY 1")
con.execute(f"""CREATE TEMP TABLE t AS
  SELECT ts.status, ts.s1_id, ts.match_id, (ts.match_id LIKE 'S3%') AS is_s3,
    l.name_core lc, r.name_core rc, r.script rsc, l.addr_norm la, r.addr_norm ra,
    coalesce(cc.n, 0) AS n_s1_same_rcore,
    coalesce(nc.n, 0) AS n_s1_same_nosp,
    (l.name_core = r.name_core) AS core_eq,
    (r.name_core NOT LIKE '% %' AND l.name_core LIKE '% %' AND replace(l.name_core,' ','') = r.name_core) AS concat_eq,
    (r.name_core NOT LIKE '% %' AND l.name_core LIKE '% %' AND replace(l.name_core,' ','') <> r.name_core
       AND jaro_winkler_similarity(replace(l.name_core,' ',''), r.name_core) > 0.85) AS concat_near,
    (coalesce(r.addr_norm,'') = '') AS r_addr_empty,
    (r.script <> 'latin') AS r_nonlatin,
    jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core,
    (l.postal <> '' AND l.postal = r.postal) AS postal_eq,
    (len(list_intersect(string_split(l.addr_nums,' '), string_split(r.addr_nums,' '))) > 0 AND l.addr_nums <> '') AS num_shared,
    ts.p
  FROM read_parquet('{OUT}/tp_status.parquet') ts
  JOIN l ON l.entity_id = ts.s1_id JOIN rn r ON r.entity_id = ts.match_id
  LEFT JOIN core_cnt cc ON cc.name_core = r.name_core
  LEFT JOIN nosp_cnt nc ON nc.k = replace(r.name_core,' ','')""")
con.execute("""CREATE TEMP TABLE tc AS SELECT *,
  CASE WHEN r_nonlatin THEN 'r_nonlatin_script'
       WHEN concat_eq OR concat_near THEN 'r_name_concatenated(domain-style)'
       WHEN r_addr_empty AND core_eq THEN 'r_addr_empty,name_core_equal'
       WHEN r_addr_empty THEN 'r_addr_empty,name_noisy'
       WHEN jw_core < 0.6 THEN 'name_replaced(jw<0.6),addr_only'
       WHEN core_eq THEN 'name_core_equal'
       ELSE 'name_noisy,addr_present' END AS cat FROM t""")
con.execute(f"COPY tc TO '{OUT}/tp_cat.parquet' (FORMAT parquet)")
q = """SELECT cat, count(*) n_true, sum((status='block_miss')::INT) miss, round(avg((status='block_miss')::INT),4) miss_rate,
         sum((status='dec_fn')::INT) dec_fn, sum((status LIKE 'o2o%')::INT) o2o,
         round(median(n_s1_same_rcore),1) med_s1_same_core
       FROM tc GROUP BY 1 ORDER BY miss DESC"""
for r in con.execute(q).fetchall():
    print(r)
print("crowding among block misses with core_eq:",
      con.execute("""SELECT CASE WHEN n_s1_same_rcore<=1 THEN '1' WHEN n_s1_same_rcore<=3 THEN '2-3'
                 WHEN n_s1_same_rcore<=10 THEN '4-10' ELSE '>10' END b, count(*),
                 sum((status='block_miss')::INT) FROM tc WHERE core_eq GROUP BY 1 ORDER BY 1""").fetchall())
print("S2 vs S3 miss:", con.execute("""SELECT is_s3, count(*), sum((status='block_miss')::INT),
       round(avg((status='block_miss')::INT),4), sum((status='dec_fn')::INT) FROM tc GROUP BY 1""").fetchall())
print("postal_eq / num_shared among misses:", con.execute("""SELECT round(avg(postal_eq::INT),3), round(avg(num_shared::INT),3)
       FROM tc WHERE status='block_miss'""").fetchall())
