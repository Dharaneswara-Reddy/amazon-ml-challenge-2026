"""Build the analysis samples (FIT group only).

dist: unmatched right records (not in GT) sampled by md5(rid) % 100 < 15; parent = candidate S1 with the highest
      jw_core + jw_addr in cand_train; kept only when the parent is in the fit group.
true: GT pairs with the S1 in the fit group, sampled by md5(match_id) % 100 < 5 (all true pairs, in candidates or not).
Also: for distractors, the raw text of the parent's true copies (siblings) to test whether distractors derive from
S1 or from a noisy copy. Output: out/f01_{dist,true,sib}.parquet with raw texts.
"""
from f_common import C, OUT, connect, raw_right
from ber.split import group_where

con = connect()
con.execute(f"CREATE TEMP TABLE gt AS SELECT * FROM read_parquet('{C}/train_gt_pairs.parquet')")
con.execute(f"""CREATE TEMP TABLE dcand AS
  SELECT c.lid, c.rid, c.jw_core, c.jw_addr, c.block_score
  FROM read_parquet('{C}/cand_train.parquet') c
  WHERE md5_number(c.rid) % 100 < 15 AND c.rid NOT IN (SELECT match_id FROM gt)""")
print("dist cand rows", con.execute("SELECT count(*), count(DISTINCT rid) FROM dcand").fetchone(), flush=True)
con.execute(f"""CREATE TEMP TABLE dpar AS
  SELECT rid, arg_max(lid, coalesce(jw_core,0) + coalesce(jw_addr,0)) AS lid,
         max(coalesce(jw_core,0) + coalesce(jw_addr,0)) AS best_s, count(*) AS n_cand
  FROM dcand GROUP BY rid""")
con.execute(f"DELETE FROM dpar WHERE NOT {group_where('lid', 'fit')}")
R = raw_right()
con.execute(f"""COPY (SELECT d.lid, d.rid, d.best_s, d.n_cand, l.business_name ln, l.business_address la, l.country,
                 r.business_name rn, r.business_address ra, r.source
       FROM dpar d JOIN read_parquet('{C}/raw_train_s1.parquet') l ON l.entity_id = d.lid
       JOIN {R} r ON r.entity_id = d.rid) TO '{OUT}/f01_dist.parquet' (FORMAT parquet)""")
con.execute(f"""COPY (SELECT g.s1_id lid, g.match_id rid, l.business_name ln, l.business_address la, l.country,
                 r.business_name rn, r.business_address ra, r.source
       FROM gt g JOIN read_parquet('{C}/raw_train_s1.parquet') l ON l.entity_id = g.s1_id
       JOIN {R} r ON r.entity_id = g.match_id
       WHERE md5_number(g.match_id) % 100 < 5 AND {group_where('g.s1_id', 'fit')}) TO '{OUT}/f01_true.parquet' (FORMAT parquet)""")
con.execute(f"""COPY (SELECT d.rid, g.match_id sid, r.business_name sn, r.business_address sa
       FROM dpar d JOIN gt g ON g.s1_id = d.lid JOIN {R} r ON r.entity_id = g.match_id
       WHERE md5_number(d.rid) % 100 < 5) TO '{OUT}/f01_sib.parquet' (FORMAT parquet)""")
for f in ("dist", "true", "sib"):
    print(f, con.execute(f"SELECT count(*) FROM '{OUT}/f01_{f}.parquet'").fetchone(), flush=True)
print(con.execute(f"SELECT country, source, count(*) FROM '{OUT}/f01_dist.parquet' GROUP BY ALL ORDER BY ALL").fetchall())
print(con.execute(f"SELECT histogram(round(best_s,1)) FROM '{OUT}/f01_dist.parquet'").fetchall())
