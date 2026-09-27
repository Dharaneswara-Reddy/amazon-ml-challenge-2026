"""B13: structural shortcuts? ID / row-order correlation between matched records; do distractors have an S1 'parent'?"""
import duckdb, numpy as np
C = "artifacts/cache"; S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb", read_only=True); con.execute(f"SET memory_limit='3GB'; SET threads=4; SET temp_directory='{S}/tmp'")
con.execute(f"""CREATE TEMP TABLE pos AS SELECT entity_id, source, row_number() OVER () AS rn FROM read_parquet(['{C}/raw_train_s1.parquet','{C}/raw_train_s2.parquet','{C}/raw_train_s3.parquet'])""")
d = con.execute(f"""SELECT try_cast(substr(g.s1_id, 4) AS BIGINT) a, try_cast(substr(g.match_id, 4) AS BIGINT) b, p1.rn, p2.rn
   FROM (SELECT * FROM '{C}/train_gt_pairs.parquet' USING SAMPLE 200000 ROWS (reservoir, 1)) g
   JOIN pos p1 ON p1.entity_id = g.s1_id JOIN pos p2 ON p2.entity_id = g.match_id""").fetchnumpy()
def spear(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); return np.corrcoef(rx, ry)[0, 1]
print("spearman id(S1) vs id(match):", round(spear(d["a"], d["b"]), 4), " row order:", round(spear(d["rn"], d["rn_1"] if "rn_1" in d else d[list(d)[3]]), 4))
# distractor structure on subset: unmatched rights (no GT) vs matched: best v2 candidate similarity
r = con.execute("""WITH lab AS (SELECT r.id, EXISTS (SELECT 1 FROM gt WHERE gt.rid = r.id) AS matched FROM R r),
  top AS (SELECT rid, max(jw_core) jw, max(jacc_addr) ja, max(block_score) bs FROM base GROUP BY rid)
  SELECT matched, count(*), avg((jw >= 0.9)::INT), avg((jw >= 0.9 AND ja >= 0.5)::INT), median(jw), median(ja)
  FROM lab JOIN top ON top.rid = lab.id GROUP BY 1""").fetchall()
print("matched?, n, P(top jw_core>=.9), P(jw>=.9 & jacc_addr>=.5), median jw, median jacc_addr:", r)
# does the distractor's closest S1 also have true matches? (parent = best candidate)
r = con.execute(f"""WITH un AS (SELECT r.id FROM R r ANTI JOIN gt ON gt.rid = r.id),
  top1 AS (SELECT lid, rid FROM base QUALIFY row_number() OVER (PARTITION BY rid ORDER BY block_score DESC) = 1)
  SELECT count(*), avg((EXISTS (SELECT 1 FROM '{C}/train_gt_pairs.parquet' g WHERE g.s1_id = t.lid))::INT) FROM un JOIN top1 t ON t.rid = un.id""").fetchall()
print("unmatched rights: n, share whose best S1 candidate has >=1 true match:", r)
print("share of all S1 with >=1 match:", con.execute(f"SELECT (SELECT count(DISTINCT s1_id) FROM '{C}/train_gt_pairs.parquet')::DOUBLE / (SELECT count(*) FROM L)").fetchone())
