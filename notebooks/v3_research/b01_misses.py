"""B01: materialize missed true pairs (GT not in cand_train) and split never-generated vs cut-by-K on union_sample."""
import duckdb, time
C = "artifacts/cache"
S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb"); con.execute("SET memory_limit='3GB'; SET threads=4; SET preserve_insertion_order=false")
t = time.time()
con.execute(f"""CREATE OR REPLACE TABLE miss AS
  SELECT g.s1_id AS lid, g.match_id AS rid FROM '{C}/train_gt_pairs.parquet' g
  ANTI JOIN (SELECT lid, rid FROM '{C}/cand_train.parquet') c ON c.lid = g.s1_id AND c.rid = g.match_id""")
n_gt = con.execute(f"SELECT count(*) FROM '{C}/train_gt_pairs.parquet'").fetchone()[0]
n_miss = con.execute("SELECT count(*) FROM miss").fetchone()[0]
print("gt", n_gt, "miss", n_miss, "recall", 1 - n_miss / n_gt, time.time() - t)
# union sample split
con.execute(f"CREATE OR REPLACE TABLE us_r AS SELECT DISTINCT rid FROM '{C}/learned/union_sample.parquet'")
r = con.execute(f"""SELECT count(*) AS miss_in_sample,
    sum((u.lid IS NOT NULL)::INT) AS generated_but_cut
  FROM miss m JOIN us_r USING (rid)
  LEFT JOIN (SELECT lid, rid FROM '{C}/learned/union_sample.parquet') u USING (lid, rid)""").fetchone()
print("misses whose rid in union_sample:", r)
tot = con.execute(f"""SELECT count(*) FROM '{C}/train_gt_pairs.parquet' g JOIN us_r ON us_r.rid = g.match_id""").fetchone()[0]
print("gt pairs with rid in sample", tot)
# country split of misses
print(con.execute(f"""SELECT country_norm, count(*) FROM miss m JOIN '{C}/norm_train_s1.parquet' l ON l.entity_id = m.lid GROUP BY 1""").fetchall())
print(con.execute(f"""SELECT l.country_norm, count(*) FROM '{C}/train_gt_pairs.parquet' g JOIN '{C}/norm_train_s1.parquet' l ON l.entity_id = g.s1_id GROUP BY 1""").fetchall())
print(con.execute("SELECT substr(rid,1,2), count(*) FROM miss GROUP BY 1").fetchall())
print(time.time() - t)
