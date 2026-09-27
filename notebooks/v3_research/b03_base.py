"""B03: base tables in the scratch DB: L (all S1 train), R subset (5% of rights by hash), GT on subset, left token df."""
import duckdb, time
C = "artifacts/cache"
S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb"); con.execute(f"SET memory_limit='3GB'; SET threads=4; SET preserve_insertion_order=false; SET temp_directory='{S}/tmp'")
t = time.time()
cols = "entity_id AS id, country_norm AS c, name_core, name_skel, addr_norm, addr_nums, addr_house, addr_street, addr_city, script"
con.execute(f"CREATE OR REPLACE TABLE L AS SELECT {cols} FROM '{C}/norm_train_s1.parquet'")
con.execute(f"""CREATE OR REPLACE TABLE R AS SELECT {cols} FROM read_parquet(['{C}/norm_train_s2.parquet','{C}/norm_train_s3.parquet'])
               WHERE hash(entity_id) % 20 = 0""")
con.execute(f"""CREATE OR REPLACE TABLE gt AS SELECT g.s1_id AS lid, g.match_id AS rid FROM '{C}/train_gt_pairs.parquet' g
               SEMI JOIN R ON R.id = g.match_id""")
con.execute(f"""CREATE OR REPLACE TABLE base AS SELECT lid, rid, jw_core, jw_addr, jacc_addr, num_match, tok_score, q3_score, n_pass, block_score
               FROM '{C}/cand_train.parquet' c SEMI JOIN R ON R.id = c.rid""")
con.execute("ALTER TABLE gt ADD COLUMN found BOOLEAN")
con.execute("UPDATE gt SET found = EXISTS (SELECT 1 FROM base b WHERE b.lid = gt.lid AND b.rid = gt.rid)")
print(con.execute("SELECT (SELECT count(*) FROM L), (SELECT count(*) FROM R), count(*), avg(found::INT) FROM gt").fetchall(), time.time()-t)
# left token df (name and address words), per country
con.execute("""CREATE OR REPLACE TABLE ldf AS SELECT c, t, count(DISTINCT id) AS dl FROM (
    SELECT id, c, 'n:' || t AS t FROM L, unnest(string_split(name_core, ' ')) u(t) WHERE length(t) >= 1
    UNION ALL SELECT id, c, 'a:' || t FROM L, unnest(string_split(addr_norm, ' ')) u(t) WHERE length(t) >= 1 AND t <> ',')
  GROUP BY ALL""")
print(con.execute("SELECT count(*) FROM ldf").fetchall(), time.time()-t)
