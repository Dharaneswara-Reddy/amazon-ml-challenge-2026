"""B02: 20k random missed pairs + 20k found pairs with raw and normalized fields of both sides -> scratch parquet."""
import duckdb
C = "artifacts/cache"
S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb"); con.execute("SET memory_limit='3GB'; SET threads=4; SELECT setseed(0.42)")
con.execute("CREATE OR REPLACE TABLE ms AS SELECT lid, rid, 1 AS miss FROM miss USING SAMPLE 20000 ROWS (reservoir, 7)")
con.execute(f"""CREATE OR REPLACE TABLE fs AS SELECT g.s1_id AS lid, g.match_id AS rid, 0 AS miss
  FROM (SELECT * FROM '{C}/train_gt_pairs.parquet' USING SAMPLE 40000 ROWS (reservoir, 7)) g
  ANTI JOIN miss m ON m.lid = g.s1_id AND m.rid = g.match_id LIMIT 20000""")
cols = "entity_id, name_core, name_skel, addr_norm, addr_nums, script, country_norm, addr_house, addr_street, addr_city, addr_admin, legal_form"
con.execute(f"""CREATE OR REPLACE TABLE pairs AS
  WITH p AS (SELECT * FROM ms UNION ALL SELECT * FROM fs),
  N AS (SELECT {cols} FROM read_parquet(['{C}/norm_train_s1.parquet','{C}/norm_train_s2.parquet','{C}/norm_train_s3.parquet'])
        WHERE entity_id IN (SELECT lid FROM p UNION SELECT rid FROM p)),
  RW AS (SELECT entity_id, business_name, business_address FROM read_parquet(['{C}/raw_train_s1.parquet','{C}/raw_train_s2.parquet','{C}/raw_train_s3.parquet'])
        WHERE entity_id IN (SELECT lid FROM p UNION SELECT rid FROM p))
  SELECT p.*, rl.business_name AS l_rname, rl.business_address AS l_raddr, rr.business_name AS r_rname, rr.business_address AS r_raddr,
    COLUMNS(nl.*) AS "l_\\0", COLUMNS(nr.*) AS "r_\\0"
  FROM p JOIN RW rl ON rl.entity_id = p.lid JOIN RW rr ON rr.entity_id = p.rid
  JOIN N nl ON nl.entity_id = p.lid JOIN N nr ON nr.entity_id = p.rid""")
con.execute(f"COPY pairs TO '{S}/pairs_sample.parquet' (FORMAT parquet)")
print(con.execute("SELECT miss, count(*) FROM pairs GROUP BY 1").fetchall())
print([r[0] for r in con.execute("DESCRIBE pairs").fetchall()])
