"""B12: S2<->S3 transitivity potential: residual misses (l, r) where another right r' of l is a candidate and r' ~ r by name."""
import duckdb, sys
C = "artifacts/cache"; S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb", read_only=True); con.execute(f"SET memory_limit='3GB'; SET threads=4; SET temp_directory='{S}/tmp'")
passes = sys.argv[1].split(",")
u = " UNION ".join(["SELECT lid, rid FROM base"] + [f"SELECT lid, rid FROM cand_{p}" for p in passes])
con.execute(f"CREATE TEMP TABLE m AS WITH u AS ({u}) SELECT * FROM gt ANTI JOIN u USING (lid, rid)")
con.execute(f"CREATE TEMP TABLE RA AS SELECT entity_id AS id, country_norm c, name_core, addr_norm FROM read_parquet(['{C}/norm_train_s2.parquet','{C}/norm_train_s3.parquet'])")
r = con.execute(f"""
  WITH sib AS (SELECT m.lid, m.rid, g.match_id AS rid2 FROM m JOIN '{C}/train_gt_pairs.parquet' g ON g.s1_id = m.lid AND g.match_id <> m.rid),
  sibf AS (SELECT s.* FROM sib s SEMI JOIN '{C}/cand_train.parquet' c ON c.lid = s.lid AND c.rid = s.rid2),
  j AS (SELECT s.lid, s.rid, max(jaro_winkler_similarity(a.name_core, b.name_core)) AS jw, bool_or(a.name_core = b.name_core) eq
        FROM sibf s JOIN RA a ON a.id = s.rid JOIN RA b ON b.id = s.rid2 GROUP BY ALL)
  SELECT (SELECT count(*) FROM m), (SELECT count(DISTINCT (lid, rid)) FROM sibf), sum(eq::INT), sum((jw >= 0.92)::INT) FROM j""").fetchone()
print("residual misses, with a found sibling right, sibling name identical, sibling name JW>=.92:", r)
# cost proxy: for residual rights with identical-name sibling, how many rights share that exact name_core in the country
r2 = con.execute(f"""
  WITH rdf AS (SELECT c, name_core, count(*) n FROM RA GROUP BY ALL)
  SELECT quantile_cont(n, [0.25, 0.5, 0.75, 0.9]), avg((n <= 20)::INT) FROM m JOIN RA a ON a.id = m.rid JOIN rdf USING (c, name_core)""").fetchone()
print("right-side df of residual right's exact name_core (quartiles, share<=20):", r2)
# actual transitive pass on the subset: r (subset) -> rights r' with the same (c, name_core), name R-df <= cap -> top-1 left of r' in cand_train
for cap in (5, 20):
    con.execute(f"""CREATE OR REPLACE TEMP TABLE tr AS
      WITH rdf AS (SELECT c, name_core, count(*) n FROM RA GROUP BY ALL HAVING count(*) <= {cap}),
      top1 AS (SELECT lid, rid FROM '{C}/cand_train.parquet' QUALIFY row_number() OVER (PARTITION BY rid ORDER BY block_score DESC) = 1)
      SELECT DISTINCT t.lid, r.id AS rid FROM R r JOIN rdf USING (c, name_core) JOIN RA b USING (c, name_core) JOIN top1 t ON t.rid = b.id
      WHERE b.id <> r.id""")
    n, gain_all, gain_res = con.execute(f"""SELECT (SELECT count(*) FROM tr),
       (SELECT count(*) FROM gt SEMI JOIN tr USING (lid, rid) WHERE NOT found),
       (SELECT count(*) FROM m SEMI JOIN tr USING (lid, rid))""").fetchone()
    ngt, nr = con.execute("SELECT count(*), (SELECT count(*) FROM R) FROM gt").fetchone()
    print(f"transitive exact-name cap {cap}: pairs/right {n/nr:.2f}, gain over v2 {100*gain_all/ngt:.2f}pp, gain over v2+new passes {100*gain_res/ngt:.3f}pp")
