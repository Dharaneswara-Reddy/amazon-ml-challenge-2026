"""Most uncertain French S1 entities under v4 (sum of p(1-p) over their pairs): print candidates, p, accepted flag."""
import duckdb
c = duckdb.connect(); c.execute("SET memory_limit='3GB'; SET threads=4")
P = "artifacts/runs/20260926T063247_1a0d20f_99304c0b_v4/pred2_test.parquet"
SUB = "submissions/06_20260926-1234/matching_results.tsv"
c.execute(f"""CREATE TABLE acc AS SELECT source1_entity_id lid, unnest(string_split(nullif(matched_entity_ids,''), ',')) rid
              FROM read_csv('{SUB}', delim='\t', header=true, all_varchar=true)""")
c.execute(f"""CREATE TABLE fp AS SELECT p.* FROM '{P}' p JOIN 'artifacts/cache/raw_test_s1.parquet' s ON s.entity_id = p.lid
              WHERE s.country = 'France'""")
print("France uncertain-pair share (0.05<p<0.95):", c.execute("SELECT avg((p>0.05 AND p<0.95)::INT) FROM fp").fetchone(),
      " India/US:", c.execute(f"""SELECT s.country, avg((p>0.05 AND p<0.95)::INT) FROM '{P}' p JOIN 'artifacts/cache/raw_test_s1.parquet' s
          ON s.entity_id=p.lid WHERE s.country<>'France' GROUP BY 1""").fetchall())
ents = [r[0] for r in c.execute("""SELECT lid FROM fp GROUP BY lid HAVING sum(p*(1-p)) > 0.4 USING SAMPLE 10 ROWS""").fetchall()]
R = "read_parquet(['artifacts/cache/raw_test_s2.parquet','artifacts/cache/raw_test_s3.parquet'])"
for e in ents:
    l = c.execute(f"SELECT business_name, business_address FROM 'artifacts/cache/raw_test_s1.parquet' WHERE entity_id='{e}'").fetchone()
    print(f"\nS1 {e}: {l[0]} | {l[1]}")
    for r in c.execute(f"""SELECT round(fp.p,3), (a.rid IS NOT NULL) acc, x.business_name, x.business_address,
            (SELECT count(*) FROM fp f2 WHERE f2.rid = fp.rid) n_s1_cands
          FROM fp JOIN {R} x ON x.entity_id = fp.rid LEFT JOIN acc a ON a.lid = fp.lid AND a.rid = fp.rid
          WHERE fp.lid = '{e}' AND fp.p > 0.02 ORDER BY fp.p DESC LIMIT 8""").fetchall():
        print(f"   p={r[0]:<6} {'ACC' if r[1] else '   '} cands={r[4]:<2} {r[2]} | {r[3]}")
