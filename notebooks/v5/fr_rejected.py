"""Why are clean near-identical best pairs rejected in France? For each such pair: is its right record matched to
another S1 (sibling confusion) or to none (low p)? Plus examples with the competing S1."""
import duckdb
c = duckdb.connect(); c.execute("SET memory_limit='5GB'; SET threads=8")
F, SUB = "artifacts/cache/feat_v4_test.parquet", "submissions/06_20260926-1234/matching_results.tsv"
P = "artifacts/runs/20260926T063247_1a0d20f_99304c0b_v4/pred2_test.parquet"
c.execute(f"""CREATE TABLE acc AS SELECT source1_entity_id lid, unnest(string_split(nullif(matched_entity_ids,''), ',')) rid
              FROM read_csv('{SUB}', delim='\t', header=true, all_varchar=true)""")
c.execute(f"""CREATE TABLE rej AS SELECT f.lid, f.rid, s.country, p.p FROM '{F}' f JOIN 'artifacts/cache/raw_test_s1.parquet' s ON s.entity_id = f.lid
    JOIN '{P}' p USING (lid, rid)
    WHERE f.sg_num_changed = 0 AND f.sg_num_l_only = 0 AND f.sg_num_r_only = 0 AND f.sg_num_eq >= 1 AND f.core_ratio >= 0.95 AND f.r_rank = 1
      AND NOT EXISTS (SELECT 1 FROM acc a WHERE a.lid = f.lid AND a.rid = f.rid)""")
print(c.execute("""SELECT country, count(*), count(a.rid) AS right_matched_elsewhere, round(avg(p),3), round(quantile_cont(p,0.5),3)
    FROM rej LEFT JOIN (SELECT DISTINCT rid FROM acc) a USING (rid) GROUP BY 1 ORDER BY 1""").fetchall())
print("p buckets (France):", c.execute("""SELECT CASE WHEN p<0.1 THEN '<.1' WHEN p<0.3 THEN '.1-.3' WHEN p<0.5 THEN '.3-.5' ELSE '>=.5' END b, count(*)
    FROM rej WHERE country='France' GROUP BY 1 ORDER BY 1""").fetchall())
ex = c.execute(f"""SELECT l.business_name, l.business_address, r.business_name, r.business_address, round(x.p,3),
      o.business_name, o.business_address
    FROM (SELECT * FROM rej WHERE country='France' USING SAMPLE 12 ROWS) x
    JOIN 'artifacts/cache/raw_test_s1.parquet' l ON l.entity_id = x.lid
    JOIN read_parquet(['artifacts/cache/raw_test_s2.parquet','artifacts/cache/raw_test_s3.parquet']) r ON r.entity_id = x.rid
    LEFT JOIN acc a ON a.rid = x.rid LEFT JOIN 'artifacts/cache/raw_test_s1.parquet' o ON o.entity_id = a.lid""").fetchall()
for e in ex:
    print(f"  S1: {e[0]} | {e[1]}\n  R : {e[2]} | {e[3]}  (p={e[4]})\n  -> matched instead to: {e[5]} | {e[6]}\n")
