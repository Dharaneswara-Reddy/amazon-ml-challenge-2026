"""Cut diagnosis: why France keeps 3.7 candidates per right. Per-country cut-feature distributions on test candidates,
kept-by-cut pairs (lrank > 1), and S1 mutual similarity (shared names, streets, city concentration).
"""
from h_common import C, TMP, connect, norm_right

con = connect()
con.execute(f"""CREATE TEMP TABLE cc AS SELECT c.*, s.country FROM read_parquet('{C}/cand3_test.parquet') c
  JOIN read_parquet('{C}/raw_test_s1.parquet') s ON s.entity_id = c.lid""")
print("pass membership among lrank>1 kept:")
print(con.execute("""SELECT country, avg(in_v2::INT), avg(in_px::INT), avg(in_hs::INT), avg(in_n2::INT), avg(in_na::INT), avg(in_phx::INT), avg(in_tok::INT), avg(in_q3::INT),
   avg(in_sn_name::INT), avg(in_nameonly::INT) FROM cc WHERE lrank>1 GROUP BY 1""").fetchall())
# pseudo-label view of French kept pairs: which are E (easy neg) and how many would survive a higher cut threshold
con.execute(f"CREATE TEMP TABLE ps AS SELECT * FROM read_parquet('{TMP}/fr_pseudo.parquet')")
print("France kept pairs by pseudo label and cut_p band (lrank>1):")
print(con.execute("""SELECT coalesce(ps.rule,'none'), count(*), avg((cc.cut_p>=1e-2)::INT), avg((cc.cut_p>=3e-2)::INT), avg((cc.cut_p>=0.1)::INT)
   FROM cc LEFT JOIN ps USING (lid, rid) WHERE cc.country='France' AND cc.lrank>1 GROUP BY 1 ORDER BY 1""").fetchall())
print("France positives (T3/T4) lost at cut threshold t (they are kept at 1e-3 or lrank=1):")
print(con.execute("""SELECT t, count(*) FILTER (WHERE ps.y=1 AND cc.lrank>1 AND cc.cut_p < t) lost_pos,
   count(*) FILTER (WHERE ps.y=1) pos, count(*) FILTER (WHERE cc.lrank>1 AND cc.cut_p < t) dropped_pairs
   FROM cc JOIN ps USING (lid, rid), (VALUES (0.003),(0.01),(0.03),(0.1)) v(t) WHERE cc.country='France' GROUP BY t ORDER BY t""").fetchall())
# mutual similarity of S1 records per country
print("S1 mutual similarity: n, distinct name_core share, share with shared name_core, distinct street share, top-10 city share, S1 per city (median)")
print(con.execute(f"""WITH s AS (SELECT n.*, r.country FROM read_parquet('{C}/norm_test_s1.parquet') n JOIN read_parquet('{C}/raw_test_s1.parquet') r USING (entity_id)),
  nm AS (SELECT country, name_core, count(*) k FROM s GROUP BY ALL),
  ct AS (SELECT country, addr_city, count(*) k, row_number() OVER (PARTITION BY country ORDER BY count(*) DESC) rk FROM s GROUP BY ALL)
  SELECT s.country, count(*), (SELECT count(*) FROM nm WHERE nm.country=s.country)/count(*),
    (SELECT sum(k) FILTER (WHERE k>1) FROM nm WHERE nm.country=s.country)/count(*),
    count(DISTINCT addr_street)/count(*),
    (SELECT sum(k) FILTER (WHERE rk<=10) FROM ct WHERE ct.country=s.country)/count(*),
    (SELECT count(*) FROM ct WHERE ct.country=s.country)
  FROM s GROUP BY 1""").fetchall())
print("top French cities / streets / name tokens:")
print(con.execute(f"""SELECT addr_city, count(*) FROM read_parquet('{C}/norm_test_s1.parquet') n JOIN read_parquet('{C}/raw_test_s1.parquet') r USING (entity_id)
  WHERE country='France' GROUP BY 1 ORDER BY 2 DESC LIMIT 15""").fetchall())
print(con.execute(f"""SELECT addr_street, count(*) FROM read_parquet('{C}/norm_test_s1.parquet') n JOIN read_parquet('{C}/raw_test_s1.parquet') r USING (entity_id)
  WHERE country='France' GROUP BY 1 ORDER BY 2 DESC LIMIT 15""").fetchall())
for c in ("France", "US", "India"):
    print(c, "top name_core tokens (share of S1):", con.execute(f"""SELECT t, count(*)/(SELECT count(*) FROM read_parquet('{C}/raw_test_s1.parquet') WHERE country='{c}') FROM (
      SELECT DISTINCT entity_id, unnest(string_split(name_core,' ')) t FROM read_parquet('{C}/norm_test_s1.parquet') n JOIN read_parquet('{C}/raw_test_s1.parquet') r USING (entity_id) WHERE country='{c}')
      GROUP BY 1 ORDER BY 2 DESC LIMIT 15""").fetchall())
