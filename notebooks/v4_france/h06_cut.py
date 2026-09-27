"""Cut diagnosis: why France keeps 3.7 candidates per right. Per-country cut-feature distributions on test candidates,
kept-by-cut pairs (lrank > 1), and S1 mutual similarity (shared names, streets, city concentration).
"""
from h_common import C, TMP, connect, norm_right

con = connect()
con.execute(f"""CREATE TEMP TABLE cc AS SELECT c.*, s.country FROM read_parquet('{C}/cand3_test.parquet') c
  JOIN read_parquet('{C}/raw_test_s1.parquet') s ON s.entity_id = c.lid""")
print("per country: pairs, rights, per right, share lrank=1, share cut_p>=1e-3, >=1e-2, >=0.1")
print(con.execute("""SELECT country, count(*), count(DISTINCT rid), count(*)/count(DISTINCT rid),
  avg((lrank=1)::INT), avg((cut_p>=1e-3)::INT), avg((cut_p>=1e-2)::INT), avg((cut_p>=0.1)::INT) FROM cc GROUP BY 1""").fetchall())
print("rights by #candidates (1,2,3,4-5,6-10,>10) share")
print(con.execute("""WITH r AS (SELECT country, rid, count(*) n FROM cc GROUP BY ALL)
  SELECT country, avg((n=1)::INT), avg((n=2)::INT), avg((n=3)::INT), avg((n BETWEEN 4 AND 5)::INT), avg((n BETWEEN 6 AND 10)::INT), avg((n>10)::INT) FROM r GROUP BY 1""").fetchall())
feats = ["cut_p", "jw_core", "jw_addr", "jacc_addr", "jw_strip", "jw_phn", "tok_score", "q3_score", "s_px", "s_hs",
         "s_n2", "s_na", "s_phx", "n_pass", "n_pass_new", "in_v2", "num_match", "block_score"]
for grp, w in (("non-best kept (lrank>1)", "lrank > 1"), ("lrank=1", "lrank = 1")):
    print("==", grp, "mean / median per country")
    sel = ", ".join(f"round(avg({f}),3), round(median({f}),3)" for f in feats)
    for r in con.execute(f"SELECT country, count(*), {sel} FROM cc WHERE {w} GROUP BY 1 ORDER BY 1").fetchall():
        print(r[0], r[1], {f: (r[2 + 2 * i], r[3 + 2 * i]) for i, f in enumerate(feats)})
# pass membership shares among non-best kept pairs
print("pass membership among lrank>1 kept:")
print(con.execute("""SELECT country, avg(in_v2), avg(in_px), avg(in_hs), avg(in_n2), avg(in_na), avg(in_phx), avg(in_tok), avg(in_q3),
   avg(in_sn_name), avg(in_nameonly) FROM cc WHERE lrank>1 GROUP BY 1""").fetchall())
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
