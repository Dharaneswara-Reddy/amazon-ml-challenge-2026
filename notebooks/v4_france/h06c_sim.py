"""S1 mutual similarity per country + recall cost of a higher France cut threshold (train analogue + France matches)."""
from h_common import C, TMP, RUN, connect

con = connect()
con.execute(f"""CREATE TEMP TABLE s AS SELECT n.entity_id, n.name_core, n.addr_street, n.addr_city, n.postal, r.country
  FROM read_parquet('{C}/norm_test_s1.parquet') n JOIN read_parquet('{C}/raw_test_s1.parquet') r USING (entity_id)""")
print("country, n, distinct name_core/n, share S1 whose name_core is shared, distinct street/n, distinct city, top10 city share, distinct postal")
print(con.execute("""WITH nm AS (SELECT country, name_core, count(*) k FROM s GROUP BY ALL),
  a AS (SELECT country, count(*) n, count(DISTINCT name_core) dn, count(DISTINCT addr_street) ds, count(DISTINCT addr_city) dc, count(DISTINCT postal) dp FROM s GROUP BY 1),
  b AS (SELECT country, sum(k) FILTER (WHERE k > 1) shared FROM nm GROUP BY 1),
  c AS (SELECT country, sum(k) top10 FROM (SELECT country, addr_city, count(*) k, row_number() OVER (PARTITION BY country ORDER BY count(*) DESC) rk FROM s GROUP BY country, addr_city) WHERE rk <= 10 GROUP BY 1)
  SELECT a.country, n, dn/n, shared/n, ds/n, dc, top10/n, dp FROM a JOIN b USING (country) JOIN c USING (country)""").fetchall())
print("top French cities:", con.execute("SELECT addr_city, count(*) FROM s WHERE country='France' GROUP BY 1 ORDER BY 2 DESC LIMIT 12").fetchall())
print("top French streets:", con.execute("SELECT addr_street, count(*) FROM s WHERE country='France' GROUP BY 1 ORDER BY 2 DESC LIMIT 12").fetchall())
for c in ("France", "US", "India"):
    n = con.execute(f"SELECT count(*) FROM s WHERE country='{c}'").fetchone()[0]
    print(c, "top name_core tokens (share of S1):", [(t, round(k / n, 3)) for t, k in con.execute(f"""SELECT t, count(*) FROM (
      SELECT DISTINCT entity_id, unnest(string_split(name_core,' ')) t FROM s WHERE country='{c}') GROUP BY 1 ORDER BY 2 DESC LIMIT 15""").fetchall()])
    print(c, "5 most repeated name_core:", con.execute(f"SELECT name_core, count(*) FROM s WHERE country='{c}' GROUP BY 1 ORDER BY 2 DESC LIMIT 5").fetchall())
# train: recall cost of a stricter cut on non-lrank-1 pairs
con.execute(f"""CREATE TEMP TABLE tr AS SELECT c.lid, c.rid, c.lrank, c.cut_p, s.country, (g.s1_id IS NOT NULL) y
  FROM read_parquet('{C}/cand3_train.parquet') c JOIN read_parquet('{C}/raw_train_s1.parquet') s ON s.entity_id = c.lid
  LEFT JOIN read_parquet('{C}/train_gt_pairs.parquet') g ON g.s1_id = c.lid AND g.match_id = c.rid""")
print("train: country, t, true pairs in cand, true lost (lrank>1 & cut_p<t), share lost, pairs dropped share")
print(con.execute("""SELECT country, t, count(*) FILTER (WHERE y), count(*) FILTER (WHERE y AND lrank>1 AND cut_p<t),
   count(*) FILTER (WHERE y AND lrank>1 AND cut_p<t)/count(*) FILTER (WHERE y), avg((lrank>1 AND cut_p<t)::INT)
   FROM tr, (VALUES (0.003),(0.01),(0.03),(0.1)) v(t) GROUP BY ALL ORDER BY ALL""").fetchall())
con.execute(f"""CREATE TEMP TABLE fr AS SELECT x.*, c.cut_p c_cut, c.lrank c_lrank FROM read_parquet('{TMP}/pairs_fr.parquet') x
  JOIN read_parquet('{C}/cand3_test.parquet') c USING (lid, rid)""")
print("France: t, currently accepted (rid-best p>=.69) pairs that would be dropped (lrank>1 & cut_p<t), of which D1, pairs dropped")
print(con.execute("""SELECT t, count(*) FILTER (WHERE r_rank=1 AND p>=0.69 AND c_lrank>1 AND c_cut<t),
   count(*) FILTER (WHERE r_rank=1 AND p>=0.69 AND c_lrank>1 AND c_cut<t AND sg_sole_off_in_s=1),
   count(*) FILTER (WHERE c_lrank>1 AND c_cut<t), count(*) FILTER (WHERE r_rank=1 AND p>=0.69)
   FROM fr, (VALUES (0.003),(0.01),(0.03),(0.1)) v(t) GROUP BY t ORDER BY t""").fetchall())
