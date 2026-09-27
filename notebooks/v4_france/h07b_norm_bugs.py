"""Quantify the French normalization issues seen in h07: 'ter' -> terrace, elision glued before accented letters,
region/department words leaking into addr_city, admin empty; city_eq / admin_eq on pseudo-positive pairs vs train."""
from h_common import C, TMP, connect, norm_right, raw_right

con = connect()
con.execute(f"""CREATE TEMP TABLE fx AS SELECT r.entity_id, r.business_name, r.business_address, n.* EXCLUDE (entity_id)
  FROM (SELECT * FROM read_parquet('{C}/raw_test_s1.parquet') UNION ALL SELECT * FROM {raw_right()}) r
  JOIN (SELECT * FROM read_parquet('{C}/norm_test_s1.parquet') UNION ALL SELECT * FROM {norm_right()}) n USING (entity_id)
  WHERE r.country = 'France'""")
n = con.execute("SELECT count(*) FROM fx").fetchone()[0]
print("France records (S1+S2+S3):", n)
print("raw ' ter ' after a number:", con.execute(r"SELECT count(*) FROM fx WHERE regexp_matches(lower(business_address), '\d+\s*ter\b')").fetchone()[0],
      "-> normalized contains 'terrace':", con.execute("SELECT count(*) FROM fx WHERE addr_norm LIKE '%terrace%'").fetchone()[0],
      "raw bis:", con.execute(r"SELECT count(*) FROM fx WHERE regexp_matches(lower(business_address), '\d+\s*bis\b')").fetchone()[0])
print("elision before accented capital/letter (raw l'/d' + accented):", con.execute(r"""SELECT count(*) FROM fx WHERE regexp_matches(business_address || ' ' || business_name, '(?i)\b[ld][''’][ÉÈÊÀÂÎÔéèêàâîôÉ]')""").fetchone()[0])
print("examples:", con.execute(r"""SELECT business_address, addr_norm FROM fx WHERE regexp_matches(business_address, '(?i)\b[ld][''’][ÉÈÊÀÂÎÔéèêàâîô]') LIMIT 4""").fetchall())
print("elision before plain letter examples:", con.execute(r"""SELECT business_name, name_norm, business_address, addr_norm FROM fx WHERE regexp_matches(business_address || business_name, '(?i)\b[ld][''’][a-z]') LIMIT 4""").fetchall())
print("curly apostrophe ’ records:", con.execute("SELECT count(*) FROM fx WHERE business_address LIKE '%’%' OR business_name LIKE '%’%'").fetchone()[0])
print("addr_admin non-empty share:", con.execute("SELECT avg((coalesce(addr_admin,'')<>'')::INT), avg((coalesce(postal,'')<>'')::INT) FROM fx").fetchone())
print("top addr_admin:", con.execute("SELECT addr_admin, count(*) FROM fx GROUP BY 1 ORDER BY 2 DESC LIMIT 10").fetchall())
print("top addr_city (all France records):", con.execute("SELECT addr_city, count(*) FROM fx GROUP BY 1 ORDER BY 2 DESC LIMIT 25").fetchall())
print("legal_form combos with US forms:", con.execute("SELECT legal_form, count(*) FROM fx WHERE legal_form LIKE '% %' OR legal_form IN ('co','lp','pa','inc','llc','ltd') GROUP BY 1 ORDER BY 2 DESC LIMIT 10").fetchall())
print("examples:", con.execute("SELECT business_name, name_norm, legal_form FROM fx WHERE legal_form IN ('co sarl','lp sarl','pa sarl','co') LIMIT 6").fetchall())
# feature view: city_eq / admin_eq / postal_state on positives (France pseudo T3, train label 1)
print("France T3 pairs: city_eq, admin_eq, postal_state means:", con.execute(f"""SELECT avg(f.city_eq), avg(f.admin_eq), avg(f.postal_state), avg(f.street_jw), avg(f.addr_ratio)
  FROM read_parquet('{TMP}/fr_pseudo.parquet') p JOIN read_parquet('{C}/feat_v3_test.parquet') f USING (lid, rid) WHERE p.rule='T3'""").fetchone())
print("train holdout positives: city_eq, admin_eq, postal_state, street_jw, addr_ratio:", con.execute(f"""SELECT s.country, avg(f.city_eq), avg(f.admin_eq), avg(f.postal_state), avg(f.street_jw), avg(f.addr_ratio)
  FROM read_parquet('{C}/feat_v3_train_holdout.parquet') f JOIN read_parquet('{C}/raw_train_s1.parquet') s ON s.entity_id=f.lid WHERE f.label GROUP BY 1""").fetchall())
