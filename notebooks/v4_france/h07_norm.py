"""Print raw vs normalized French records (S1 + right copies) to check legal forms, elision, accents, street types."""
from h_common import C, connect, norm_right, raw_right

con = connect()
q = f"""SELECT r.business_name, n.name_norm, n.name_core, n.legal_form, n.type_words, r.business_address, n.addr_norm,
  n.addr_nums, n.addr_house, n.addr_street, n.addr_city, n.postal
  FROM read_parquet('{C}/raw_test_s1.parquet') r JOIN read_parquet('{C}/norm_test_s1.parquet') n USING (entity_id)
  WHERE r.country='France' AND ({{w}}) ORDER BY hash(entity_id) LIMIT {{k}}"""
for w, k in (("true", 12), ("regexp_matches(lower(business_name), '\\b(sarl|sas|sasu|eurl|sci|snc|ei|sa)\\b')", 8),
             ("regexp_matches(business_name, '[''’]')", 6), ("regexp_matches(lower(business_address), '\\b(bis|ter)\\b')", 5),
             ("regexp_matches(lower(business_address), '\\b(bd|av|all[ée]e|impasse|chemin|place|quai|cours)\\b')", 6)):
    print("=====", w)
    for row in con.execute(q.format(w=w, k=k)).fetchall():
        print(" | ".join(str(x) for x in row))
print("===== legal_form values in France S1:", con.execute(f"""SELECT n.legal_form, count(*) FROM read_parquet('{C}/raw_test_s1.parquet') r
  JOIN read_parquet('{C}/norm_test_s1.parquet') n USING (entity_id) WHERE r.country='France' GROUP BY 1 ORDER BY 2 DESC LIMIT 15""").fetchall())
print("===== raw legal tokens present in France S1 names:", con.execute(f"""SELECT t, count(*) FROM (SELECT unnest(regexp_extract_all(lower(business_name), '\\b(sarl|sas|sasu|eurl|sci|snc|ei|sa|scop|selarl|gie|scm|association)\\b')) t
  FROM read_parquet('{C}/raw_test_s1.parquet') WHERE country='France') GROUP BY 1 ORDER BY 2 DESC""").fetchall())
print("===== France right records sample:")
for row in con.execute(f"""SELECT r.business_name, n.name_norm, n.name_core, n.legal_form, r.business_address, n.addr_norm, n.addr_nums
  FROM {raw_right()} r JOIN {norm_right()} n USING (entity_id) WHERE r.country='France' ORDER BY hash(entity_id) LIMIT 12""").fetchall():
    print(" | ".join(str(x) for x in row))
