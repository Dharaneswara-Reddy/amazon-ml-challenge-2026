"""C3: in France, how many #6-ACCEPTED pairs fall in the chance-dominated cells of C1/C2 (same core name, same house
number, street not the same)? accepted vs estimated true per cell -> lower bound on FPs. Same for US test (reference)."""
import json
from common import *
c = connect()
c2 = json.load(open(OUT / "c2_autocorr.json")); c1 = json.load(open(OUT / "c1_collision.json"))
c.execute(f"""CREATE TABLE acc AS SELECT source1_entity_id lid, unnest(string_split(nullif(coalesce(matched_entity_ids,''),''), ',')) rid
              FROM read_csv('{SUB6}', delim='\t', header=true, all_varchar=true)""")
for name, cn, lw in (("France", "france", "true"), ("US_test", "us", "hash(entity_id) % 4 = 0")):
    c.execute(f"""CREATE OR REPLACE TABLE L AS SELECT entity_id id, name_core, try_cast(addr_house AS INT) h, addr_street st, addr_city ci
        FROM '{CACHE}/norm_test_s1.parquet' WHERE country_norm = '{cn}' AND {lw} AND try_cast(addr_house AS INT) IS NOT NULL AND name_core <> ''""")
    c.execute(f"""CREATE OR REPLACE TABLE R AS SELECT entity_id id, name_core, try_cast(addr_house AS INT) h, addr_street st, addr_city ci
        FROM read_parquet(['{CACHE}/norm_test_s2.parquet', '{CACHE}/norm_test_s3.parquet']) WHERE country_norm = '{cn}'
        AND try_cast(addr_house AS INT) IS NOT NULL AND name_core <> ''""")
    c.execute(f"""CREATE OR REPLACE TABLE P AS SELECT l.id lid, r.id rid, r.h - l.h d,
          CASE WHEN jaro_winkler_similarity(l.st, r.st) >= 0.9 THEN 'same' WHEN jaro_winkler_similarity(l.st, r.st) >= 0.6 THEN 'mid' ELSE 'diff' END sb,
          (l.ci = r.ci)::INT ce
        FROM L l JOIN R r ON r.name_core = l.name_core AND r.h = l.h""")
    rows = c.execute(f"""SELECT sb, ce, count(*), count(a.rid), count(p.p), count(*) FILTER (WHERE p.p >= 0.5)
        FROM P LEFT JOIN acc a USING (lid, rid) LEFT JOIN '{RUN}/pred2_test.parquet' p USING (lid, rid) GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()
    ratio = c2[name]["ratio"]
    base = {(r["sb"], r["ce"]): r["base"] * ratio for r in c1[name]}
    print(f"\n== {name}: cell, n(d=0), in candidates, accepted #6, p2>=.5, est. true, accepted - est.true (FP lower bound)")
    for sb, ce, n, acc, cand, hi in rows:
        est = n - base[(sb, ce)]
        print(f"  {sb:5s} city_eq={ce}  n={n:8,d}  cand={cand:8,d}  accepted={acc:8,d}  p>=.5={hi:8,d}  est_true={est:9.0f}  excess={acc - est:8.0f}")
