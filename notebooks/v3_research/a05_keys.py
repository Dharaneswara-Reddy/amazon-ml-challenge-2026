"""Evaluate candidate new blocking keys: coverage of holdout blocking misses and pair volume on full train."""
from pathlib import Path

import duckdb

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
OUT = ROOT / "notebooks/v3_research/out"
C = ROOT / "artifacts/cache"
con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=4"); con.execute("SET temp_directory='/home/toji339/Documents/Amazon_ML_Hackathon/artifacts/tmp/v3a'")
COLS = "entity_id, name_core, name_skel, addr_nums, addr_house, addr_street, addr_city, addr_admin, country_norm"
KEYS = {
    "nosp_name": "nullif(replace(name_core,' ',''),'')",
    "nosp_name_admin": "CASE WHEN name_core<>'' THEN replace(name_core,' ','') || '|' || addr_admin END",
    "house_street": "CASE WHEN addr_house<>'' AND addr_street<>'' THEN split_part(addr_house,' ',1) || '|' || addr_street END",
    "house_admin_city": "CASE WHEN addr_house<>'' AND addr_city<>'' THEN split_part(addr_house,' ',1) || '|' || addr_city || '|' || addr_admin END",
    "street_admin": "CASE WHEN addr_street<>'' THEN addr_street || '|' || addr_admin END",
    "skel_admin": "CASE WHEN name_skel<>'' THEN name_skel || '|' || addr_admin END",
    "tok1_2_admin": "CASE WHEN name_core LIKE '% %' THEN split_part(name_core,' ',1)||' '||split_part(name_core,' ',2)||'|'||addr_admin END",
    "nums_admin": "CASE WHEN addr_nums<>'' THEN array_to_string(list_sort(string_split(addr_nums,' ')),' ') || '|' || addr_admin END",
}
kexpr = ", ".join(f"{v} AS {k}" for k, v in KEYS.items())
con.execute(f"CREATE TEMP TABLE L AS SELECT entity_id, {kexpr} FROM (SELECT {COLS} FROM read_parquet('{C}/norm_train_s1.parquet'))")
con.execute(f"""CREATE TEMP TABLE R AS SELECT entity_id, {kexpr} FROM (
  SELECT {COLS} FROM read_parquet('{C}/norm_train_s2.parquet') UNION ALL SELECT {COLS} FROM read_parquet('{C}/norm_train_s3.parquet'))""")
nR = con.execute("SELECT count(*) FROM R").fetchone()[0]
con.execute(f"CREATE TEMP TABLE ts AS SELECT s1_id, match_id, status FROM read_parquet('{OUT}/tp_status.parquet')")
cov_any_miss = []
import os
for k in (KEYS if os.environ.get('PART1') else []):
    con.execute(f"CREATE OR REPLACE TEMP TABLE lc AS SELECT {k} AS kv, count(*) nl FROM L WHERE {k} IS NOT NULL GROUP BY 1")
    con.execute(f"CREATE OR REPLACE TEMP TABLE rc AS SELECT {k} AS kv, count(*) nr FROM R WHERE {k} IS NOT NULL GROUP BY 1")
    for cap in (20, 100):
        vol = con.execute(f"SELECT sum(nl*nr) FROM lc JOIN rc USING (kv) WHERE nl <= {cap}").fetchone()[0] or 0
        r = con.execute(f"""SELECT count(*), sum((ts.status='block_miss')::INT),
              sum((ts.status='block_miss' AND l.{k} = r.{k})::INT), sum((l.{k} = r.{k})::INT)
            FROM ts JOIN L l ON l.entity_id = ts.s1_id JOIN R r ON r.entity_id = ts.match_id
            JOIN lc ON lc.kv = l.{k} AND lc.nl <= {cap}""").fetchone()
        tot = con.execute("SELECT count(*), sum((status='block_miss')::INT) FROM ts").fetchone()
        print(f"{k:18s} cap{cap:4d} pairs/R={vol / nR:6.2f}  covers all-true {r[3] / tot[0]:.3f}  "
              f"covers misses {r[2]}/{tot[1]} = {r[2] / tot[1]:.3f}", flush=True)

# ---- union coverage of misses, and would the current linear ranker keep the pair in the top-3?
import json  # noqa: E402
w = json.loads((C / "learned/ranker.json").read_text())["weights"]
cond = " OR ".join(f"(l.{k} = r.{k} AND l.{k} IN (SELECT kv FROM lc_{k}))" for k in KEYS)
for k in KEYS:
    con.execute(f"CREATE OR REPLACE TEMP TABLE lc_{k} AS SELECT {k} AS kv FROM L WHERE {k} IS NOT NULL GROUP BY 1 HAVING count(*) <= 20")
con.execute(f"""CREATE OR REPLACE TEMP TABLE mm AS SELECT ts.s1_id, ts.match_id, ({cond}) AS covered
  FROM ts JOIN L l ON l.entity_id = ts.s1_id JOIN R r ON r.entity_id = ts.match_id WHERE ts.status = 'block_miss'""")
print("union of all keys (cap 20) covers misses:", con.execute("SELECT avg(covered::INT), sum(covered::INT) FROM mm").fetchall())
AW = ("array_to_string(list_sort(list_filter(string_split({c}, ' '), x -> length(x) >= 2 AND NOT "
      "regexp_matches(x, '^[0-9]+$'))), ' ')")
con.execute(f"""CREATE OR REPLACE TEMP TABLE sc AS
  SELECT mm.*, {w['bias']!r} + {w['jw_core']!r} * jaro_winkler_similarity(l.name_core, r.name_core)
    + {w['jw_addr']!r} * jaro_winkler_similarity({AW.format(c='l.addr_norm')}, {AW.format(c='r.addr_norm')})
    + {w['jacc_addr']!r} * (len(list_intersect(string_split(l.addr_norm,' '), string_split(r.addr_norm,' ')))::DOUBLE
         / greatest(len(list_distinct(list_concat(string_split(l.addr_norm,' '), string_split(r.addr_norm,' ')))),1))
    + {w['num_match']!r} * (len(list_intersect(string_split(l.addr_nums,' '), string_split(r.addr_nums,' '))) > 0)::INT
    + {w['n_pass']!r} * 1 AS bs
  FROM mm JOIN read_parquet('{C}/norm_train_s1.parquet') l ON l.entity_id = mm.s1_id
  JOIN (SELECT * FROM read_parquet('{C}/norm_train_s2.parquet') UNION ALL SELECT * FROM read_parquet('{C}/norm_train_s3.parquet')) r
    ON r.entity_id = mm.match_id""")
con.execute(f"""CREATE OR REPLACE TEMP TABLE th AS SELECT rid, min(block_score) min3, max(block_score) max3
  FROM read_parquet('{C}/cand_train.parquet') WHERE rid IN (SELECT match_id FROM mm) GROUP BY 1""")
print("misses whose own ranker score beats the rid's current 3rd candidate (n_pass=1 assumed):",
      con.execute("""SELECT avg((bs > min3)::INT), sum((bs > min3)::INT), avg((bs > max3)::INT),
                     sum((covered AND bs > min3)::INT) FROM sc JOIN th ON th.rid = sc.match_id""").fetchall())
