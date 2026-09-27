"""Generator signatures: house-number edit type (true vs distractor), pseudo-brand names, GT structure, S2-S3 siblings."""
from pathlib import Path

import duckdb

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
OUT = ROOT / "notebooks/v3_research/out"
C = ROOT / "artifacts/cache"
con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=4")
con.execute(f"CREATE VIEW hp AS SELECT * FROM read_parquet('{OUT}/hold_pairs.parquet')")
COLS = "entity_id, name_core, addr_house, addr_nums, addr_norm, script"
con.execute(f"""CREATE TEMP TABLE rn AS SELECT {COLS} FROM read_parquet('{C}/norm_train_s2.parquet')
                UNION ALL SELECT {COLS} FROM read_parquet('{C}/norm_train_s3.parquet')""")
con.execute(f"CREATE TEMP TABLE ln AS SELECT {COLS} FROM read_parquet('{C}/norm_train_s1.parquet')")
con.execute("""CREATE TEMP TABLE pr AS
  SELECT hp.lid, hp.rid, hp.p, hp.label, hp.kept, hp.is_best, hp.sel,
    split_part(l.addr_nums,' ',1) AS a, split_part(r.addr_nums,' ',1) AS b,
    jaro_winkler_similarity(l.name_core, r.name_core) AS jw, l.name_core lc, r.name_core rc, r.script rsc,
    coalesce(r.addr_norm,'') = '' AS r_empty
  FROM hp JOIN ln l ON l.entity_id = hp.lid JOIN rn r ON r.entity_id = hp.rid""")
REL = """CASE WHEN a = '' OR b = '' THEN 'missing'
  WHEN a = b THEN 'equal'
  WHEN length(a) = length(b) + 1 AND starts_with(a, b) THEN 'R=L minus last digit'
  WHEN length(a) = length(b) + 1 AND (SELECT bool_or(a[1:i-1] || a[i+1:] = b) FROM range(1, length(a)+1) t(i)) THEN 'R=L minus inner digit'
  WHEN length(b) > length(a) AND starts_with(b, a) THEN 'R=L plus digits'
  WHEN try_cast(a AS BIGINT) IS NOT NULL AND try_cast(b AS BIGINT) IS NOT NULL AND abs(a::BIGINT - b::BIGINT) = 1 THEN '+-1'
  WHEN try_cast(a AS BIGINT) IS NOT NULL AND try_cast(b AS BIGINT) IS NOT NULL AND abs(a::BIGINT - b::BIGINT) <= 20 THEN '+-2..20'
  WHEN try_cast(a AS BIGINT) IS NOT NULL AND try_cast(b AS BIGINT) IS NOT NULL AND abs(a::BIGINT - b::BIGINT) <= 200 THEN '+-21..200'
  ELSE 'other' END"""
con.execute(f"CREATE TEMP TABLE pr2 AS SELECT *, {REL} AS rel FROM pr WHERE jw >= 0.9")
print("== first address number relation, candidate pairs with name JW>=0.9 (label=1 vs 0), holdout")
for r in con.execute("""SELECT rel, sum(label::INT) n_true, sum((NOT label)::INT) n_false,
        round(avg(label::INT),4) true_rate, sum((sel AND NOT label)::INT) fp, sum((label AND kept AND is_best AND NOT sel)::INT) dec_fn
        FROM pr2 GROUP BY 1 ORDER BY 2 DESC""").fetchall():
    print(r)
print("== signed diff (b - a) distribution for numeric, small |diff|<=20, by label")
for r in con.execute("""SELECT label, b::BIGINT - a::BIGINT d, count(*) FROM pr2
        WHERE rel IN ('+-1','+-2..20') GROUP BY 1,2 ORDER BY 1,2""").fetchall():
    print(r)

print("== pseudo-brand names: R name_core single token absent from S1 name vocabulary")
con.execute("CREATE TEMP TABLE vocab AS SELECT DISTINCT unnest(string_split(name_core,' ')) t FROM ln")
con.execute("""CREATE TEMP TABLE pb AS SELECT pr.*, (rc NOT LIKE '% %' AND rc <> '' AND rsc = 'latin'
        AND rc NOT IN (SELECT t FROM vocab) AND replace(lc,' ','') <> rc AND jw < 0.75) AS pseudo FROM pr""")
print(con.execute("""SELECT pseudo, count(*), sum(label::INT), round(avg(label::INT),4), sum((sel AND NOT label)::INT) fp,
        sum((label AND kept AND is_best AND NOT sel)::INT) dec_fn, count(DISTINCT rid) FROM pb GROUP BY 1""").fetchall())
print("sample pseudo names:", [x[0] for x in con.execute("SELECT rc FROM pb WHERE pseudo AND label USING SAMPLE 25").fetchall()])
print("sample pseudo names on non-label:", [x[0] for x in con.execute("SELECT rc FROM pb WHERE pseudo AND NOT label USING SAMPLE 15").fetchall()])

print("== GT structure: per S1 matches by source (train)")
for r in con.execute(f"""SELECT n2, n3, count(*) FROM (SELECT s1_id, sum((match_id LIKE 'S2%')::INT) n2,
        sum((match_id LIKE 'S3%')::INT) n3 FROM read_parquet('{C}/train_gt_pairs.parquet') GROUP BY 1)
        GROUP BY 1,2 ORDER BY 3 DESC LIMIT 20""").fetchall():
    print(r)

print("== S2-S3 siblings: name / address similarity R-R vs R-S1 (holdout S1 with >=1 S2 and >=1 S3 match)")
con.execute(f"""CREATE TEMP TABLE g AS SELECT gp.s1_id, gp.match_id FROM read_parquet('{C}/train_gt_pairs.parquet') gp
                WHERE gp.s1_id IN (SELECT DISTINCT lid FROM hp USING SAMPLE 20000)""")
print(con.execute("""SELECT round(avg(jaro_winkler_similarity(a.name_core, b.name_core)),4) rr_name,
        round(avg(jaro_winkler_similarity(l.name_core, a.name_core)),4) lr_name,
        round(avg(jaro_winkler_similarity(a.addr_norm, b.addr_norm)),4) rr_addr,
        round(avg(jaro_winkler_similarity(l.addr_norm, a.addr_norm)),4) lr_addr,
        round(avg((a.name_core = b.name_core)::INT),4) rr_name_eq, round(avg((l.name_core = a.name_core)::INT),4) lr_name_eq,
        count(*)
      FROM g x JOIN g y ON x.s1_id = y.s1_id AND x.match_id < y.match_id
      JOIN rn a ON a.entity_id = x.match_id JOIN rn b ON b.entity_id = y.match_id JOIN ln l ON l.entity_id = x.s1_id""").fetchall())
