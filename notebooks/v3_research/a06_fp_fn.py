"""Decision-level errors: calibration, k vs n, slices, FP / decision-FN examples with pattern tags."""
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
OUT = ROOT / "notebooks/v3_research/out"
C = ROOT / "artifacts/cache"
con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=4")
con.execute(f"CREATE VIEW hp AS SELECT * FROM read_parquet('{OUT}/hold_pairs.parquet')")
con.execute(f"CREATE VIEW ec AS SELECT * FROM read_parquet('{OUT}/hold_ent_counts.parquet')")

print("== calibration of pcal on selectable (kept & is_best) pairs")
for r in con.execute("""SELECT floor(pcal*10)/10 b, count(*), round(avg(pcal),4), round(avg(label::INT),4)
                        FROM hp WHERE kept AND is_best GROUP BY 1 ORDER BY 1""").fetchall():
    print(r)
print("sum pcal vs sum label (selectable):", con.execute("SELECT sum(pcal), sum(label::INT) FROM hp WHERE kept AND is_best").fetchall())
print("selected: n, mean pcal, precision:", con.execute("SELECT count(*), avg(pcal), avg(label::INT) FROM hp WHERE sel").fetchall())
print("unselected selectable: n, mean pcal, label rate:", con.execute(
    "SELECT count(*), avg(pcal), avg(label::INT), sum(label::INT) FROM hp WHERE kept AND is_best AND NOT sel").fetchall())
print("== k_star vs available true (label among selectable)")
con.execute("""CREATE TEMP TABLE kv AS SELECT ec.idx, ec.n_true, ec.k_star, ec.country, ec.f,
   coalesce(a.avail,0) avail, coalesce(a.nsel,0) nsel FROM ec LEFT JOIN
   (SELECT idx, sum((is_best AND label AND kept)::INT) avail, sum(sel::INT) nsel FROM hp GROUP BY 1) a USING (idx)""")
print(con.execute("""SELECT sum((k_star < avail)::INT) n_under, sum((k_star > avail)::INT) n_over, sum((k_star = avail)::INT) n_eq,
                     sum(k_star) sk, sum(avail) sa, sum(n_true) sn FROM kv""").fetchall())
print("== loss by country")
for r in con.execute("""SELECT country, count(*), round(avg(f),5), round(sum(1-f)/(SELECT count(*) FROM ec),5) loss_pts,
          sum(fn_block), sum(fn_dec), sum(fn_o2o), sum(fp), round(sum(fn_block)/sum(n_true),4) miss_rate
          FROM ec GROUP BY 1""").fetchall():
    print(r)

# examples
con.execute(f"""CREATE TEMP TABLE r AS SELECT * FROM read_parquet('{C}/raw_train_s2.parquet') UNION ALL
                SELECT * FROM read_parquet('{C}/raw_train_s3.parquet')""")
con.execute(f"""CREATE TEMP TABLE rn AS SELECT entity_id, name_core, addr_norm, addr_nums, script FROM read_parquet('{C}/norm_train_s2.parquet')
                UNION ALL SELECT entity_id, name_core, addr_norm, addr_nums, script FROM read_parquet('{C}/norm_train_s3.parquet')""")
TAG = """CASE WHEN rn.script <> 'latin' THEN 'nonlatin'
   WHEN coalesce(rn.addr_norm,'') = '' THEN 'r_addr_empty'
   WHEN ln.name_core = rn.name_core AND ln.addr_nums <> rn.addr_nums THEN 'name_eq,nums_differ'
   WHEN ln.name_core = rn.name_core THEN 'name_eq,nums_eq'
   WHEN len(list_distinct(list_concat(string_split(ln.name_core,' '), string_split(rn.name_core,' '))))
        - len(list_intersect(string_split(ln.name_core,' '), string_split(rn.name_core,' '))) = 1 THEN 'name_1word_added/dropped'
   WHEN jaro_winkler_similarity(ln.name_core, rn.name_core) < 0.6 THEN 'name_very_different'
   ELSE 'name_other_noise' END"""
for what, cond in (("FP", "sel AND NOT label"), ("DEC_FN", "kept AND is_best AND label AND NOT sel")):
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ex AS
      SELECT hp.lid, hp.rid, hp.p, hp.pcal, ec.n_true, ec.country, {TAG} AS tag,
        l.business_name ln_, l.business_address la, r.business_name rn_, r.business_address ra,
        (SELECT count(*) FROM read_parquet('{C}/train_gt_pairs.parquet') g WHERE g.match_id = hp.rid) AS rid_matched
      FROM hp JOIN ec USING (idx)
      JOIN read_parquet('{C}/raw_train_s1.parquet') l ON l.entity_id = hp.lid
      JOIN read_parquet('{C}/norm_train_s1.parquet') ln ON ln.entity_id = hp.lid
      JOIN r ON r.entity_id = hp.rid JOIN rn ON rn.entity_id = hp.rid
      WHERE {cond}""")
    print(f"== {what} tags (n, singleton-entity count, mean pcal)")
    for x in con.execute("""SELECT tag, count(*), sum((n_true=0)::INT), round(avg(pcal),3), country, count(*) FILTER (WHERE rid LIKE 'S3%')
                            FROM ex GROUP BY tag, country ORDER BY tag, country""").fetchall():
        print(x)
    rows = con.execute("SELECT * FROM ex USING SAMPLE 36 ROWS (reservoir, 7)").fetchall()
    for x in rows:
        print(f"--- {x[5]} n_true={x[4]} p={x[2]:.3f} pcal={x[3]:.3f} tag={x[6]} rid_is_match={x[11]}\n  L: {x[7]} | {x[8]}\n  R: {x[9]} | {x[10]}")
    con.execute(f"COPY ex TO '{OUT}/ex_{what}.parquet' (FORMAT parquet)")
