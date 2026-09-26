"""Task 4: characterize holdout blocking misses (GT pairs of holdout S1 absent from cand3_train) and a sibling-propagation
candidate rule: for each SELECTED holdout pair (L, R'), propose (L, R) for rights R with the same country and exact
name_core as R' (R not already a candidate of L). Measures recovered misses and pair cost."""
import json
from common import ROOT, OUT, con, note
from ber.split import group_where
C = ROOT / "artifacts/cache"
c = con()
c.execute(f"""CREATE OR REPLACE TEMP TABLE miss AS
  SELECT g.s1_id lid, g.match_id rid FROM read_parquet('{C}/train_gt_pairs.parquet') g
  ANTI JOIN read_parquet('{C}/cand3_train.parquet') k ON k.lid = g.s1_id AND k.rid = g.match_id
  WHERE {group_where('g.s1_id', 'holdout')}""")
c.execute(f"""CREATE OR REPLACE TEMP TABLE rn AS SELECT entity_id, name_core, coalesce(length(trim(addr_norm)),0)=0 noaddr, country_norm
  FROM read_parquet(['{C}/norm_train_s2.parquet','{C}/norm_train_s3.parquet'])""")
c.execute(f"CREATE OR REPLACE TEMP TABLE ln AS SELECT entity_id, name_core, addr_norm, country_norm FROM read_parquet('{C}/norm_train_s1.parquet') WHERE {group_where('entity_id','holdout')}")
q = lambda s: c.execute(s).fetchall()
out = {}
out["misses"] = q("SELECT count(*) FROM miss")[0][0]
out["miss_R_noaddr"] = q("SELECT sum(r.noaddr::INT) FROM miss m JOIN rn r ON r.entity_id = m.rid")[0][0]
out["miss_by_src"] = q("SELECT left(rid,2), count(*) FROM miss GROUP BY ALL")
out["miss_name_core_equal"] = q("SELECT sum((r.name_core = l.name_core)::INT) FROM miss m JOIN rn r ON r.entity_id=m.rid JOIN ln l ON l.entity_id=m.lid")[0][0]
out["miss_jw_name_bins"] = q("""SELECT CASE WHEN s>=0.95 THEN '>=.95' WHEN s>=0.85 THEN '.85-.95' WHEN s>=0.7 THEN '.7-.85' ELSE '<.7' END b, count(*)
  FROM (SELECT jaro_winkler_similarity(r.name_core, l.name_core) s FROM miss m JOIN rn r ON r.entity_id=m.rid JOIN ln l ON l.entity_id=m.lid) GROUP BY ALL ORDER BY ALL""")
# sibling propagation from v4 selected pairs
c.execute(f"""CREATE OR REPLACE TEMP TABLE selp AS SELECT e.s1_id lid, p.rid_s rid FROM read_parquet('{OUT}/holdout_pairs_v4.parquet') p
  JOIN read_parquet('{OUT}/holdout_ent_v4.parquet') e USING (li) WHERE p.sel""")
c.execute(f"""CREATE OR REPLACE TEMP TABLE prop AS
  SELECT DISTINCT s.lid, r2.entity_id rid FROM selp s JOIN rn r1 ON r1.entity_id = s.rid
  JOIN rn r2 ON r2.name_core = r1.name_core AND r2.country_norm = r1.country_norm AND r2.entity_id <> r1.entity_id
  WHERE length(r1.name_core) >= 4""")
c.execute(f"""CREATE OR REPLACE TEMP TABLE prop_new AS SELECT p.* FROM prop p
  ANTI JOIN read_parquet('{C}/cand3_train.parquet') k ON k.lid = p.lid AND k.rid = p.rid""")
out["prop_new_pairs"] = q("SELECT count(*), count(DISTINCT lid) FROM prop_new")[0]
out["prop_new_true"] = q("SELECT count(*) FROM prop_new p JOIN miss m USING (lid, rid)")[0][0]
# restrict: R name-core df (number of rights with that name) small
c.execute("CREATE OR REPLACE TEMP TABLE ndf AS SELECT name_core, country_norm, count(*) df FROM rn GROUP BY ALL")
out["prop_new_by_rdf"] = q("""SELECT least(n.df, 10) d, count(*) pairs, count(m.lid) n_true FROM prop_new p JOIN rn r ON r.entity_id=p.rid
  JOIN ndf n ON n.name_core=r.name_core AND n.country_norm=r.country_norm LEFT JOIN miss m ON m.lid=p.lid AND m.rid=p.rid GROUP BY ALL ORDER BY 1""")
out["holdout_S1"] = q(f"SELECT count(*) FROM ln")[0][0]
print(json.dumps(out, indent=1, default=str))
note("## T4. Blocking misses (holdout) and a sibling-propagation candidate rule\n\n```\n" + json.dumps(out, indent=1, default=str) + "\n```")
