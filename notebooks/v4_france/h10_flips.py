"""Inspect decision flips between v3 and the French-vocab rescoring: switches vs new/dropped rights, their signatures
and examples; plus India/US uncertain shares from pred2_test for reference."""
import json
import sys

import numpy as np

from h_common import C, RUN, TMP, connect, raw_right

sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber.decision import best_per_right, calibrate  # noqa: E402

params = json.loads((RUN / "decision_params.json").read_text())
pj = json.loads((RUN / "predict.json").read_text())
T_CAL = float(calibrate(params, np.array([True]), np.array([params["threshold"]]), shift=0.0)[0])
con = connect()
d = con.execute(f"""SELECT x.lid, x.rid, dense_rank() OVER (ORDER BY x.lid)-1 li, dense_rank() OVER (ORDER BY x.rid)-1 rc,
    x.p p_old, n.p p_new FROM read_parquet('{TMP}/pairs_fr.parquet') x JOIN read_parquet('{TMP}/pred2_fr_fr.parquet') n USING (lid, rid)
    ORDER BY x.lid, x.rid""").fetchnumpy()
li, rc = d["li"].astype(np.int64), d["rc"].astype(np.int64)
masks = {}
for k in ("p_old", "p_new"):
    p = d[k].astype(np.float64)
    b = best_per_right(rc, li, p)
    masks[k] = b & (calibrate(params, b, p, shift=pj["prior_shift_by_country"]["France"]["logit_delta"]) >= T_CAL)
old, new = masks["p_old"], masks["p_new"]
con.register("fl", {"lid": d["lid"], "rid": d["rid"], "old": old, "new": new, "p_old": d["p_old"], "p_new": d["p_new"]})
con.execute("CREATE TEMP TABLE f AS SELECT * FROM fl WHERE old <> new")
con.execute("""CREATE TEMP TABLE rr AS SELECT rid, bool_or(old AND NOT new) had_drop, bool_or(new AND NOT old) had_add FROM f GROUP BY rid""")
print("rights with flips: switch (drop+add), add only, drop only:", con.execute(
    "SELECT count(*) FILTER (WHERE had_drop AND had_add), count(*) FILTER (WHERE had_add AND NOT had_drop), count(*) FILTER (WHERE had_drop AND NOT had_add) FROM rr").fetchone())
sig = f"""SELECT f.*, CASE WHEN f.new THEN 'added' ELSE 'removed' END kind,
   CASE WHEN rr.had_drop AND rr.had_add THEN 'switch' ELSE 'solo' END sw, x.sg_num_changed, x.sg_first_off, x.sg_sole_off_in_s,
   x.sg_n_ins, x.sg_n_repl, x.sg_num_eq, x.sg_r_addr_empty FROM f JOIN rr USING (rid) JOIN read_parquet('{TMP}/pairs_fr.parquet') x USING (lid, rid)"""
print("kind, switch?, n, share nums all equal(changed=0 & eq>=1), share changed with off +1/+2, share D1, share name ins/repl, share r addr empty")
for r in con.execute(f"""SELECT kind, sw, count(*), avg((sg_num_changed=0 AND sg_num_eq>=1)::INT), avg((sg_num_changed=1 AND sg_first_off IN (1,2))::INT),
   avg(sg_sole_off_in_s), avg((sg_n_ins+sg_n_repl>0)::INT), avg(sg_r_addr_empty) FROM ({sig}) GROUP BY ALL ORDER BY ALL""").fetchall():
    print(r)
print("--- examples")
for kind, sw in (("removed", "solo"), ("added", "solo"), ("removed", "switch"), ("added", "switch")):
    print("==", kind, sw)
    for r in con.execute(f"""SELECT round(s.p_old,3), round(s.p_new,3), l.business_name, l.business_address, r.business_name, r.business_address
       FROM ({sig}) s JOIN read_parquet('{C}/raw_test_s1.parquet') l ON l.entity_id=s.lid JOIN {raw_right()} r ON r.entity_id=s.rid
       WHERE kind='{kind}' AND sw='{sw}' AND NOT (sg_num_changed=1 AND sg_first_off IN (1,2)) ORDER BY hash(s.rid) LIMIT 10""").fetchall():
        print("  ", r)
print("--- India/US reference from pred2_test: rid-best share 0.05<p<0.95, all-pair share")
con.execute(f"""CREATE TEMP TABLE q AS SELECT p.lid, p.rid, p.p, s.country, row_number() OVER (PARTITION BY p.rid ORDER BY p.p DESC, p.lid) rk
   FROM read_parquet('{RUN}/pred2_test.parquet') p JOIN read_parquet('{C}/raw_test_s1.parquet') s ON s.entity_id=p.lid""")
print(con.execute("SELECT country, avg((p>0.05 AND p<0.95)::INT) FILTER (WHERE rk=1), avg((p>0.05 AND p<0.95)::INT) FROM q GROUP BY 1").fetchall())
