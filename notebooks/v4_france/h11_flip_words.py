"""Generator-word rate (French distractor vocab isolated in h05, lo>=3) among flipped pairs: added (now accepted) and
removed pairs with all numbers equal. Distractor share ~= hit rate / 0.454 (hit rate of number-anchored distractors)."""
import multiprocessing as mp
from collections import Counter

import h05_generator_vocab as h5
from h_common import C, TMP, connect, norm_right

con = connect()
voc = h5.run(con, f"{TMP}/pairs_fr.parquet", "test", "FR")
gen = {k: {w for w, v in voc[k].items() if v >= 3.0} for k in voc}
import numpy as np, json, sys
sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber.decision import best_per_right, calibrate
RUN = h5.__dict__.get("RUN")
from h_common import RUN
params = json.loads((RUN / "decision_params.json").read_text()); pj = json.loads((RUN / "predict.json").read_text())
t = float(calibrate(params, np.array([True]), np.array([params["threshold"]]), shift=0.0)[0])
d = con.execute(f"""SELECT x.lid, x.rid, dense_rank() OVER (ORDER BY x.lid)-1 li, dense_rank() OVER (ORDER BY x.rid)-1 rc, x.p a, n.p b,
   (coalesce(x.sg_num_changed,1)=0 AND coalesce(x.sg_num_eq,0)>=1) neq
   FROM read_parquet('{TMP}/pairs_fr.parquet') x JOIN read_parquet('{TMP}/pred2_fr_fr.parquet') n USING (lid, rid) ORDER BY x.lid, x.rid""").fetchnumpy()
li, rc = d["li"].astype(np.int64), d["rc"].astype(np.int64)
m = {}
for k in ("a", "b"):
    p = d[k].astype(np.float64); bb = best_per_right(rc, li, p)
    m[k] = bb & (calibrate(params, bb, p, shift=pj["prior_shift_by_country"]["France"]["logit_delta"]) >= t)
grp = np.where(m["b"] & ~m["a"], "added", np.where(m["a"] & ~m["b"], np.where(d["neq"], "removed_numeq", "removed_other"),
               np.where(m["a"] & m["b"], "kept", "")))
con.register("g", {"lid": d["lid"], "rid": d["rid"], "g": grp})
rows = con.execute(f"""WITH s AS (SELECT *, row_number() OVER (PARTITION BY g ORDER BY hash(lid||rid)) k FROM g WHERE g <> '')
   SELECT s.g, -1, l.name_norm, r.name_norm FROM s JOIN read_parquet('{C}/norm_test_s1.parquet') l ON l.entity_id=s.lid
   JOIN {norm_right()} r ON r.entity_id=s.rid WHERE k <= 60000""").fetchall()
with mp.get_context("fork").Pool(4) as pool:
    res = pool.map(h5.ops, rows, chunksize=5000)
cnt = Counter(); hit = Counter(); words = {}
for g, _, ins, rep, tgt in res:
    cnt[g] += 1
    hs = (set(ins) & gen["ins"]) | (set(rep) & gen["repl"]) | (set(tgt) & gen["rtgt"])
    if hs:
        hit[g] += 1
        words.setdefault(g, Counter()).update(hs)
for g in cnt:
    print(g, "n", cnt[g], "generator-word share", round(hit[g] / cnt[g], 4), "-> est distractor share", round(hit[g] / cnt[g] / 0.454, 3),
          "top:", words.get(g, Counter()).most_common(8))
print("group sizes:", Counter(grp.tolist()))
