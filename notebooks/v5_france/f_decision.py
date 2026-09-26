"""F: France-only decision variants on the v4 stage-2 scores (pred2_test, France pairs; everything else fixed).
Variants: logit shift s of calibrated p (EM estimate for France = logit(0.625) - logit(0.720) = -0.43), empty_weight,
max_k. For each: pairs changed vs v4 (#6 France) and label-free characterization of the ADDED / REMOVED pairs:
same-name same-number same-street (near-certain true), same-name baseline offsets (near-certain false),
+1/+2 vs -1/-2 single-number changes (excess = distractors), generator offsets >= 3, entity empty share, matches/S1."""
import json
import numpy as np, pyarrow as pa
from common import *
from ber.decision import prepare, choose_k, calibrate, MissingModel, Calibrator, mask_from_k
c = connect()
params = json.load(open(RUN / "decision_params.json"))
c.execute(f"""CREATE TABLE fp AS SELECT row_number() OVER () - 1 i, x.lid, x.rid, x.p,
    dense_rank() OVER (ORDER BY x.lid) - 1 li, dense_rank() OVER (ORDER BY x.rid) - 1 rc
    FROM '{RUN}/pred2_test.parquet' x JOIN '{CACHE}/raw_test_s1.parquet' s ON s.entity_id = x.lid AND s.country = 'France'""")
c.execute(f"""CREATE TABLE ann AS SELECT fp.i,
    coalesce(max(CASE WHEN x.d = 0 AND x.sb = 'same' THEN 1 END), 0) sure_t,
    coalesce(max(CASE WHEN x.d <> 0 THEN 1 END), 0) sure_f,
    coalesce(max(CASE WHEN x.d = 0 AND x.sb <> 'same' THEN 1 END), 0) coll
    FROM fp JOIN '{TMP}/fr_cells.parquet' x USING (lid, rid) GROUP BY 1""")
c.execute(f"""CREATE TABLE sig AS SELECT fp.i, (f.sg_num_changed = 1 AND f.sg_first_off IN (1, 2))::INT p12,
    (f.sg_num_changed = 1 AND f.sg_first_off IN (-1, -2))::INT m12, (f.sg_num_changed = 1 AND f.sg_first_off IN (3,4,5,7,9,11,13,21))::INT gen
    FROM fp JOIN '{FEAT['test']}' f USING (lid, rid)""")
d = c.execute("SELECT li, rc, p FROM fp ORDER BY i").fetchnumpy()
li, rc, p = d["li"].astype(np.int64), d["rc"].astype(np.int64), d["p"].astype(np.float64)
n_ent = int(li.max()) + 1
def run(shift=0.0, w=None, max_k=None):
    """Decision mask for France under a variant."""
    prep = prepare(li, rc, p, int(max_k or params["max_k"]))
    pc = calibrate(params, prep.is_best, p, shift=shift)
    k, ef = choose_k(prep, pc, MissingModel.from_dict(params["missing"]), float(params["empty_weight"] if w is None else w),
                     Calibrator.from_dict(params["cal_empty"]))
    return mask_from_k(prep, k)
base = run(0.0)
def describe(name, m):
    """Label-free summary of a variant vs the v4 France decision."""
    add, rem = m & ~base, base & ~m
    c.register("ad", pa.table({"i": np.flatnonzero(add)})); c.register("rm", pa.table({"i": np.flatnonzero(rem)}))
    def prof(t):
        return c.execute(f"""SELECT count(*), coalesce(sum(a.sure_t),0), coalesce(sum(a.sure_f),0), coalesce(sum(a.coll),0),
            coalesce(sum(s.p12),0), coalesce(sum(s.m12),0), coalesce(sum(s.gen),0)
            FROM {t} LEFT JOIN ann a USING (i) LEFT JOIN sig s USING (i)""").fetchone()
    k = np.bincount(li[m], minlength=n_ent)
    r = {"accepted": int(m.sum()), "empty_share": float((k == 0).mean()), "added": prof("ad"), "removed": prof("rm")}
    print(f"{name:28s} acc={r['accepted']:,} empty={r['empty_share']:.4f}  ADDED n/sameStreetTrue/baseOffFalse/coll/+12/-12/gen={r['added']}  REMOVED={r['removed']}", flush=True)
    return r
res = {"v4": describe("v4 (shift 0)", base)}
for s in (-0.75, -0.43, -0.25, 0.25, 0.5, 1.0):
    res[f"shift{s}"] = describe(f"shift {s:+.2f}", run(s))
for w in (0.6, 1.0, 1.5):
    res[f"ew{w}"] = describe(f"empty_weight {w}", run(0.0, w=w))
for mk in (8, 24):
    res[f"maxk{mk}"] = describe(f"max_k {mk}", run(0.0, max_k=mk))
json.dump(res, open(OUT / "f_decision.json", "w"), indent=1)
