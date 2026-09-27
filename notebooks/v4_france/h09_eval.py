"""Compare France decisions for pred2 variants (v3 submitted = pred2_test France rows, base = reproduction, fr = French
vocab rescoring, and optional extra files) with label-free precision proxies and pseudo-label agreement."""
import json
import sys

import numpy as np

from h_common import C, RUN, TMP, connect

sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber.decision import best_per_right, calibrate, em_prior_shift  # noqa: E402

params = json.loads((RUN / "decision_params.json").read_text())
pj = json.loads((RUN / "predict.json").read_text())
TRAIN_PRIOR = pj["train_prior"]
FR_DELTA = pj["prior_shift_by_country"]["France"]["logit_delta"]
T_CAL = float(calibrate(params, np.array([True]), np.array([params["threshold"]]), shift=0.0)[0])


def decide(p, rc, li, delta=None):
    """Accept mask (v3 threshold rule) and the delta used; delta None = re-estimate by EM on this variant."""
    best = best_per_right(rc, li, p)
    if delta is None:
        pc0 = calibrate(params, best, p, shift=0.0)
        prior, _ = em_prior_shift(pc0[best], TRAIN_PRIOR)
        delta = float(np.log(prior / (1 - prior)) - np.log(TRAIN_PRIOR / (1 - TRAIN_PRIOR)))
    pc = calibrate(params, best, p, shift=delta)
    return best & (pc >= T_CAL), delta, best


def main(variants):
    con = connect()
    srcs = {"v3": f"(SELECT p.* FROM read_parquet('{RUN}/pred2_test.parquet') p SEMI JOIN read_parquet('{TMP}/pairs_fr.parquet') x USING (lid, rid))"}
    srcs.update({v: f"read_parquet('{TMP}/pred2_fr_{v}.parquet')" for v in variants})
    base = con.execute(f"""SELECT x.lid, x.rid, dense_rank() OVER (ORDER BY x.lid)-1 li, dense_rank() OVER (ORDER BY x.rid)-1 rc,
        coalesce(ps.rule,'none') AS rl, x.sg_sole_off_in_s=1 AS d1,
        CASE WHEN x.sg_num_changed=1 AND x.sg_num_trunc=0 AND x.sg_num_add_digit=0 THEN x.sg_first_off END AS off
        FROM read_parquet('{TMP}/pairs_fr.parquet') x LEFT JOIN read_parquet('{TMP}/fr_pseudo.parquet') ps USING (lid, rid)
        ORDER BY x.lid, x.rid""").fetchnumpy()
    li, rc = base["li"].astype(np.int64), base["rc"].astype(np.int64)
    rule = base["rl"].astype(str); off = np.nan_to_num(base["off"].astype(float), nan=-999)
    n_l = li.max() + 1
    res = {}
    for v, s in srcs.items():
        p = con.execute(f"""SELECT coalesce(s.p, 0) p FROM read_parquet('{TMP}/pairs_fr.parquet') x LEFT JOIN {s} s USING (lid, rid)
                            ORDER BY x.lid, x.rid""").fetchnumpy()["p"].astype(np.float64)
        for dname, dl in (("fixedshift", FR_DELTA), ("em", None)):
            m, d, best = decide(p, rc, li, dl)
            npl = np.bincount(li[m], minlength=n_l)
            ex = lambda a: int(((off == a) & m).sum())  # noqa: E731
            r = {"delta": round(d, 3), "accepted": int(m.sum()), "matches_per_S1": round(m.sum() / n_l, 4),
                 "empty_S1": round(float((npl == 0).mean()), 4),
                 "uncertain_best_share(0.05<p<0.95)": round(float(((p > 0.05) & (p < 0.95) & best).sum() / best.sum()), 4),
                 "uncertain_all_share": round(float(((p > 0.05) & (p < 0.95)).mean()), 4),
                 "acc_D1": int((m & (rule == "D1")).sum()), "acc_D2": int((m & (rule == "D2")).sum()),
                 "acc_E": int((m & (rule == "E")).sum()),
                 "acc_T3_share": round(float(m[rule == "T3"].mean()), 4), "acc_T4_share": round(float(m[rule == "T4"].mean()), 4),
                 "acc_unlabelled": int((m & (rule == "none")).sum()),
                 "acc_off+1+2": ex(1) + ex(2), "acc_off-1-2": ex(-1) + ex(-2),
                 "est_P12_distractors": ex(1) + ex(2) - ex(-1) - ex(-2),
                 "acc_offS>=3": int(sum(ex(a) for a in (3, 4, 5, 7, 9, 11, 13, 21)))}
            res[(v, dname)] = (r, m)
            print(v, dname, r, flush=True)
    if "base" in variants:
        pv = con.execute(f"""SELECT max(abs(a.p-b.p)), avg(abs(a.p-b.p)) FROM {srcs['v3']} a JOIN {srcs['base']} b USING (lid, rid)""").fetchone()
        print("reproduction v3 vs base: max|dp|, mean|dp|", pv)
    keys = list(res)
    ref = res[("v3", "fixedshift")][1]
    for k in keys:
        m = res[k][1]
        print(k, "flips vs v3: added", int((m & ~ref).sum()), "removed", int((~m & ref).sum()),
              "removed by rule:", {r: int((~m & ref & (rule == r)).sum()) for r in ("T3", "T4", "D1", "D2", "E", "none")},
              "removed with off+1/+2:", int((~m & ref & ((off == 1) | (off == 2))).sum()),
              "added with off+1/+2:", int((m & ~ref & ((off == 1) | (off == 2))).sum()))


if __name__ == "__main__":
    main(sys.argv[1:])
