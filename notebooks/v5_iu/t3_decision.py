"""Task 3a: decision-layer variants on v4 pred2 (fit on tune, score holdout). No retraining."""
import json, sys
import numpy as np
from common import RUN, arrays, note
from ber.decision import (compare_decisions, choose_decision, fit_decision, decide, macro_f05, fit_calibrator,
                          best_per_right, prepare, calibrate, choose_k, MissingModel, Calibrator, mask_from_k)

pred = sys.argv[1] if len(sys.argv) > 1 else "v4"
if pred == "v4":
    t = arrays(RUN / "pred2_tune.parquet", "tune", extra_ids=True)
    h = arrays(RUN / "pred2_holdout.parquet", "holdout", extra_ids=True)
res = {}
def ev(name, tt, hh, **kw):
    """Fit decision on tt, score both rules on hh; record expected_f and threshold holdout F."""
    r = compare_decisions(tt, hh, **kw)
    res[name] = {"exp_f": round(r["holdout"]["expected_f"], 5), "thr": round(r["holdout"]["threshold"], 5),
                 "exp_f_model": round(r["holdout"]["expected_f_model"], 5), "tune_exp_f": round(r["tune"]["expected_f"], 5)}
    print(name, res[name], flush=True)
    return r
base = ev("base", t, h)
for mk in (8, 24):
    ev(f"max_k={mk}", t, h, max_k=mk)
ev("missing_bins=40", t, h, n_missing_bins=40)

# per-source isotonic pre-calibration (S2 vs S3 x R-best vs rival), fitted on tune
def src(a):
    return np.array([s.startswith("S3-") for s in a["rid_s"]])
def precal(tt, hh):
    out = []
    bt = best_per_right(tt["rid"], tt["lid_idx"], tt["p"]); st = src(tt)
    cals = {}
    for b in (True, False):
        for s in (True, False):
            m = (bt == b) & (st == s)
            cals[(b, s)] = fit_calibrator(tt["p"][m], tt["label"][m])
    for a in (tt, hh):
        b = best_per_right(a["rid"], a["lid_idx"], a["p"]); s = src(a)
        q = np.empty_like(a["p"])
        for (bb, ss), c in cals.items():
            m = (b == bb) & (s == ss)
            q[m] = c(a["p"][m])
        # keep raw order inside ties: add tiny raw-score term
        out.append({**a, "p": np.clip(q, 0, 1) * (1 - 1e-6) + 1e-6 * a["p"]})
    return out
tt2, hh2 = precal(t, h)
ev("per_source_precal", tt2, hh2)

# second-choice reassignment: rights whose best pair is rejected get their next pair considered
def second_choice(params, a):
    li, rid, p = a["lid_idx"], a["rid"], a["p"]
    sel = decide(params, li, rid, p)
    matched_r = np.zeros(rid.max() + 1, bool); matched_r[rid[sel]] = True
    best = best_per_right(rid, li, p)
    drop = best & ~matched_r[rid]
    keep = ~drop
    sel2 = np.zeros(len(p), bool)
    sel2[keep] = decide(params, li[keep], rid[keep], p[keep])
    return sel, sel2
P = base["params"]
s1, s2 = second_choice(P, h)
res["second_choice"] = {"exp_f": round(macro_f05(h["lid_idx"], s2, h["label"], h["n_true"]), 5),
                        "first": round(macro_f05(h["lid_idx"], s1, h["label"], h["n_true"]), 5),
                        "added_pairs": int((s2 & ~s1).sum()), "added_true": int((s2 & ~s1 & h["label"]).sum()),
                        "removed": int((s1 & ~s2).sum()), "removed_true": int((s1 & ~s2 & h["label"]).sum())}
print("second_choice", res["second_choice"])
note("## T3a. Decision-layer variants on v4 pred2 (fit tune -> holdout)\n\n```\n" + json.dumps(res, indent=1) + "\n```")
