"""Task 1: reproduce v4 holdout F0.5 from pred2_holdout + decision_params, and a loss decomposition."""
import json, time
import numpy as np
from common import RUN, arrays, params, con, note, cfg
from ber.decision import decide, per_entity_f05
from ber.pipeline import _fetch_numpy
from ber.split import group_where

t0 = time.time()
h = arrays(RUN / "pred2_holdout.parquet", "holdout", extra_ids=True)
t = arrays(RUN / "pred2_tune.parquet", "tune", extra_ids=True)
print("arrays", time.time() - t0, len(h["p"]), len(t["p"]))
P = params()
li, p, lab, nt = h["lid_idx"], h["p"], h["label"], h["n_true"]
sel = decide(P, li, h["rid"], p)
f = per_entity_f05(li, sel, lab, nt)
n = len(nt)
tot_in_file = _fetch_numpy(con(), f"""SELECT count(*) AS c FROM read_parquet('{RUN / "pred2_holdout.parquet"}')
    WHERE label = 1 AND {group_where('lid', 'holdout')}""")["c"][0]
in_cand = np.bincount(li, weights=lab.astype(float), minlength=n)
F = lambda s, ntt=nt: per_entity_f05(li, s, lab, ntt).mean()
r = {"holdout_f05": f.mean(), "entities": n, "pairs": len(p), "true_pairs": int(nt.sum()),
     "TP": int((sel & lab).sum()), "FP": int((sel & ~lab).sum()), "FP_singletons": int((sel & ~lab & (nt[li] == 0)).sum()),
     "decision_FN": int((~sel & lab).sum()), "blocking_miss": int(nt.sum() - tot_in_file),
     "one_to_one_loss": int(tot_in_file - lab.sum()),
     "oracle_perfect_decisions": F(lab.copy()), "fix_decision_FN": F(sel | lab), "fix_all_FP": F(sel & lab),
     "fix_FP_singletons": F(sel & (lab | (nt[li] > 0))),
     "fix_blocking+o2o": F(sel, np.minimum(nt, in_cand)),
     "fix_blocking_only(approx)": None}
for c in sorted(set(h["country"])):
    m = h["country"] == c
    r[f"f05_{c}"] = f[m].mean(); r[f"n_{c}"] = int(m.sum())
for lo, hi, name in ((0, 0, "singleton"), (1, 1, "n=1"), (2, 3, "n=2-3"), (4, 10**6, "n>=4")):
    m = (nt >= lo) & (nt <= hi)
    r[f"f05_{name}"] = f[m].mean(); r[f"share_{name}"] = m.mean(); r[f"loss_pts_{name}"] = (1 - f[m]).sum() / n * 100
r.pop("fix_blocking_only(approx)")
out = {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
print(json.dumps(out, indent=1))
note("## T1. v4 reproduction and loss decomposition (holdout)\n\n```\n" + json.dumps(out, indent=1) + "\n```")
