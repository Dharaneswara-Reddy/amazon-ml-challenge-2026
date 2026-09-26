"""Loss decomposition (holdout) for a pred spec (tune-fitted expected_f decision); compare to v4."""
import sys, json
import numpy as np
from common import arrays, note
from t5_eval import combo
from ber.decision import fit_decision, decide, per_entity_f05
for spec in sys.argv[1:]:
    t, h = arrays(combo(spec, "tune"), "tune"), arrays(combo(spec, "holdout"), "holdout")
    P = fit_decision(t)
    li, lab, nt = h["lid_idx"], h["label"], h["n_true"]
    sel = decide(P, li, h["rid"], h["p"])
    F = lambda s: per_entity_f05(li, s, lab, nt).mean()
    r = {"spec": spec, "F": round(F(sel), 5), "TP": int((sel & lab).sum()), "FP": int((sel & ~lab).sum()),
         "FN_in_cands": int((~sel & lab).sum()), "oracle_on_cands": round(F(lab.copy()), 5),
         "fix_FN": round(F(sel | lab), 5), "fix_FP": round(F(sel & lab), 5)}
    print(r, flush=True)
    note(f"- decomposition `{spec}`: {r}")
