"""Per-country / per-size holdout F0.5 of a pred spec vs v4 (tune-fitted expected_f decision)."""
import sys, json
import numpy as np
from common import RUN, arrays, note
from t5_eval import entity_f
h4 = arrays(RUN / "pred2_holdout.parquet", "holdout", extra_ids=True)
country, nt = h4["country"].astype(str), h4["n_true"]
rows = []
for spec in ["v4"] + sys.argv[1:]:
    f = entity_f(spec)
    r = {"spec": spec, "all": round(f.mean(), 5)}
    for c in ("india", "us"):
        r[c] = round(f[country == c].mean(), 5)
    for lo, hi, nm in ((0, 0, "n0"), (1, 1, "n1"), (2, 3, "n2-3"), (4, 99, "n4+")):
        m = (nt >= lo) & (nt <= hi); r[nm] = round(f[m].mean(), 5)
    rows.append(r); print(r, flush=True)
note("### Per-country / entity-size holdout F0.5\n\n| spec | all | india | us | n=0 | n=1 | n=2-3 | n>=4 |\n|---|---|---|---|---|---|---|---|\n" +
     "\n".join(f"| {r['spec']} | {r['all']} | {r['india']} | {r['us']} | {r['n0']} | {r['n1']} | {r['n2-3']} | {r['n4+']} |" for r in rows))
