"""Sweep the decision's empty_weight on a run: test empty-prediction share per country (with the run's per-country
EM shift, France unshifted) and holdout macro F0.5 (in-domain cost). Usage: python empty_sweep.py RUN_DIR w1 w2 ..."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber import io  # noqa: E402
from ber.config import load_config  # noqa: E402
from ber.decision import best_per_right, calibrate, decide, em_prior_shift, per_entity_f05  # noqa: E402
from ber.duck import connect  # noqa: E402
from ber.pipeline import _fetch_numpy, _group_arrays  # noqa: E402

run = Path(sys.argv[1])
ws = [float(x) for x in sys.argv[2:]]
cfg = load_config()
params = json.loads((run / "decision_params.json").read_text())
man = json.loads((run / "manifest.json").read_text())
con = connect(cfg)
s1 = io.raw_parquet_path(cfg, "test", 1)
con.execute(f"""CREATE TABLE pt AS SELECT x.lid, x.rid, x.p, dense_rank() OVER (ORDER BY x.lid) - 1 AS li,
    dense_rank() OVER (ORDER BY x.rid) - 1 AS rc, s.country FROM read_parquet('{run / "pred2_test.parquet"}') x
    JOIN read_parquet('{s1}') s ON s.entity_id = x.lid""")
d = _fetch_numpy(con, "SELECT li, rc, p, country FROM pt")
li, rc, p, cty = d["li"].astype(np.int64), d["rc"].astype(np.int64), d["p"].astype(np.float64), d["country"]
n_s1 = dict(con.execute(f"SELECT country, count(*) FROM read_parquet('{s1}') GROUP BY 1").fetchall())
ent_cty = {}
for i, c in zip(li, cty):
    ent_cty.setdefault(int(i), c)
tp = float(man["metrics"]["train_prior_best"])
is_best = best_per_right(rc, li, p)
pc = calibrate(params, is_best, p, shift=0.0)
shift = np.full(len(p), float(params.get("shift", 0.0)))
for c in set(cty):
    if c == "France":
        continue
    m = cty == c
    q, _ = em_prior_shift(pc[is_best & m], tp)
    shift[m] += np.log(q / (1 - q)) - np.log(tp / (1 - tp))
h = _group_arrays(con, cfg, run / "pred2_holdout.parquet", "holdout")
ec = np.array([ent_cty[i] for i in range(li.max() + 1)])
for w in ws:
    prm = {**params, "empty_weight": w}
    mask = decide(prm, li, rc, p, shift)
    has = np.bincount(li, weights=mask.astype(float), minlength=li.max() + 1) > 0
    row = {c: round(1 - (has[ec == c].sum()) / n_s1[c], 4) for c in sorted(n_s1)}
    tot = round(1 - has.sum() / sum(n_s1.values()), 4)
    hf = per_entity_f05(h["lid_idx"], decide(prm, h["lid_idx"], h["rid"], h["p"]), h["label"], h["n_true"]).mean()
    print(json.dumps({"empty_weight": w, "test_empty_share": tot, "by_country": row, "holdout_f05": round(float(hf), 5)}), flush=True)
