"""Task 3c: retrain stage 2 on s2x files with a chosen feature set / seed; score holdout with a tune-fitted decision.

usage: t3_train.py NAME FEATSET SEED [extra lgb params as k=v ...]
FEATSET: top (top-60 v4 stage-2 features) | topE (top-60 + E entity features) | E-list names comma separated.
"""
import json, sys, time, resource
import numpy as np
from pathlib import Path
from common import OUT, arrays, note
from t3_feats import TOP, E
from ber.model import train, predict_file
from ber.decision import compare_decisions

name, fs, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
extra = {k: (float(v) if "." in v else int(v)) for k, v in (a.split("=") for a in sys.argv[4:])}
feats = {"top": TOP, "topE": TOP + E}.get(fs) or (TOP + fs.split(","))
params = {"learning_rate": 0.2, **extra}
t0 = time.time()
import os
b = train(Path(os.environ.get("FIT", OUT / "s2x_fit_in.parquet")), OUT / "s2x_tune_in.parquet", params, seed, 5, rounds=5000, features=feats)
tt = time.time() - t0
b.save_model(str(OUT / f"model_{name}.txt"), num_iteration=b.best_iteration)
for g in ("tune", "holdout"):
    predict_file(b, OUT / f"s2x_{g}.parquet", OUT / f"pred_{name}_{g}.parquet", 200000, features=feats)
t = arrays(OUT / f"pred_{name}_tune.parquet", "tune")
h = arrays(OUT / f"pred_{name}_holdout.parquet", "holdout")
r = compare_decisions(t, h)
imp = sorted(zip(feats, b.feature_importance("gain")), key=lambda x: -x[1])
tot = sum(v for _, v in imp)
res = {"name": name, "feats": fs, "seed": seed, "params": params, "best_iter": b.best_iteration,
       "valid_logloss": round(float(b.best_score["valid"]["binary_logloss"]), 6), "train_s": round(tt),
       "holdout": {k: round(v, 5) for k, v in r["holdout"].items()}, "tune": {k: round(v, 5) for k, v in r["tune"].items()},
       "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
       "top_imp": [(k, round(v / tot * 100, 2)) for k, v in imp[:12] if v > 0]}
print(json.dumps(res), flush=True)
with open(OUT / "t3_results.jsonl", "a") as fh:
    fh.write(json.dumps(res) + "\n")
note(f"- stage-2 retrain `{name}` feats={fs} seed={seed} params={params}: holdout expF **{res['holdout']['expected_f']}** "
     f"(thr {res['holdout']['threshold']}), tune expF {res['tune']['expected_f']}, valid logloss {res['valid_logloss']}, "
     f"iters {res['best_iter']}, {res['train_s']} s. top gain: {res['top_imp'][:8]}")
