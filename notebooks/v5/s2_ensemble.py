"""Stage-2 seed ensemble on a finished run: train stage 2 with extra seeds on the run's s2 files, average the
stage-2 probabilities with the run's own pred2, and score holdout with a tune-chosen decision (same protocol as
ber.pipeline.train_stage). Usage: python s2_ensemble.py RUN_DIR SEED [SEED ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber.config import load_config  # noqa: E402
from ber.decision import choose_decision  # noqa: E402
from ber.duck import connect  # noqa: E402
from ber.model import predict_file, train  # noqa: E402
from ber.pipeline import _group_arrays  # noqa: E402
from ber.stage2 import MODEL2_FEATURES  # noqa: E402

run = Path(sys.argv[1])
seeds = [int(s) for s in sys.argv[2:]]
cfg = load_config()
params = dict(dict(cfg.get("model", {})).get("params", {}))
out = run / "ens"
out.mkdir(exist_ok=True)
for s in seeds:
    if (out / f"pred2_holdout_s{s}.parquet").exists():
        continue
    b = train(run / "s2_fit_in.parquet", run / "s2_tune_in.parquet", params, s, 10, rounds=5000, features=MODEL2_FEATURES)
    b.save_model(str(out / f"model2_s{s}.txt"), num_iteration=b.best_iteration)
    for g in ("tune", "holdout"):
        predict_file(b, run / f"s2_{g}.parquet", out / f"pred2_{g}_s{s}.parquet", 200_000, features=MODEL2_FEATURES)
    print("seed", s, "iters", b.best_iteration, flush=True)


def averaged(g: str, members: list[Path]) -> Path:
    """Average p over prediction files (identical row order) into one parquet; return its path."""
    base = pq.read_table(members[0])
    p = np.mean([pq.read_table(m, columns=["p"]).column("p").to_numpy() for m in members], axis=0)
    t = base.set_column(base.schema.get_field_index("p"), "p", pa.array(p.astype(np.float32)))
    dst = out / f"pred2_{g}_avg{len(members)}.parquet"
    pq.write_table(t, dst)
    return dst


con = connect(cfg)
res = {}
for k in range(0, len(seeds) + 1):
    arrs = {}
    for g in ("tune", "holdout"):
        members = [run / f"pred2_{g}.parquet"] + [out / f"pred2_{g}_s{s}.parquet" for s in seeds[:k]]
        arrs[g] = _group_arrays(con, cfg, averaged(g, members) if k else members[0], g)
    rep = choose_decision(arrs["tune"], arrs["holdout"])
    rule = rep["params"]["rule"]
    res[f"models={k + 1}"] = {"rule": rule, "holdout": round(rep["holdout"][rule], 5), "all": rep["holdout"]}
    print(k + 1, "models:", res[f"models={k + 1}"]["holdout"], rule, flush=True)
(out / "ensemble.json").write_text(json.dumps(res, indent=1, default=float))
