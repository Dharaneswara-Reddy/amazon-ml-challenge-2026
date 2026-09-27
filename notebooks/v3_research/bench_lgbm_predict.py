"""LightGBM prediction throughput (question 2): booster.predict by num_threads, batch size, num_iteration,
using the REAL model1.txt (16MB, stage-1, 2358 trees) from the last full run, on a ~1M-row sample of test
features. Also checks whether lleaves (MIT, LLVM-compiled LightGBM inference, pip-installable) is available and,
if so, times it too. Kept to <=4 threads / small memory per the shared-laptop budget except one brief
thread-scaling check.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

RUN = "/home/toji339/Documents/Amazon_ML_Hackathon/artifacts/runs/20260925T161436_b538130_1242e7cd_v2"

sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber.features import FEATURES


def load_sample(path, n, seed=42):
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=min(n, total), replace=False))
    x = np.empty((len(idx), len(FEATURES)), dtype=np.float32)
    pos, row0 = 0, 0
    for batch in pf.iter_batches(batch_size=500_000, columns=FEATURES):
        m = batch.num_rows
        batch_idx = idx[(idx >= row0) & (idx < row0 + m)] - row0
        if len(batch_idx):
            for j, f in enumerate(FEATURES):
                x[pos:pos + len(batch_idx), j] = batch.column(f).to_numpy(zero_copy_only=False)[batch_idx]
            pos += len(batch_idx)
        row0 += m
    return x[:pos]


def timeit(fn, reps=1):
    t0 = time.time()
    for _ in range(reps):
        out = fn()
    return (time.time() - t0) / reps, out


if __name__ == "__main__":
    t0 = time.time()
    # pred1_test.parquet already has predictions, not features; fit_in.parquet has the real feature columns.
    x = load_sample(f"{RUN}/fit_in.parquet", 1_000_000, seed=99)
    print(f"# loaded x={x.shape} in {time.time()-t0:.1f}s", file=sys.stderr)

    booster = lgb.Booster(model_file=f"{RUN}/model1.txt")
    n_trees = booster.num_trees()
    print(f"# model1.txt: {n_trees} trees", file=sys.stderr)

    results = []

    # 1) threads sweep (brief, small reps) at full batch, all trees
    for th in (4, 8, 12):
        booster.params = booster.params or {}
        dt, _ = timeit(lambda: booster.predict(x, num_threads=th))
        rate = len(x) / dt
        row = {"variant": f"threads={th}", "time_s": round(dt, 2), "rows_per_s": int(rate)}
        print(json.dumps(row)); results.append(row)

    # 2) num_iteration (fewer trees) at threads=4
    for frac in (1.0, 0.5, 0.25, 0.1):
        ni = max(1, int(n_trees * frac))
        dt, _ = timeit(lambda: booster.predict(x, num_threads=4, num_iteration=ni))
        row = {"variant": f"num_iteration={ni} ({frac*100:.0f}% trees)", "time_s": round(dt, 2), "rows_per_s": int(len(x) / dt)}
        print(json.dumps(row)); results.append(row)

    # 3) pred_early_stop
    for margin in (None, 0.1, 0.05):
        kw = {} if margin is None else {"pred_early_stop": True, "pred_early_stop_freq": 10, "pred_early_stop_margin": margin}
        dt, _ = timeit(lambda: booster.predict(x, num_threads=4, **kw))
        row = {"variant": f"pred_early_stop_margin={margin}", "time_s": round(dt, 2), "rows_per_s": int(len(x) / dt)}
        print(json.dumps(row)); results.append(row)

    # 4) batch size effect (predict in chunks vs all at once) at threads=4, all trees
    for bs in (50_000, 200_000, 1_000_000):
        def chunked():
            for i in range(0, len(x), bs):
                booster.predict(x[i:i + bs], num_threads=4)
        dt, _ = timeit(chunked)
        row = {"variant": f"batch_size={bs}", "time_s": round(dt, 2), "rows_per_s": int(len(x) / dt)}
        print(json.dumps(row)); results.append(row)

    with open("/home/toji339/Documents/Amazon_ML_Hackathon/notebooks/v3_research/lgbm_predict_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"# total wall time {time.time()-t0:.1f}s", file=sys.stderr)
