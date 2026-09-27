"""lleaves prediction speed (MIT license, LLVM-compiled LightGBM inference, https://github.com/siboehm/lleaves).
Run via `uv run --with lleaves --with lightgbm --with pyarrow --with numpy` -- a throwaway env, does NOT touch
the project .venv. Compares to plain lightgbm booster.predict on the same sample for a fair speedup number.
"""
import json
import sys
import time

import numpy as np
import pyarrow.parquet as pq

RUN = "/home/toji339/Documents/Amazon_ML_Hackathon/artifacts/runs/20260925T161436_b538130_1242e7cd_v2"
sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber.features import FEATURES  # noqa: E402


def load_sample(path, n, seed=99):
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


if __name__ == "__main__":
    import lightgbm as lgb
    x = load_sample(f"{RUN}/fit_in.parquet", 1_000_000)
    print(f"# x={x.shape}", file=sys.stderr)

    booster = lgb.Booster(model_file=f"{RUN}/model1.txt")
    t0 = time.time()
    p_lgb = booster.predict(x, num_threads=4)
    lgb_dt = time.time() - t0
    print(json.dumps({"variant": "lightgbm booster.predict threads=4", "time_s": round(lgb_dt, 2), "rows_per_s": int(len(x) / lgb_dt)}))

    import lleaves
    t0 = time.time()
    model = lleaves.Model(model_file=f"{RUN}/model1.txt")
    model.compile()
    compile_s = time.time() - t0
    print(f"# lleaves compile took {compile_s:.1f}s", file=sys.stderr)

    t0 = time.time()
    p_ll = model.predict(x, n_jobs=4)
    ll_dt = time.time() - t0
    print(json.dumps({"variant": "lleaves n_jobs=4", "time_s": round(ll_dt, 2), "rows_per_s": int(len(x) / ll_dt), "compile_s": round(compile_s, 1)}))
    print(json.dumps({"speedup_vs_lightgbm": round(lgb_dt / ll_dt, 2), "max_abs_diff": float(np.max(np.abs(p_lgb - p_ll)))}))
