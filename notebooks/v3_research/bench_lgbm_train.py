"""LightGBM training speed vs accuracy on a SAMPLE of the real stage-1 features (question 1 of the v3 speed
research). Loads a 2M-row sample of fit_in.parquet (9.29M rows full) and a 300k-row sample of tune_in.parquet
(1.55M rows full) from the last full run, trains with varying params, reports time + best logloss + best_iteration.

Kept small per the shared-laptop research budget: samples (not full 9.3M rows), early-stopping patience 30,
max 500 rounds (real run needed up to 2358 at lr=0.1 on the full 9.3M rows -- we're not chasing full convergence,
just relative time/accuracy tradeoffs between configs).
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber.features import FEATURES

RUN = "/home/toji339/Documents/Amazon_ML_Hackathon/artifacts/runs/20260925T161436_b538130_1242e7cd_v2"


def load_sample(path, n, seed=42):
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(total, size=min(n, total), replace=False))
    cols = [*FEATURES, "label"]
    x = np.empty((len(idx), len(FEATURES)), dtype=np.float32)
    y = np.empty(len(idx), dtype=np.int8)
    pos = 0
    row0 = 0
    take_ptr = 0
    for batch in pf.iter_batches(batch_size=500_000, columns=cols):
        m = batch.num_rows
        batch_idx = idx[(idx >= row0) & (idx < row0 + m)] - row0
        if len(batch_idx):
            for j, f in enumerate(FEATURES):
                x[pos:pos + len(batch_idx), j] = batch.column(f).to_numpy(zero_copy_only=False)[batch_idx]
            y[pos:pos + len(batch_idx)] = batch.column("label").to_numpy(zero_copy_only=False)[batch_idx]
            pos += len(batch_idx)
        row0 += m
    return x[:pos], y[:pos]


def run(params, x_fit, y_fit, x_val, y_val, threads, rounds=300, patience=20, label=""):
    full = {
        "objective": "binary", "deterministic": True, "verbose": -1, "seed": 42,
        "num_threads": threads, **params,
    }
    d_fit = lgb.Dataset(x_fit, y_fit, feature_name=list(FEATURES), free_raw_data=True)
    d_val = lgb.Dataset(x_val, y_val, reference=d_fit, free_raw_data=True)
    t0 = time.time()
    booster = lgb.train(
        full, d_fit, num_boost_round=rounds, valid_sets=[d_val], valid_names=["valid"],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )
    dt = time.time() - t0
    best = booster.best_score["valid"]["binary_logloss"]
    row = {
        "label": label, "time_s": round(dt, 1), "best_iter": booster.best_iteration,
        "logloss": round(float(best), 5), "s_per_tree": round(dt / max(booster.best_iteration, 1), 4),
        "threads": threads, **params,
    }
    print(json.dumps(row))
    return row


if __name__ == "__main__":
    t0 = time.time()
    x_fit, y_fit = load_sample(f"{RUN}/fit_in.parquet", 1_200_000)
    x_val, y_val = load_sample(f"{RUN}/tune_in.parquet", 200_000, seed=7)
    print(f"# loaded fit={x_fit.shape} val={x_val.shape} pos_rate_fit={y_fit.mean():.4f} in {time.time()-t0:.1f}s", file=sys.stderr)

    base = dict(learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, force_row_wise=True, max_bin=255)

    results = []
    # baseline (current pipeline defaults, minus rounds cap)
    results.append(run(base, x_fit, y_fit, x_val, y_val, threads=4, label="baseline_lr0.1_leaves63"))
    # learning rate sweep
    for lr in (0.2, 0.3):
        results.append(run({**base, "learning_rate": lr}, x_fit, y_fit, x_val, y_val, threads=4, label=f"lr{lr}"))
    # num_leaves sweep
    for nl in (127, 255):
        results.append(run({**base, "num_leaves": nl}, x_fit, y_fit, x_val, y_val, threads=4, label=f"leaves{nl}"))
    # max_bin
    results.append(run({**base, "max_bin": 63}, x_fit, y_fit, x_val, y_val, threads=4, label="max_bin63"))
    # force_col_wise instead of row_wise
    results.append(run({**base, "force_row_wise": False, "force_col_wise": True}, x_fit, y_fit, x_val, y_val, threads=4, label="col_wise"))
    # GOSS
    results.append(run({**base, "boosting": "goss", "bagging_freq": 0, "bagging_fraction": 1.0}, x_fit, y_fit, x_val, y_val, threads=4, label="goss"))

    with open("/home/toji339/Documents/Amazon_ML_Hackathon/notebooks/v3_research/lgbm_train_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"# total wall time {time.time()-t0:.1f}s", file=sys.stderr)
