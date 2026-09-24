"""LightGBM pair classifier: load feature parquet, train with early stopping, predict in streamed batches.

LightGBM is MIT-licensed; models are saved in LightGBM's text format (no pickle).
"""

from __future__ import annotations

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber.features import FEATURES

log = logging.getLogger(__name__)

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def load_matrix(path: Path, with_label: bool = True) -> tuple[np.ndarray, np.ndarray | None, pa.Table]:
    """Read a feature parquet into a float32 matrix (FEATURES order), optional labels, and the id columns."""
    cols = ["lid", "rid", *FEATURES] + (["label"] if with_label else [])
    tbl = pq.read_table(path, columns=cols)
    x = np.column_stack([tbl.column(f).to_numpy(zero_copy_only=False).astype(np.float32) for f in FEATURES])
    y = tbl.column("label").to_numpy().astype(np.int8) if with_label else None
    return x, y, tbl.select(["lid", "rid"])


def train(fit_path: Path, valid_path: Path, params: dict, seed: int, threads: int, rounds: int = 2000) -> lgb.Booster:
    """Train a binary LightGBM on fit_path with early stopping on valid_path; return the booster at best iteration."""
    x_fit, y_fit, _ = load_matrix(fit_path)
    x_val, y_val, _ = load_matrix(valid_path)
    full = {**DEFAULT_PARAMS, **params, "seed": seed, "num_threads": threads}
    d_fit = lgb.Dataset(x_fit, y_fit, feature_name=FEATURES, free_raw_data=True)
    d_val = lgb.Dataset(x_val, y_val, reference=d_fit, free_raw_data=True)
    del x_fit, x_val
    booster = lgb.train(
        full,
        d_fit,
        num_boost_round=rounds,
        valid_sets=[d_val],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    log.info("trained lightgbm best_iteration=%d best_score=%s", booster.best_iteration, dict(booster.best_score))
    return booster


def predict_file(booster: lgb.Booster, feat_path: Path, out_path: Path, batch_rows: int) -> int:
    """Score every row of a feature parquet in batches and write (lid, rid, p[, label]) parquet; return rows."""
    pf = pq.ParquetFile(feat_path)
    has_label = "label" in pf.schema_arrow.names
    cols = ["lid", "rid", *FEATURES] + (["label"] if has_label else [])
    writer = None
    n = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
            x = np.column_stack([batch.column(f).to_numpy(zero_copy_only=False).astype(np.float32) for f in FEATURES])
            p = booster.predict(x, num_iteration=booster.best_iteration or None).astype(np.float32)
            arrays = [batch.column("lid"), batch.column("rid"), pa.array(p)]
            names = ["lid", "rid", "p"]
            if has_label:
                arrays.append(batch.column("label"))
                names.append("label")
            out = pa.RecordBatch.from_arrays(arrays, names=names)
            if writer is None:
                writer = pq.ParquetWriter(out_path, out.schema, compression="zstd")
            writer.write_batch(out)
            n += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    return n
