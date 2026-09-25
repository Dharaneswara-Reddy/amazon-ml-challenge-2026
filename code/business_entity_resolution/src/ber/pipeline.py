"""End-to-end pipeline stages. Every stage reads and writes files under paths.cache_dir / a run dir, so stages can be
re-run independently and the whole chain is reproducible from the raw TSVs.

    prepare → normalize → block → features → train (fit / tune threshold / score holdout) → predict → write outputs
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import lightgbm as lgb
import numpy as np

from ber import io
from ber.blocking import BlockingConfig, recall_by_pass, register_inputs, run_blocking
from ber.config import Config
from ber.decision import macro_f05, select, tune_threshold
from ber.duck import connect
from ber.features import FEATURE_SET_VERSION, FEATURES, write_features
from ber.manifest import new_run_dir, write_manifest
from ber.model import predict_file, train
from ber.obs import kv, stage
from ber.split import group_where
from ber.writer import write_outputs

log = logging.getLogger(__name__)

TRAIN_GROUPS = ("fit", "tune", "holdout")


def norm_path(cfg: Config, split: str, source: int) -> Path:
    """Path of the normalized parquet for one split/source."""
    return io.cache_dir(cfg) / f"norm_{split}_s{source}.parquet"


def cand_path(cfg: Config, split: str) -> Path:
    """Path of the blocking candidates parquet for a split."""
    return io.cache_dir(cfg) / f"cand_{split}.parquet"


def feat_path(cfg: Config, split: str, group: str | None = None) -> Path:
    """Path of a feature parquet (train: one per group; test: one file)."""
    suffix = f"_{group}" if group else ""
    return io.cache_dir(cfg) / f"feat_{FEATURE_SET_VERSION}_{split}{suffix}.parquet"


def blocking_config(cfg: Config) -> BlockingConfig:
    """Build the BlockingConfig from the `blocking` config section (defaults for missing keys)."""
    section = dict(cfg.get("blocking", {}))
    for key in ("sn_keys", "token_fields"):
        if key in section:
            section[key] = tuple(section[key])
    return BlockingConfig(**section)


def _views(con: duckdb.DuckDBPyConnection, cfg: Config, split: str) -> dict[str, int]:
    """Register views L (Source 1) and R (Sources 2+3) over normalized parquet for a split."""
    return register_inputs(
        con, str(norm_path(cfg, split, 1)), [str(norm_path(cfg, split, 2)), str(norm_path(cfg, split, 3))]
    )


def normalize_stage(cfg: Config, split: str, force: bool = False) -> dict[str, Any]:
    """Normalize the raw parquet of every source of a split into norm_{split}_s{n}.parquet."""
    from ber.normalize import normalize_parquet

    out: dict[str, Any] = {}
    for source in io.SOURCES:
        dst = norm_path(cfg, split, source)
        if dst.exists() and not force:
            out[dst.name] = "cached"
            continue
        with stage("normalize", split=split, source=source) as st:
            normalize_parquet(
                io.raw_parquet_path(cfg, split, source), dst, int(cfg.engine.workers), int(cfg.engine.batch_rows)
            )
            st["out"] = str(dst)
        out[dst.name] = "written"
    return out


def block_stage(cfg: Config, split: str) -> dict[str, Any]:
    """Run blocking for a split in a file-backed DuckDB (spills to disk) and save candidates parquet + stats."""
    bcfg = blocking_config(cfg)
    db = io.cache_dir(cfg) / f"block_{split}.duckdb"
    db.unlink(missing_ok=True)
    con = connect(cfg, db)
    try:
        with stage("block", split=split) as st:
            counts = _views(con, cfg, split)
            stats = run_blocking(con, bcfg)
            con.execute(f"COPY candidates TO '{cand_path(cfg, split)}' (FORMAT parquet, COMPRESSION zstd)")
            per_left = con.execute(
                """SELECT avg(n), quantile_cont(n, 0.5), quantile_cont(n, 0.99), max(n)
                   FROM (SELECT lid, count(*) n FROM candidates GROUP BY lid)"""
            ).fetchone()
            stats.update(counts)
            stats["cand_per_left_mean_p50_p99_max"] = per_left
            stats["left_without_candidates"] = (
                counts["n_left"] - con.execute("SELECT count(DISTINCT lid) FROM candidates").fetchone()[0]
            )
            stats["reduction_ratio"] = 1 - stats["final_pairs"] / (counts["n_left"] * counts["n_right"])
            if split == "train":
                gt_sql = f"SELECT s1_id, match_id FROM read_parquet('{io.gt_pairs_path(cfg)}')"
                stats["recall"] = recall_by_pass(con, gt_sql, ["tok", *bcfg.sn_keys])
            st.update(stats)
    finally:
        con.close()
        db.unlink(missing_ok=True)
    return stats


def features_stage(cfg: Config, split: str) -> dict[str, Any]:
    """Compute feature parquet(s): per train group (with labels) or one file for test."""
    con = connect(cfg)
    try:
        _views(con, cfg, split)
        con.execute(f"CREATE TABLE candidates AS SELECT * FROM read_parquet('{cand_path(cfg, split)}')")
        workers, rows = int(cfg.engine.workers), int(cfg.engine.batch_rows)
        out: dict[str, Any] = {}
        if split == "train":
            label_sql = f"SELECT s1_id, match_id FROM read_parquet('{io.gt_pairs_path(cfg)}')"
            for g in TRAIN_GROUPS:
                with stage("features", split=split, group=g) as st:
                    st["rows"] = write_features(
                        con, feat_path(cfg, split, g), label_sql, rows, workers, group_where("c.lid", g)
                    )
                out[g] = st["rows"]
        else:
            with stage("features", split=split) as st:
                st["rows"] = write_features(con, feat_path(cfg, split), None, rows, workers)
            out["test"] = st["rows"]
        return out
    finally:
        con.close()


def _group_arrays(con: duckdb.DuckDBPyConnection, cfg: Config, preds: Path, group: str) -> dict[str, np.ndarray]:
    """Arrays for the decision layer on one train group: every S1 in the group (incl. ones without candidates)."""
    s1 = str(io.raw_parquet_path(cfg, "train", 1))
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE s1g AS
            SELECT s.entity_id AS s1_id, row_number() OVER (ORDER BY s.entity_id) - 1 AS idx,
                   coalesce(g.n_matches, 0)::DOUBLE AS n_true
            FROM read_parquet('{s1}') s
            LEFT JOIN read_parquet('{io.gt_s1_path(cfg)}') g ON g.s1_id = s.entity_id
            WHERE {group_where("s.entity_id", group)}"""
    )
    n_true = con.execute("SELECT n_true FROM s1g ORDER BY idx").fetchnumpy()["n_true"]
    pairs = con.execute(
        f"""SELECT s.idx, p.rid, p.p, p.label::BOOLEAN AS label
            FROM read_parquet('{preds}') p JOIN s1g s ON s.s1_id = p.lid"""
    ).fetchnumpy()
    rid_codes = np.unique(np.asarray(pairs["rid"], dtype=object), return_inverse=True)[1].astype(np.int64)
    return {
        "lid_idx": pairs["idx"].astype(np.int64),
        "rid": rid_codes,
        "p": pairs["p"].astype(np.float64),
        "label": pairs["label"].astype(bool),
        "n_true": n_true,
    }


def train_stage(cfg: Config, name: str = "lgbm") -> Path:
    """Train on the fit group, tune the threshold on the tune group, score the holdout group; return the run dir."""
    run_dir = new_run_dir(cfg, name)
    seed, threads = int(cfg.seed), int(cfg.engine.duckdb_threads)
    params = dict(cfg.get("model", {}).get("params", {}))
    with stage("train", run_dir=run_dir) as st:
        booster = train(feat_path(cfg, "train", "fit"), feat_path(cfg, "train", "tune"), params, seed, threads)
        booster.save_model(str(run_dir / "model.txt"), num_iteration=booster.best_iteration)
        st["best_iteration"] = booster.best_iteration
    metrics: dict[str, Any] = {}
    con = connect(cfg)
    try:
        for g in ("tune", "holdout"):
            out = run_dir / f"pred_{g}.parquet"
            predict_file(booster, feat_path(cfg, "train", g), out, int(cfg.engine.batch_rows))
            arr = _group_arrays(con, cfg, out, g)
            if g == "tune":
                threshold, curve = tune_threshold(arr["lid_idx"], arr["rid"], arr["p"], arr["label"], arr["n_true"])
                metrics["threshold"] = threshold
                metrics["tune_f05"] = max(s for _, s in curve)
                (run_dir / "threshold_curve.json").write_text(json.dumps(curve))
            sel = select(arr["lid_idx"], arr["rid"], arr["p"], metrics["threshold"])
            metrics[f"{g}_f05"] = macro_f05(arr["lid_idx"], sel, arr["label"], arr["n_true"])
            metrics[f"{g}_n_s1"] = len(arr["n_true"])
            metrics[f"{g}_candidate_recall"] = float(arr["label"].sum() / max(arr["n_true"].sum(), 1))
    finally:
        con.close()
    importance = dict(zip(FEATURES, booster.feature_importance("gain").round(1).tolist(), strict=True))
    write_manifest(run_dir, cfg, metrics=metrics, feature_set=FEATURE_SET_VERSION, importance_gain=importance)
    log.info(kv(event="train_done", run_dir=run_dir, **metrics))
    return run_dir


def predict_stage(cfg: Config, run_dir: Path) -> dict[str, Any]:
    """Score test candidates with a trained run, apply its locked threshold, and write + verify both outputs."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    threshold = float(manifest["metrics"]["threshold"])
    booster = lgb.Booster(model_file=str(run_dir / "model.txt"))
    preds = run_dir / "pred_test.parquet"
    with stage("predict", run_dir=run_dir):
        predict_file(booster, feat_path(cfg, "test"), preds, int(cfg.engine.batch_rows))
    con = connect(cfg)
    try:
        con.execute(
            f"""CREATE TEMP TABLE matches AS
                SELECT lid AS s1_id, rid AS match_id FROM read_parquet('{preds}')
                WHERE p >= {threshold}
                QUALIFY row_number() OVER (PARTITION BY rid ORDER BY p DESC, lid) = 1"""
        )
        con.execute(
            f"""CREATE TEMP TABLE cands AS
                SELECT lid AS s1_id, rid AS match_id FROM read_parquet('{cand_path(cfg, "test")}')"""
        )
        s1_path = io.raw_parquet_path(cfg, "test", 1)
        con.execute(f"CREATE TEMP TABLE s1 AS SELECT entity_id AS s1_id FROM read_parquet('{s1_path}')")
        r2, r3 = io.raw_parquet_path(cfg, "test", 2), io.raw_parquet_path(cfg, "test", 3)
        con.execute(f"CREATE TEMP TABLE valid_right AS SELECT entity_id FROM read_parquet(['{r2}', '{r3}'])")
        with stage("write_outputs") as st:
            result = write_outputs(con, "s1", "cands", "matches", "valid_right", Path(cfg.paths.output_dir))
            st.update({k: v for k, v in result.items() if not isinstance(v, dict)})
    finally:
        con.close()
    (run_dir / "predict.json").write_text(json.dumps({"threshold": threshold, **result}, indent=2, default=str))
    return result


RULE_SCALE = 3.3  # upper bound of block_score (jw + jw + jaccard + 0.3)


def _rule_group_arrays(con: duckdb.DuckDBPyConnection, cfg: Config, group: str) -> dict[str, np.ndarray]:
    """Arrays for the rule decision on one train group: each right record's best candidate over ALL candidates."""
    s1 = str(io.raw_parquet_path(cfg, "train", 1))
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE s1g AS
            SELECT s.entity_id AS s1_id, row_number() OVER (ORDER BY s.entity_id) - 1 AS idx,
                   coalesce(g.n_matches, 0)::DOUBLE AS n_true
            FROM read_parquet('{s1}') s
            LEFT JOIN read_parquet('{io.gt_s1_path(cfg)}') g ON g.s1_id = s.entity_id
            WHERE {group_where("s.entity_id", group)}"""
    )
    n_true = con.execute("SELECT n_true FROM s1g ORDER BY idx").fetchnumpy()["n_true"]
    pairs = con.execute(
        f"""SELECT s.idx, least(b.block_score / {RULE_SCALE}, 1.0) AS p, (g.s1_id IS NOT NULL) AS label
            FROM best b JOIN s1g s ON s.s1_id = b.lid
            LEFT JOIN read_parquet('{io.gt_pairs_path(cfg)}') g ON g.s1_id = b.lid AND g.match_id = b.rid"""
    ).fetchnumpy()
    return {
        "lid_idx": pairs["idx"].astype(np.int64),
        "p": pairs["p"].astype(np.float64),
        "label": pairs["label"].astype(bool),
        "n_true": n_true,
    }


def rule_stage(cfg: Config) -> dict[str, Any]:
    """Rule baseline: each right record goes to its top block_score candidate if block_score clears a threshold.

    The threshold maximizes macro F0.5 on the train tune group, is reported on the holdout group, and is then
    applied to test candidates to write both submission files. No model is trained.
    """
    run_dir = new_run_dir(cfg, "rule")
    con = connect(cfg)
    metrics: dict[str, Any] = {}
    try:
        con.execute(
            f"""CREATE TEMP TABLE best AS SELECT lid, rid, block_score FROM read_parquet('{cand_path(cfg, "train")}')
                QUALIFY row_number() OVER (PARTITION BY rid ORDER BY block_score DESC, lid) = 1"""
        )
        grid = np.round(np.arange(0.30, 1.0, 0.005), 3)
        tune = _rule_group_arrays(con, cfg, "tune")
        curve = [(float(t), macro_f05(tune["lid_idx"], tune["p"] >= t, tune["label"], tune["n_true"])) for t in grid]
        threshold, metrics["tune_f05"] = max(curve, key=lambda x: (x[1], -x[0]))
        metrics["threshold"] = threshold
        hold = _rule_group_arrays(con, cfg, "holdout")
        metrics["holdout_f05"] = macro_f05(hold["lid_idx"], hold["p"] >= threshold, hold["label"], hold["n_true"])
        metrics["holdout_candidate_recall"] = float(hold["label"].sum() / max(hold["n_true"].sum(), 1))
        (run_dir / "threshold_curve.json").write_text(json.dumps(curve))
        con.execute(
            f"""CREATE TEMP TABLE matches AS
                SELECT lid AS s1_id, rid AS match_id FROM read_parquet('{cand_path(cfg, "test")}')
                WHERE least(block_score / {RULE_SCALE}, 1.0) >= {threshold}
                QUALIFY row_number() OVER (PARTITION BY rid ORDER BY block_score DESC, lid) = 1"""
        )
        con.execute(
            f"""CREATE TEMP TABLE cands AS
                SELECT lid AS s1_id, rid AS match_id FROM read_parquet('{cand_path(cfg, "test")}')"""
        )
        s1_path = io.raw_parquet_path(cfg, "test", 1)
        con.execute(f"CREATE TEMP TABLE s1 AS SELECT entity_id AS s1_id FROM read_parquet('{s1_path}')")
        r2, r3 = io.raw_parquet_path(cfg, "test", 2), io.raw_parquet_path(cfg, "test", 3)
        con.execute(f"CREATE TEMP TABLE valid_right AS SELECT entity_id FROM read_parquet(['{r2}', '{r3}'])")
        with stage("write_outputs") as st:
            result = write_outputs(con, "s1", "cands", "matches", "valid_right", Path(cfg.paths.output_dir))
            st.update({k: v for k, v in result.items() if not isinstance(v, dict)})
    finally:
        con.close()
    write_manifest(run_dir, cfg, method="rule", metrics=metrics)
    (run_dir / "predict.json").write_text(json.dumps({"threshold": metrics["threshold"], **result}, default=str))
    log.info(kv(event="rule_done", run_dir=run_dir, **metrics))
    return {"run_dir": str(run_dir), **metrics}
