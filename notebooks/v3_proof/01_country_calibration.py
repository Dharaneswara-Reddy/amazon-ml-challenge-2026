"""Per-country calibration/prior/expected-F analysis for test, plus train-holdout baseline (India/US only).

Reproduces the production calibration path (ber.decision.calibrate / em_prior_shift / prepare / choose_k) but
per country instead of pooled, to see whether France needs a different correction than India/US, and whether the
model's own expected F0.5 already signals it is worse there. Processes one country at a time to keep Python RSS
low (DuckDB does the filtering/joins; only per-country arrays reach numpy, at most ~14M rows for India).

Writes notebooks/v3_proof/out/01_country_calibration.json. Read-only w.r.t. code/business_entity_resolution/.
Run: .venv/bin/python notebooks/v3_proof/01_country_calibration.py
"""
import json
from pathlib import Path

import duckdb
import numpy as np

from ber.decision import (
    Calibrator,
    MissingModel,
    best_per_right,
    calibrate,
    choose_k,
    em_prior_shift,
    entity_stats,
    prepare,
)

RUN = Path("artifacts/runs/20260925T161436_b538130_1242e7cd_v2")
OUT = Path("notebooks/v3_proof/out")


def connect() -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection under the laptop's shared-resource limits."""
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'")
    con.execute("SET threads=2")
    con.execute("PRAGMA temp_directory='artifacts/v3_proof_tmp'")
    return con


def fetch_country_pairs(con: duckdb.DuckDBPyConnection, pred_path: str, s1_path: str, country: str) -> dict:
    """lid (dense-ranked int), rid (dense-ranked int), raw p (float64) for one country's candidate pairs."""
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE cc AS
            SELECT dense_rank() OVER (ORDER BY p.lid) - 1 AS li,
                   dense_rank() OVER (ORDER BY p.rid) - 1 AS rc,
                   p.p AS p
            FROM read_parquet('{pred_path}') p
            JOIN read_parquet('{s1_path}') l ON l.entity_id = p.lid
            WHERE l.country = '{country}'"""
    )
    d = con.execute("SELECT li, rc, p FROM cc").fetchnumpy()
    return {"li": d["li"].astype(np.int64), "rc": d["rc"].astype(np.int64), "p": d["p"].astype(np.float64)}


def country_report(params: dict, train_prior: float, arrs: dict, n_entities: int) -> dict:
    """Calibration/prior/expected-F summary for one country's candidate-pair arrays (production calibrator, no
    logit shift applied yet -- shift is analyzed separately)."""
    li, rc, p = arrs["li"], arrs["rc"], arrs["p"]
    is_best = best_per_right(rc, li, p)
    p_cal = calibrate(params, is_best, p, shift=0.0)  # production isotonic calibration, pre-shift
    best_p = p_cal[is_best]
    prior_em, _ = em_prior_shift(best_p, train_prior)
    uncertain = float(np.mean((p_cal > 0.05) & (p_cal < 0.95)))
    low = float(np.mean(best_p < 0.1))  # distractor proxy: best-per-right candidate still looks like a non-match
    prep = prepare(li, rc, p_cal, max_k=16, n=n_entities)
    miss = MissingModel.from_dict(params["missing"])
    cal_empty = Calibrator.from_dict(params["cal_empty"])
    k_star, ef = choose_k(prep, p_cal, miss, float(params["empty_weight"]), cal_empty)
    sum_p, _, _ = entity_stats(li, p_cal, n_entities)
    return {
        "n_pairs": int(len(p)),
        "n_entities_with_cand": int(len(np.unique(li))),
        "n_entities_total": int(n_entities),
        "n_best_per_right": int(is_best.sum()),
        "mean_best_p_cal": float(best_p.mean()),
        "median_best_p_cal": float(np.median(best_p)),
        "share_uncertain_pairs_0.05_0.95": uncertain,
        "share_best_p_lt_0.10": low,
        "prior_em_no_shift": prior_em,
        "mean_expected_f_at_kstar": float(ef.mean()),
        "share_kstar_0": float(np.mean(k_star == 0)),
        "mean_kstar": float(k_star.mean()),
        "mean_sum_p_per_entity_with_cand": float(sum_p[sum_p > 0].mean()) if (sum_p > 0).any() else float("nan"),
    }


def main() -> None:
    """Compute per-country reports for test (India/US/France) and train holdout (India/US only); dump JSON."""
    OUT.mkdir(parents=True, exist_ok=True)
    params = json.loads((RUN / "decision_params.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    train_prior = float(manifest["metrics"]["train_prior_best"])
    con = connect()

    report: dict = {"train_prior_best": train_prior, "params_shift": params.get("shift", 0.0)}

    # --- test, per country ---
    n_by_country = dict(
        con.execute("SELECT country, count(*) FROM read_parquet('artifacts/cache/raw_test_s1.parquet') GROUP BY 1").fetchall()
    )
    report["test"] = {}
    for country in ("India", "US", "France"):
        arrs = fetch_country_pairs(con, str(RUN / "pred2_test.parquet"), "artifacts/cache/raw_test_s1.parquet", country)
        report["test"][country] = country_report(params, train_prior, arrs, n_by_country[country])
        print(country, "test done", report["test"][country])

    # --- train holdout (India/US only; has labels but we ignore labels here, same pre-shift calibration lens) ---
    n_by_country_train = dict(
        con.execute(
            "SELECT country, count(*) FROM read_parquet('artifacts/cache/raw_train_s1.parquet') GROUP BY 1"
        ).fetchall()
    )
    report["holdout_train_country_totals_all_s1"] = n_by_country_train  # NB: holdout is a split of this, not all of it
    report["holdout"] = {}
    for country in ("India", "US"):
        con.execute(
            f"""CREATE OR REPLACE TEMP TABLE hc AS
                SELECT dense_rank() OVER (ORDER BY p.lid) - 1 AS li,
                       dense_rank() OVER (ORDER BY p.rid) - 1 AS rc,
                       p.p AS p
                FROM read_parquet('{RUN}/pred2_holdout.parquet') p
                JOIN read_parquet('artifacts/cache/raw_train_s1.parquet') l ON l.entity_id = p.lid
                WHERE l.country = '{country}'"""
        )
        d = con.execute("SELECT li, rc, p FROM hc").fetchnumpy()
        arrs = {"li": d["li"].astype(np.int64), "rc": d["rc"].astype(np.int64), "p": d["p"].astype(np.float64)}
        n_ent = int(arrs["li"].max()) + 1 if len(arrs["li"]) else 0
        report["holdout"][country] = country_report(params, train_prior, arrs, n_ent)
        print(country, "holdout done", report["holdout"][country])

    (OUT / "01_country_calibration.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT / "01_country_calibration.json")


if __name__ == "__main__":
    main()
