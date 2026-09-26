"""Per-country: (a) holdout expected-F vs REALIZED F (labels known, India/US only) -- calibration sanity check;
(b) test expected-F under the production global shift vs under a per-country EM-implied shift (India/US/France).

This tells us whether the model's own expected F0.5 already "knows" France is worse (compare pre-shift expected F,
computed in 01_country_calibration.py, against holdout realized F), and how much of France's shortfall might be a
shift-magnitude mismatch (global -0.49 vs France's own EM-implied ~-0.32) rather than an intrinsic model/feature gap.

Writes notebooks/v3_proof/out/02_expected_vs_realized.json. Read-only w.r.t. code/business_entity_resolution/.
Run: .venv/bin/python notebooks/v3_proof/02_expected_vs_realized.py
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
    per_entity_f05,
    prepare,
)

RUN = Path("artifacts/runs/20260925T161436_b538130_1242e7cd_v2")
OUT = Path("notebooks/v3_proof/out")


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection under the shared-laptop resource limits."""
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'")
    con.execute("SET threads=2")
    con.execute("PRAGMA temp_directory='artifacts/v3_proof_tmp'")
    return con


def holdout_country_arrays(con: duckdb.DuckDBPyConnection, country: str) -> dict:
    """lid/rid (dense-ranked), raw p, label, and true n_true per entity for one country's holdout pairs."""
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE hc AS
            SELECT dense_rank() OVER (ORDER BY h.lid) - 1 AS li,
                   dense_rank() OVER (ORDER BY h.rid) - 1 AS rc,
                   h.p AS p, h.label AS label, h.lid AS lid
            FROM read_parquet('{RUN}/pred2_holdout.parquet') h
            JOIN read_parquet('artifacts/cache/raw_train_s1.parquet') l ON l.entity_id = h.lid
            WHERE l.country = '{country}'"""
    )
    d = con.execute("SELECT li, rc, p, label FROM hc ORDER BY li").fetchnumpy()
    n_ent = int(d["li"].max()) + 1 if len(d["li"]) else 0
    # true n_true per entity from ground truth (includes matches missed by blocking)
    nt = con.execute(
        """SELECT li, any_value(g.n_matches) AS n_matches
           FROM (SELECT DISTINCT li, lid FROM hc) x
           JOIN read_parquet('artifacts/cache/train_gt_s1.parquet') g ON g.s1_id = x.lid
           GROUP BY li ORDER BY li"""
    ).fetchnumpy()
    n_true = np.zeros(n_ent, dtype=np.float64)
    n_true[nt["li"].astype(np.int64)] = nt["n_matches"].astype(np.float64)
    return {
        "li": d["li"].astype(np.int64),
        "rc": d["rc"].astype(np.int64),
        "p": d["p"].astype(np.float64),
        "label": d["label"].astype(bool),
        "n_true": n_true,
        "n_ent": n_ent,
    }


def test_country_arrays(con: duckdb.DuckDBPyConnection, country: str) -> dict:
    """lid/rid (dense-ranked), raw p per entity for one country's test pairs; n_ent = total S1 in that country."""
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE tc AS
            SELECT dense_rank() OVER (ORDER BY p.lid) - 1 AS li,
                   dense_rank() OVER (ORDER BY p.rid) - 1 AS rc,
                   p.p AS p
            FROM read_parquet('{RUN}/pred2_test.parquet') p
            JOIN read_parquet('artifacts/cache/raw_test_s1.parquet') l ON l.entity_id = p.lid
            WHERE l.country = '{country}'"""
    )
    d = con.execute("SELECT li, rc, p FROM tc").fetchnumpy()
    n_ent = con.execute(
        f"SELECT count(*) FROM read_parquet('artifacts/cache/raw_test_s1.parquet') WHERE country = '{country}'"
    ).fetchone()[0]
    return {"li": d["li"].astype(np.int64), "rc": d["rc"].astype(np.int64), "p": d["p"].astype(np.float64), "n_ent": n_ent}


def logit(x: float) -> float:
    """log(x / (1 - x))."""
    return float(np.log(x / (1 - x)))


def main() -> None:
    """Compute holdout expected-vs-realized F (India/US) and test expected F under global vs per-country shift."""
    OUT.mkdir(parents=True, exist_ok=True)
    params = json.loads((RUN / "decision_params.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    train_prior = float(manifest["metrics"]["train_prior_best"])
    miss = MissingModel.from_dict(params["missing"])
    cal_empty = Calibrator.from_dict(params["cal_empty"])
    empty_weight = float(params["empty_weight"])
    global_shift = -0.49344231087999096  # production logit_delta from predict.json (params['shift'] itself is 0.0)

    con = connect()
    report: dict = {"train_prior": train_prior, "global_shift": global_shift}

    # --- holdout: expected vs realized (production params, shift=0, labelled) ---
    report["holdout"] = {}
    for country in ("India", "US"):
        a = holdout_country_arrays(con, country)
        is_best = best_per_right(a["rc"], a["li"], a["p"])
        p_cal = calibrate(params, is_best, a["p"], shift=0.0)
        prep = prepare(a["li"], a["rc"], p_cal, max_k=16, n=a["n_ent"])
        k_star, ef = choose_k(prep, p_cal, miss, empty_weight, cal_empty)
        from ber.decision import mask_from_k

        sel = mask_from_k(prep, k_star)
        real = per_entity_f05(a["li"], sel, a["label"], a["n_true"])
        has = np.bincount(a["li"], minlength=a["n_ent"]) > 0
        report["holdout"][country] = {
            "n_entities": int(a["n_ent"]),
            "macro_f05_realized": float(real.mean()),
            "mean_expected_f_haswithcand": float(ef[has].mean()) if has.any() else None,
            "mean_realized_f_haswithcand": float(real[has].mean()) if has.any() else None,
            "gap_expected_minus_realized_haswithcand": float(ef[has].mean() - real[has].mean()) if has.any() else None,
        }
        print(country, "holdout", report["holdout"][country])

    # --- test: expected F under global shift vs per-country EM-implied shift ---
    report["test"] = {}
    for country in ("India", "US", "France"):
        a = test_country_arrays(con, country)
        is_best = best_per_right(a["rc"], a["li"], a["p"])
        p_cal0 = calibrate(params, is_best, a["p"], shift=0.0)
        prior_em, _ = em_prior_shift(p_cal0[is_best], train_prior)
        own_delta = logit(prior_em) - logit(train_prior)
        entry = {"prior_em_no_shift": prior_em, "own_implied_delta": own_delta}
        for tag, delta in (("global_shift", global_shift), ("own_shift", own_delta)):
            p_cal = calibrate(params, is_best, a["p"], shift=delta)
            prep = prepare(a["li"], a["rc"], p_cal, max_k=16, n=a["n_ent"])
            k_star, ef = choose_k(prep, p_cal, miss, empty_weight, cal_empty)
            has = np.bincount(a["li"], minlength=a["n_ent"]) > 0
            n_empty = int(np.sum((k_star == 0) & has)) + int(a["n_ent"] - has.sum())
            entry[tag] = {
                "delta": delta,
                "mean_expected_f_all_entities": float(ef.mean()),
                "mean_kstar_haswithcand": float(k_star[has].mean()) if has.any() else None,
                "share_empty_all_entities": float(n_empty / a["n_ent"]) if a["n_ent"] else None,
                "mean_matches_per_entity": float(k_star.sum() / a["n_ent"]) if a["n_ent"] else None,
            }
        report["test"][country] = entry
        print(country, "test", entry)

    (OUT / "02_expected_vs_realized.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT / "02_expected_vs_realized.json")


if __name__ == "__main__":
    main()
