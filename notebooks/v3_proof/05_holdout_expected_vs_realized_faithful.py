"""Faithful (production rival-aware) holdout expected-F vs realized-F, split by country (India/US). Supersedes the
buggy per-country holdout section of 02_expected_vs_realized.py (see that file's docstring note and Finding log).

Uses the same ber.pipeline._group_arrays as fit/compare_decisions, then the same calibrate()/prepare()/choose_k()
path decide() uses internally (mirrored here only to also get the per-entity expected-F array, which decide()
does not return). Read-only w.r.t. code/business_entity_resolution/.

Run: .venv/bin/python notebooks/v3_proof/05_holdout_expected_vs_realized_faithful.py
"""
import json
from pathlib import Path

import numpy as np

from ber import io
from ber.config import load_config
from ber.decision import Calibrator, MissingModel, best_per_right, calibrate, choose_k, mask_from_k, per_entity_f05, prepare
from ber.duck import connect
from ber.pipeline import _group_arrays
from ber.split import group_where

RUN = Path("artifacts/runs/20260925T161436_b538130_1242e7cd_v2")
OUT = Path("notebooks/v3_proof/out")


def main() -> None:
    """Compute per-country holdout mean expected-F (at k*) vs mean realized F0.5; dump JSON."""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    con = connect(cfg)
    con.execute("SET memory_limit='1200MB'")
    con.execute("SET threads=2")
    con.execute("SET temp_directory='artifacts/v3_proof_tmp'")

    params = json.loads((RUN / "decision_params.json").read_text())
    miss = MissingModel.from_dict(params["missing"])
    cal_empty = Calibrator.from_dict(params["cal_empty"])
    empty_weight = float(params["empty_weight"])
    shift = float(params.get("shift", 0.0))

    g = _group_arrays(con, cfg, RUN / "pred2_holdout.parquet", "holdout")
    lid, rid, p, label, n_true = g["lid_idx"], g["rid"], g["p"], g["label"], g["n_true"]
    is_best = best_per_right(rid, lid, p)
    p_cal = calibrate(params, is_best, p, shift=shift)
    prep = prepare(lid, rid, p_cal, max_k=int(params["max_k"]), n=len(n_true))
    k_star, ef = choose_k(prep, p_cal, miss, empty_weight, cal_empty)
    sel = mask_from_k(prep, k_star)
    real = per_entity_f05(lid, sel, label, n_true)

    s1_path = str(io.raw_parquet_path(cfg, "train", 1))
    rows = con.execute(
        f"""SELECT s.country, row_number() OVER (ORDER BY s.entity_id) - 1 AS idx
            FROM read_parquet('{s1_path}') s WHERE {group_where('s.entity_id', 'holdout')} ORDER BY idx"""
    ).fetchnumpy()
    country_arr = np.asarray(rows["country"])
    assert len(country_arr) == len(n_true)
    has = np.bincount(lid, minlength=len(n_true)) > 0

    out = {
        "pooled": {
            "n_entities": int(len(n_true)),
            "mean_expected_f_haswithcand": float(ef[has].mean()),
            "mean_realized_f_haswithcand": float(real[has].mean()),
            "macro_f05_all": float(real.mean()),
        }
    }
    for c in ("India", "US"):
        m = country_arr == c
        mh = m & has
        out[c] = {
            "n_entities": int(m.sum()),
            "mean_expected_f_haswithcand": float(ef[mh].mean()) if mh.any() else None,
            "mean_realized_f_haswithcand": float(real[mh].mean()) if mh.any() else None,
            "macro_f05_all": float(real[m].mean()),
            "gap_expected_minus_realized": float(ef[mh].mean() - real[mh].mean()) if mh.any() else None,
        }
    print(json.dumps(out, indent=2))
    (OUT / "05_holdout_expected_vs_realized_faithful.json").write_text(json.dumps(out, indent=2))
    con.close()


if __name__ == "__main__":
    main()
