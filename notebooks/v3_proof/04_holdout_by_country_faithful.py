"""Faithful reproduction of the production holdout decision (same _group_arrays rival-handling as fit/compare_decisions
in ber.pipeline), then split the realized per-entity F0.5 by country (India/US; holdout has no France). This
replaces the naive per-country holdout check in 02_expected_vs_realized.py, whose from-scratch is_best computation
on a country-only pair subset ignored cross-split rivals and gave a bogus ~0.40 realized F0.5 (vs the known
pooled local holdout 0.9715) -- that was a bug in the ad hoc script, not a real finding; documented and superseded
in research_notes/v3/G_public_gap_france.md.

Read-only w.r.t. code/business_entity_resolution/ (only imports it). Run:
  .venv/bin/python notebooks/v3_proof/04_holdout_by_country_faithful.py
"""
import json
from pathlib import Path

import numpy as np

from ber import io
from ber.config import load_config
from ber.decision import decide, per_entity_f05
from ber.duck import connect
from ber.pipeline import _group_arrays

RUN = Path("artifacts/runs/20260925T161436_b538130_1242e7cd_v2")
OUT = Path("notebooks/v3_proof/out")


def main() -> None:
    """Recompute holdout realized macro F0.5 with production rival-handling, split by country; dump JSON."""
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    con = connect(cfg)
    con.execute("SET memory_limit='1200MB'")
    con.execute("SET threads=2")
    con.execute("SET temp_directory='artifacts/v3_proof_tmp'")

    params = json.loads((RUN / "decision_params.json").read_text())

    g = _group_arrays(con, cfg, RUN / "pred2_holdout.parquet", "holdout")
    sel = decide(params, g["lid_idx"], g["rid"], g["p"])
    # decide() ignores label; need a label-aware selection too, but per_entity_f05 needs label per pair which we have
    real = per_entity_f05(g["lid_idx"], sel, g["label"], g["n_true"])
    print("pooled holdout macro F0.5 (faithful):", float(real.mean()), "n_entities:", len(real))

    # map idx (0..n-1, ordered by s1_id ascending within holdout group) back to s1_id and country
    s1_path = str(io.raw_parquet_path(cfg, "train", 1))
    from ber.split import group_where

    rows = con.execute(
        f"""SELECT s.entity_id, s.country, row_number() OVER (ORDER BY s.entity_id) - 1 AS idx
            FROM read_parquet('{s1_path}') s WHERE {group_where('s.entity_id', 'holdout')} ORDER BY idx"""
    ).fetchnumpy()
    idx = rows["idx"].astype(np.int64)
    country = rows["country"]
    assert len(idx) == len(real), (len(idx), len(real))
    country_arr = np.asarray(country)

    out = {"pooled_macro_f05": float(real.mean()), "n_entities": int(len(real))}
    for c in ("India", "US"):
        m = country_arr == c
        out[c] = {"n_entities": int(m.sum()), "macro_f05": float(real[m].mean())}
    print(out)
    (OUT / "04_holdout_by_country_faithful.json").write_text(json.dumps(out, indent=2))
    con.close()


if __name__ == "__main__":
    main()
