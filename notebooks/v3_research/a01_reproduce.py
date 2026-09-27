"""Reproduce v2 holdout F0.5 and save an enriched per-pair decision table for error analysis.

Output: notebooks/v3_research/out/hold_pairs.parquet (all holdout-group pairs incl. dropped-to-rival ones)
        notebooks/v3_research/out/hold_ent.parquet (per holdout S1 entity)
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from ber.config import load_config  # noqa: E402
from ber.decision import (  # noqa: E402
    Calibrator, MissingModel, calibrate, choose_k, decide, macro_f05, mask_from_k, per_entity_f05, prepare,
)
from ber.pipeline import _group_arrays  # noqa: E402
from ber.split import group_where  # noqa: E402

RUN = ROOT / "artifacts/runs/20260925T161436_b538130_1242e7cd_v2"
OUT = ROOT / "notebooks/v3_research/out"
OUT.mkdir(exist_ok=True)

cfg = load_config(root=ROOT)
con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=4")
params = json.loads((RUN / "decision_params.json").read_text())

g = _group_arrays(con, cfg, RUN / "pred2_holdout.parquet", "holdout")
sel = decide(params, g["lid_idx"], g["rid"], g["p"])
print("holdout F0.5 via _group_arrays+decide:", macro_f05(g["lid_idx"], sel, g["label"], g["n_true"]))

# Enriched version with ids (same logic as _group_arrays, deterministic order)
C = ROOT / "artifacts/cache"
con.execute(f"""CREATE OR REPLACE TEMP TABLE s1g AS
  SELECT s.entity_id AS s1_id, row_number() OVER (ORDER BY s.entity_id) - 1 AS idx,
         coalesce(g.n_matches, 0)::DOUBLE AS n_true, s.country
  FROM read_parquet('{C}/raw_train_s1.parquet') s
  LEFT JOIN read_parquet('{C}/train_gt_s1.parquet') g ON g.s1_id = s.entity_id
  WHERE {group_where('s.entity_id', 'holdout')}""")
con.execute(f"""CREATE TEMP TABLE pr AS
  SELECT p2.lid, p2.rid, p2.p, p2.label::BOOLEAN AS label, p1.p AS p1
  FROM read_parquet('{RUN}/pred2_holdout.parquet') p2
  JOIN read_parquet('{RUN}/pred1_holdout.parquet') p1 USING (lid, rid)""")
con.execute("""CREATE TEMP TABLE best AS SELECT rid, arg_max(lid, p) AS best_lid, max(p) AS best_p,
               count(*) AS r_ncand FROM pr GROUP BY rid""")
con.execute("""CREATE TEMP TABLE hp AS
  SELECT s.idx, pr.lid, pr.rid, pr.p, pr.p1, pr.label, b.best_lid, b.r_ncand,
         (b.best_lid IN (SELECT s1_id FROM s1g)) AS kept
  FROM pr JOIN s1g s ON s.s1_id = pr.lid JOIN best b ON b.rid = pr.rid ORDER BY s.idx, pr.rid""")
d = con.execute("SELECT * FROM hp").fetchnumpy()
ent = con.execute("SELECT s1_id, idx, n_true, country FROM s1g ORDER BY idx").fetchnumpy()
n_true = ent["n_true"]
kept = d["kept"].astype(bool)
lid_idx = d["idx"].astype(np.int64)[kept]
rid_codes = np.unique(d["rid"][kept], return_inverse=True)[1]
p = d["p"].astype(np.float64)[kept]
label = d["label"].astype(bool)[kept]
prep = prepare(lid_idx, rid_codes, p, int(params["max_k"]))
pc = calibrate(params, prep.is_best, p)
k_star, ef = choose_k(prep, pc, MissingModel.from_dict(params["missing"]), float(params["empty_weight"]),
                      Calibrator.from_dict(params["cal_empty"]))
mask = mask_from_k(prep, k_star)
print("holdout F0.5 enriched:", macro_f05(lid_idx, mask, label, n_true))

n = len(d["idx"])
full = {k: np.zeros(n, dtype=bool) for k in ("sel", "is_best")}
pcal = np.full(n, np.nan)
kidx = np.flatnonzero(kept)
full["sel"][kidx] = mask
full["is_best"][kidx] = prep.is_best
pcal[kidx] = pc
tbl = pa.table({
    "idx": d["idx"].astype(np.int64), "lid": d["lid"], "rid": d["rid"], "p": d["p"].astype(np.float64),
    "p1": d["p1"].astype(np.float64), "pcal": pcal, "label": d["label"].astype(bool), "best_lid": d["best_lid"],
    "r_ncand": d["r_ncand"].astype(np.int64), "kept": kept, "is_best": full["is_best"], "sel": full["sel"],
})
pq.write_table(tbl, OUT / "hold_pairs.parquet")
f = per_entity_f05(lid_idx, mask, label, n_true)
pq.write_table(pa.table({"idx": ent["idx"].astype(np.int64), "s1_id": ent["s1_id"], "country": ent["country"],
                         "n_true": n_true, "k_star": k_star, "ef": ef, "f": f}), OUT / "hold_ent.parquet")
print("entities", len(n_true), "pairs", n, "kept", kept.sum())
