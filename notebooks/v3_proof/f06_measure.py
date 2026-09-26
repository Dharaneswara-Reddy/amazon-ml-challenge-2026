"""Measure the generator-signature features on holdout, on top of the v2 stage-2 probability.

Correction model = LightGBM on [logit(p2)] + a feature set, fitted on TUNE pairs with p2 >= BAND (other pairs keep
p2). Tune predictions are cross-fitted (2 folds by md5(lid)); the holdout model is fitted on all tune pairs. The
decision rule (fit_decision, expected-F0.5) is refitted on tune with the corrected p, then applied once to holdout
(_group_arrays semantics: rivals outside the group dropped). Baseline = raw p2 through the same decision refit.
Reports macro F0.5, FP / TP / FN, per country, and which of A's FP / decision-FN categories are fixed.
usage: python f06_measure.py [set ...]   (sets: base num name full)
"""
import sys

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from f_common import C, OUT, RUN, connect
from ber.decision import decide, fit_decision, per_entity_f05
from ber.split import group_where
from f05_feats import COLS

BAND = 1e-3
NUM = ["num_nl", "num_nr", "num_eq", "num_l_only", "num_r_only", "num_changed", "off_in_S", "off_p12", "off_m12",
       "off_other", "num_trunc", "num_add_digit", "num_transpose", "first_off", "first_off_idx", "num_pad", "sole_off_in_S"]
V2 = [c for c in pq.read_schema(C / "feat_v2_train_tune.parquet").names if c not in ("lid", "rid", "in_group", "label")]
V2A = [f"v2_{c}" for c in V2]
SETS = {"num": NUM, "name": [c for c in COLS if c not in NUM], "full": COLS, "v2only": V2A, "fullv2": V2A + COLS,
        "numv2": V2A + NUM}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42, num_threads=4)
ROUNDS = 400
con = connect()


def load(split: str, cols: list[str]) -> dict:
    """Band rows of f05_feat_<split>: fold, p, label and the requested feature columns (float32)."""
    sel = ", ".join(f"v.{c[3:]}::FLOAT AS {c}" if c.startswith("v2_") else f"f.{c}::FLOAT AS {c}" for c in cols)
    join = (f"LEFT JOIN '{C}/feat_v2_train_{split}.parquet' v ON v.lid = f.lid AND v.rid = f.rid"
            if any(c.startswith("v2_") for c in cols) else "")
    return con.execute(f"""SELECT (md5_number(f.lid) % 2)::INT AS fold, f.p::DOUBLE AS p, f.label::INT AS label{', ' + sel if cols else ''}
                           FROM '{OUT}/f05_feat_{split}.parquet' f {join} WHERE f.p >= {BAND} ORDER BY f.lid, f.rid""").fetchnumpy()


def xmat(d: dict, cols: list[str]) -> np.ndarray:
    """Feature matrix [logit(p2), cols...] as float32."""
    p = np.clip(d["p"], 1e-7, 1 - 1e-7)
    return np.column_stack([np.log(p / (1 - p)).astype(np.float32)] + [d[c] for c in cols])


def corrected(tag: str, cols: list[str]) -> None:
    """Write out/f06_p_<tag>_{tune,holdout}.parquet (lid, rid, p, label) with the corrected p."""
    tr = load("tune", cols)
    y, fold = tr["label"], tr["fold"]
    X = xmat(tr, cols)
    del tr
    ds = lgb.Dataset(X, y, free_raw_data=False, params={"verbose": -1})
    ds.construct()
    pt = np.zeros(len(y))
    for k in (0, 1):
        m = lgb.train(PARAMS, ds.subset(np.flatnonzero(fold != k)), ROUNDS)
        for idx in np.array_split(np.flatnonzero(fold == k), 4):
            pt[idx] = m.predict(X[idx])
    m = lgb.train(PARAMS, ds, ROUNDS)
    del ds
    gain = dict(zip(["logit_p2"] + cols, m.feature_importance("gain")))
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:15]
    print(tag, "top gain:", [(k, round(v / sum(gain.values()), 3)) for k, v in top], flush=True)
    del X
    ho = load("holdout", cols)
    ph = m.predict(xmat(ho, cols))
    del ho
    for split, pv in (("tune", pt), ("holdout", ph)):
        base = con.execute(f"SELECT lid, rid, p, label FROM '{RUN}/pred2_{split}.parquet' ORDER BY lid, rid").fetch_arrow_table()
        p = base.column("p").to_numpy().astype(np.float64)
        band = p >= BAND
        assert band.sum() == len(pv)
        p[band] = pv
        pq.write_table(pa.table({"lid": base.column("lid"), "rid": base.column("rid"), "p": p.astype(np.float32),
                                 "label": base.column("label")}), OUT / f"f06_p_{tag}_{split}.parquet")


def group_arrays(preds, group: str) -> dict:
    """pipeline._group_arrays plus lid / rid strings and the S1 country (for per-country scores and FP lists)."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE s1g AS
        SELECT s.entity_id AS s1_id, s.country, row_number() OVER (ORDER BY s.entity_id) - 1 AS idx,
               coalesce(g.n_matches, 0)::DOUBLE AS n_true
        FROM read_parquet('{C}/raw_train_s1.parquet') s LEFT JOIN read_parquet('{C}/train_gt_s1.parquet') g ON g.s1_id = s.entity_id
        WHERE {group_where('s.entity_id', group)}""")
    ent = con.execute("SELECT n_true, country FROM s1g ORDER BY idx").fetchnumpy()
    pr = con.execute(f"""WITH pr AS (SELECT lid, rid, p, label FROM read_parquet('{preds}')),
             best AS (SELECT rid, arg_max(lid, p) AS best_lid FROM pr GROUP BY rid)
        SELECT s.idx, dense_rank() OVER (ORDER BY pr.rid) - 1 AS rc, pr.p::DOUBLE p, pr.label::BOOLEAN AS label, pr.lid, pr.rid
        FROM pr JOIN s1g s ON s.s1_id = pr.lid JOIN best b ON b.rid = pr.rid
        WHERE b.best_lid IN (SELECT s1_id FROM s1g) ORDER BY s.idx, rc""").fetchnumpy()
    return {"lid_idx": pr["idx"].astype(np.int64), "rid": pr["rc"].astype(np.int64), "p": pr["p"],
            "label": pr["label"].astype(bool), "n_true": ent["n_true"], "country": np.asarray(ent["country"], dtype=object),
            "lid_s": pr["lid"], "rid_s": pr["rid"]}


def score(tag: str, ptune, phold) -> set:
    """Refit the decision on tune, score holdout; print summary; return the set of selected holdout (lid, rid)."""
    gt = group_arrays(ptune, "tune")
    params = fit_decision({k: gt[k] for k in ("lid_idx", "rid", "p", "label", "n_true")})
    del gt
    gh = group_arrays(phold, "holdout")
    sel = decide(params, gh["lid_idx"], gh["rid"], gh["p"])
    f = per_entity_f05(gh["lid_idx"], sel, gh["label"], gh["n_true"])
    lab = gh["label"]
    tp, fp = int((sel & lab).sum()), int((sel & ~lab).sum())
    fn_dec = int((~sel & lab).sum())
    by = {c: round(float(f[gh["country"] == c].mean()), 6) for c in ("India", "US")}
    print(f"RESULT {tag}: holdout macroF0.5 {f.mean():.6f} TP {tp} FP {fp} in-cand FN {fn_dec} per-country {by} "
          f"shift {params['shift']} ew {params['empty_weight']}", flush=True)
    return set(zip(gh["lid_s"][sel].tolist(), gh["rid_s"][sel].tolist()))


if __name__ == "__main__":
    tags = sys.argv[1:] or ["base", "num", "full"]
    res = {}
    for tag in tags:
        if tag == "base":
            src = {s: RUN / f"pred2_{s}.parquet" for s in ("tune", "holdout")}
        else:
            corrected(tag, SETS[tag])
            src = {s: OUT / f"f06_p_{tag}_{s}.parquet" for s in ("tune", "holdout")}
        res[tag] = score(tag, src["tune"], src["holdout"])
        pq.write_table(pa.table({"lid": [a for a, _ in res[tag]], "rid": [b for _, b in res[tag]]}),
                       OUT / f"f06_sel_{tag}.parquet")
