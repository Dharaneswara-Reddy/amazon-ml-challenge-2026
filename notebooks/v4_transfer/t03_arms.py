"""Transfer arms: train LightGBM on a source country, fit the decision on source tune, score US holdout entities.

usage: t03_arms.py ARM [ARM ...]    (results -> artifacts/v4_tmp/res_{ARM}.json + a line in the notes)
"""
from __future__ import annotations

import gc
import json
import sys
import time

import lightgbm as lgb
import numpy as np

from common import CACHE, FEAT, FREQ_COLS, TMP, VOCAB_COLS, bucket, connect, note
from ber.decision import best_per_right, calibrate, decide, em_prior_shift, fit_decision
from ber.features import FEATURES
from ber.model import DEFAULT_PARAMS

PARAMS = {**DEFAULT_PARAMS, "learning_rate": 0.2, "seed": 42, "num_threads": 6}
RAW_S1 = CACHE / "raw_train_s1.parquet"
GT_S1 = CACHE / "train_gt_s1.parquet"
GROUP_B = {"fit": (0, 30), "tune": (30, 35), "holdout": (35, 40)}
import os
CON = connect(os.environ.get("V4_MEM", "3GB"))
CON.execute(f"CREATE TABLE ctry AS SELECT entity_id, country_norm AS country FROM read_parquet('{CACHE}/norm_train_s1.parquet')")


def load(group: str, where: str, vf: str | None = None, feats=FEATURES, zero=(), extra: str = "") -> dict:
    """Rows of feature file ``group`` (joined with S1 country as c.country) matching ``where``; vocab columns
    replaced from vf_{vf}.parquet when given; columns in ``zero`` set to 0. Returns X (float32), y, lid, rid."""
    cols = []
    for f in feats:
        if f in zero:
            cols.append(f"0::FLOAT AS {f}")
        elif vf and f in VOCAB_COLS:
            cols.append(f"v.{f}")
        else:
            cols.append(f"f.{f}")
    join = f"JOIN read_parquet('{TMP}/vf_{vf}.parquet') v ON v.lid = f.lid AND v.rid = f.rid" if vf else ""
    CON.execute(
        f"""CREATE OR REPLACE TEMP TABLE _m AS SELECT f.lid, f.rid, f.label, {', '.join(cols)} {extra}
            FROM read_parquet('{FEAT[group]}') f JOIN ctry c ON c.entity_id = f.lid {join} WHERE {where}"""
    )
    n = CON.execute("SELECT count(*) FROM _m").fetchone()[0]
    X = np.empty((n, len(feats)), dtype=np.float32)
    CON.execute("SET threads=1")
    ids = CON.execute("SELECT lid, rid, label FROM _m").fetch_arrow_table()
    i = 0
    for b in CON.execute(f"SELECT {', '.join(feats)} FROM _m").fetch_record_batch(500_000):
        m = b.num_rows
        for j in range(len(feats)):
            X[i:i + m, j] = b.column(j).to_numpy(zero_copy_only=False)
        i += m
    extra_cols = {}
    if extra:
        for c in [e.strip().split(" AS ")[-1] for e in extra.strip(", ").split(",")]:
            extra_cols[c] = CON.execute(f"SELECT {c} FROM _m").fetchnumpy()[c]
    CON.execute("SET threads=6")
    CON.execute("DROP TABLE _m")
    return {"X": X, "y": ids.column("label").to_numpy().astype(np.int8), "lid": ids.column("lid"),
            "rid": ids.column("rid"), **extra_cols}


def train(tr: dict, va: dict, feats=FEATURES) -> lgb.Booster:
    """LightGBM with early stopping (50) on the validation rows."""
    d = lgb.Dataset(tr["X"], tr["y"], feature_name=list(feats), free_raw_data=True)
    v = lgb.Dataset(va["X"], va["y"], reference=d, free_raw_data=True)
    d.construct()
    v.construct()
    tr.pop("X")
    va.pop("X")
    gc.collect()
    b = lgb.train(PARAMS, d, num_boost_round=2000, valid_sets=[v], callbacks=[lgb.early_stopping(50, verbose=False)])
    return b


def arrays(rows: dict, p: np.ndarray, group: str, countries: tuple[str, ...]) -> dict:
    """_group_arrays semantics restricted to S1 entities of ``countries`` in ``group``; rows = ALL rows of the group
    file for those countries (rival-context rows included), p = their scores."""
    import pyarrow as pa

    lo, hi = GROUP_B[group]
    cl = ", ".join(f"'{c}'" for c in countries)
    CON.register("pr_in", pa.table({"lid": rows["lid"], "rid": rows["rid"], "p": p, "label": rows["y"]}))
    CON.execute("CREATE OR REPLACE TEMP TABLE pr AS SELECT * FROM pr_in")
    CON.unregister("pr_in")
    CON.execute(
        f"""CREATE OR REPLACE TEMP TABLE s1g AS
            SELECT s.entity_id AS s1_id, row_number() OVER (ORDER BY s.entity_id) - 1 AS idx,
                   coalesce(g.n_matches, 0)::DOUBLE AS n_true
            FROM read_parquet('{RAW_S1}') s JOIN ctry c ON c.entity_id = s.entity_id
            LEFT JOIN read_parquet('{GT_S1}') g ON g.s1_id = s.entity_id
            WHERE {bucket('s.entity_id')} >= {lo} AND {bucket('s.entity_id')} < {hi} AND c.country IN ({cl})"""
    )
    CON.execute(
        """CREATE OR REPLACE TEMP TABLE ga AS
           WITH best AS (SELECT rid, arg_max(lid, p) AS best_lid FROM pr GROUP BY rid)
           SELECT s.idx, dense_rank() OVER (ORDER BY pr.rid) - 1 AS rc, pr.p, pr.label::BOOLEAN AS label
           FROM pr JOIN s1g s ON s.s1_id = pr.lid JOIN best b ON b.rid = pr.rid
           WHERE b.best_lid IN (SELECT s1_id FROM s1g)"""
    )
    CON.execute("SET threads=1")
    n_true = CON.execute("SELECT n_true FROM s1g ORDER BY idx").fetchnumpy()["n_true"]
    g = CON.execute("SELECT * FROM ga").fetchnumpy()
    CON.execute("SET threads=6")
    return {"lid_idx": g["idx"].astype(np.int64), "rid": g["rc"].astype(np.int64), "p": g["p"].astype(np.float64),
            "label": g["label"].astype(bool), "n_true": np.asarray(n_true, dtype=np.float64)}


def f05(a: dict, sel: np.ndarray) -> dict:
    """Macro F0.5 (singleton rule) plus FP / FN / TP pair counts and uncertain share."""
    from ber.decision import per_entity_f05

    f = per_entity_f05(a["lid_idx"], sel, a["label"], a["n_true"])
    tp = int((sel & a["label"]).sum())
    best = best_per_right(a["rid"], a["lid_idx"], a["p"])
    pb = a["p"][best]
    return {"f05": float(f.mean()), "tp": tp, "fp": int((sel & ~a["label"]).sum()), "fn": int(a["n_true"].sum() - tp),
            "uncertain": float(((pb > 0.05) & (pb < 0.95)).mean()), "n_ent": len(a["n_true"])}


def evaluate(booster, params, hold: dict, feats=FEATURES, train_prior: float | None = None) -> dict:
    """Score US holdout rows, apply the decision (and, separately, the per-country EM prior shift)."""
    p = booster.predict(hold["X"], num_iteration=booster.best_iteration or None)
    a = arrays(hold, p, "holdout", ("us",))
    out = f05(a, decide(params, a["lid_idx"], a["rid"], a["p"]))
    if train_prior is not None:  # C4: production per-country EM prior shift (D-010/D-023)
        best = best_per_right(a["rid"], a["lid_idx"], a["p"])
        pc = calibrate(params, best, a["p"], shift=0.0)
        prior, _ = em_prior_shift(pc[best], train_prior)
        delta = float(np.log(prior / (1 - prior)) - np.log(train_prior / (1 - train_prior)))
        ps = np.full(len(a["p"]), float(params.get("shift", 0.0)) + delta)
        em = f05(a, decide(params, a["lid_idx"], a["rid"], a["p"], pair_shift=ps))
        out["em"] = {"delta": delta, **{k: em[k] for k in ("f05", "fp", "fn")}}
    return out


def fit_source(tr, va_in, tune_load, tune_countries, feats=FEATURES):
    """Train, then fit the decision on the source tune group; return booster, params, train prior (best-per-right)."""
    t0 = time.time()
    b = train(tr, va_in, feats)
    tune_all = tune_load()
    tp = b.predict(tune_all["X"], num_iteration=b.best_iteration)
    ta = arrays(tune_all, tp, "tune", tune_countries)
    params = fit_decision(ta)
    best = best_per_right(ta["rid"], ta["lid_idx"], ta["p"])
    prior = float(ta["label"][best].mean())
    tune_score = f05(ta, decide(params, ta["lid_idx"], ta["rid"], ta["p"]))["f05"]
    print(f"  trained iters={b.best_iteration} tune_f05={tune_score:.5f} {time.time() - t0:.0f}s", flush=True)
    return b, params, prior, tune_score


IND, US = "c.country = 'india'", "c.country = 'us'"
FITA = f"f.in_group = 1 AND {bucket('f.lid')} < 15"


def record(arm: str, desc: str, res: dict) -> None:
    """Save the result json and append one table row to the notes."""
    (TMP / f"res_{arm}.json").write_text(json.dumps(res, default=float, indent=1))
    em = res.get("em", {})
    note(f"- **{arm}** ({desc}): US holdout F0.5 **{res['f05']:.5f}**, FP {res['fp']:,}, FN {res['fn']:,}, "
         f"uncertain {res['uncertain']:.4f}; +EM shift: {em.get('f05', float('nan')):.5f} (delta {em.get('delta', 0):+.3f}, "
         f"FP {em.get('fp', 0):,}, FN {em.get('fn', 0):,}); iters {res.get('iters')}, source tune F0.5 {res.get('tune', 0):.5f}")
    print(arm, json.dumps({k: v for k, v in res.items()}, default=float), flush=True)


def std_arm(arm, desc, train_where, tr_vf, va_where, va_vf, tune_where, tune_vf, tune_c, hold_vf, feats=FEATURES,
            zero=(), extra_train=None, eval_variants=None):
    """Generic arm: load, train, decision, evaluate (plus optional extra evaluation vocab variants)."""
    tr = load("fit", f"{FITA} AND {train_where}", tr_vf, feats)
    if extra_train is not None:
        tr = extra_train(tr)
    va = load("tune", f"f.in_group = 1 AND {va_where}", va_vf, feats)
    b, params, prior, ts = fit_source(tr, va, lambda: load("tune", tune_where, tune_vf, feats), tune_c, feats)
    del tr, va
    gc.collect()
    out = {}
    for name, (hvf, hz) in (eval_variants or {arm: (hold_vf, zero)}).items():
        hold = load("holdout", US, hvf, feats, zero=hz)
        res = evaluate(b, params, hold, feats, prior)
        res.update(iters=b.best_iteration, tune=ts, train_prior=prior)
        record(name, desc if name == arm else f"{desc}; eval vocab {hvf} zero={list(hz)[:2]}", res)
        out[name] = res
        del hold
        gc.collect()
    b.save_model(str(TMP / f"model_{arm}.txt"))
    return b, params, prior


def run(arm: str) -> None:
    """Dispatch one arm."""
    print("==", arm, flush=True)
    if arm == "A":
        std_arm("A", "in-domain: train India+US fitA, pooled vocab, decision on India+US tune", "TRUE", None, "TRUE",
                None, "TRUE", None, ("india", "us"), None)
    elif arm == "AUS":
        std_arm("AUS", "in-domain US only: train US fitA, US tune", US, None, US, None, US, None, ("us",), None)
    elif arm == "B":
        std_arm("B", "unseen US: train India fitA, India-only vocab (OOF), India tune; US rows India vocab", IND,
                "fit_IN_B", IND, "tune_IN", IND, "tune_IN", ("india",), "holdout_IN",
                eval_variants={"B": ("holdout_IN", ()), "B_zero": (None, tuple(VOCAB_COLS)),
                               "B_pooledvocab": (None, ()), "C2a_USS": ("holdout_USS", ()),
                               "C2b_MRG": ("holdout_MRG", ()), "C2_oracle_US": ("holdout_US", ())})
    elif arm == "C1v":
        feats = [f for f in FEATURES if f not in VOCAB_COLS]
        std_arm("C1v", "India-only train, vocab log-odds columns dropped", IND, None, IND, None, IND, None,
                ("india",), None, feats)
    elif arm == "C1vf":
        feats = [f for f in FEATURES if f not in VOCAB_COLS and f not in FREQ_COLS]
        std_arm("C1vf", "India-only train, vocab log-odds + name-frequency columns dropped", IND, None, IND, None, IND,
                None, ("india",), None, feats)
    elif arm == "C2c":
        std_arm("C2c", "self-labelled vocab everywhere: India rows INS (OOF), US rows USS (no labels)", IND,
                "fit_INS_B", IND, "tune_INS", IND, "tune_INS", ("india",), "holdout_USS")
    elif arm == "ASELF":
        std_arm("ASELF", "in-domain cost of self-labelled vocab: train India+US fitA, self vocab per country everywhere",
                "TRUE", "fit_SELF", "TRUE", "tune_SELF", "TRUE", "tune_SELF", ("india", "us"), "holdout_USS")
    elif arm == "ASELF2":
        std_arm("ASELF2", "in-domain cost of self-labelled vocab v2: train India+US fitA, self vocab v2 per country",
                "TRUE", "fit_SELF2", "TRUE", "tune_SELF2", "TRUE", "tune_SELF2", ("india", "us"), "holdout_USS2")
    elif arm == "C2d":
        std_arm("C2d", "self-labelled vocab v2 (pos rule + core_ratio>=0.6) everywhere: India INS2 (OOF), US USS2", IND,
                "fit_INS2_B", IND, "tune_INS2", IND, "tune_INS2", ("india",), "holdout_USS2")
    elif arm.startswith("C4o"):
        c4_oracle(arm)
    elif arm.startswith("C3"):
        c3(arm)
    else:
        raise SystemExit(f"unknown arm {arm}")


def c4_oracle(arm: str) -> None:
    """Upper bound of any target calibration remedy: keep the model, refit the decision rule on the TARGET (US) tune
    group with TRUE labels, score US holdout. C4o_B / C4o_C2c."""
    base = arm.split("_", 1)[1]
    tvf, hvf = {"B": ("tune_IN", "holdout_IN"), "C2c": ("tune_USS", "holdout_USS"),
                "C2d": ("tune_USS2", "holdout_USS2")}[base]
    b = lgb.Booster(model_file=str(TMP / f"model_{base}.txt"))
    ta = load("tune", US, tvf)
    a = arrays(ta, b.predict(ta["X"]), "tune", ("us",))
    del ta
    params = fit_decision(a)
    best = best_per_right(a["rid"], a["lid_idx"], a["p"])
    hold = load("holdout", US, hvf)
    res = evaluate(b, params, hold, FEATURES, float(a["label"][best].mean()))
    res.update(iters=b.current_iteration(), tune=0.0)
    record(arm, f"oracle calibration: {base} model, decision refit on US tune with true labels", res)


def c3(arm: str) -> None:
    """Pseudo-label self-training: score US fitA pairs (no labels) with the source model, keep confident pseudo
    labels (p > HI and best for its right record -> 1; p < LO or sole number offset in S -> 0), retrain on
    India fitA + pseudo US rows, decision still fitted on India tune, evaluate US holdout.
    C3 = on top of B (India vocab everywhere); C3s = on top of C2c (self-labelled vocab everywhere)."""
    import pyarrow as pa

    base, us_vf, in_vf, va_vf, hold_vf = {
        "C3": ("B", "fit_USIN", "fit_IN_B", "tune_IN", "holdout_IN"),
        "C3s": ("C2c", "fit_USUSS", "fit_INS_B", "tune_INS", "holdout_USS"),
    }[arm.rstrip("x")]
    hi, lo = (0.98, 0.02)
    b0 = lgb.Booster(model_file=str(TMP / f"model_{base}.txt"))
    us = load("fit", f"{FITA} AND {US}", us_vf, extra=", f.sg_sole_off_in_s AS soff")
    p = b0.predict(us["X"])
    CON.register("q", pa.table({"rid": us["rid"], "p": p, "i": np.arange(len(p))}))
    isbest = np.zeros(len(p), dtype=bool)
    isbest[CON.execute("SELECT arg_max(i, p) AS j FROM q GROUP BY rid").fetchnumpy()["j"]] = True
    CON.unregister("q")
    soff = us["soff"] == 1
    pos = (p > hi) & isbest & ~soff
    neg = ((p < lo) | (soff & (p < 0.5)))
    keep = pos | neg
    y_true = us["y"]
    acc = {"n_rows": int(len(p)), "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
           "pos_precision": float(y_true[pos].mean()), "neg_precision": float(1 - y_true[neg].mean()),
           "coverage": float(keep.mean()), "true_pos_covered": float(pos[y_true == 1].mean())}
    note(f"- {arm} pseudo labels on US fitA (labels used ONLY for this report): {json.dumps(acc)}")
    print(acc, flush=True)
    ind = load("fit", f"{FITA} AND {IND}", in_vf)
    tr = {"X": np.vstack([ind["X"], us["X"][keep]]), "y": np.concatenate([ind["y"], pos[keep].astype(np.int8)])}
    del ind, us
    gc.collect()
    va = load("tune", f"f.in_group = 1 AND {IND}", va_vf)
    b, params, prior, ts = fit_source(tr, va, lambda: load("tune", IND, va_vf), ("india",))
    del tr, va
    gc.collect()
    hold = load("holdout", US, hold_vf)
    res = evaluate(b, params, hold, FEATURES, prior)
    res.update(iters=b.best_iteration, tune=ts, train_prior=prior, pseudo=acc)
    record(arm, f"self-training on {base}: India fitA + confident pseudo-labelled US fitA pairs", res)
    b.save_model(str(TMP / f"model_{arm}.txt"))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(a)
        gc.collect()
