"""P05: train + decide + score one arm with the identical protocol.

Usage: python p05_arm.py ARM FIT_FILE TUNE_FILE HOLDOUT_FILE [extra_feature,...]
FIT_FILE rows are filtered to in_group = 1 and S1 md5 bucket in FIT-SUB (0-4); TUNE in_group = 1 for early stopping.
Model = ber DEFAULT_PARAMS + learning_rate 0.2, seed 42; features = ber.features.FEATURES + extras.
Decision = ber.decision.choose_decision(tune arrays, holdout arrays) (rule picked on tune halves only).
Reports holdout macro F0.5, per country, oracle ceilings, and the loss decomposition; appends to the E note."""
import numpy as np
from common import *
from ber.features import FEATURES
from ber.model import train, predict_file
from ber.split import group_where
from ber.decision import (choose_decision, per_entity_f05, prepare, calibrate, choose_k, mask_from_k, _logit_shift,
                          MissingModel, Calibrator, select)

arm, fit_f, tune_f, hold_f = sys.argv[1:5]
extra = [x for x in (sys.argv[5].split(",") if len(sys.argv) > 5 else []) if x]
feats = FEATURES + extra
d = T / "arms" / arm; d.mkdir(parents=True, exist_ok=True)
con = connect(d / "w.duckdb", mem="2GB", threads=4)
t = time.time()
fit_in, tune_in = d / "fit_in.parquet", d / "tune_in.parquet"
if not fit_in.exists():
    con.execute(f"COPY (SELECT * FROM '{fit_f}' WHERE in_group = 1 AND {bucket_pred('lid', 'fitsub')}) TO '{fit_in}' (FORMAT parquet)")
    con.execute(f"COPY (SELECT * FROM '{tune_f}' WHERE in_group = 1) TO '{tune_in}' (FORMAT parquet)")
nfit = con.execute(f"SELECT count(*), sum(label) FROM '{fit_in}'").fetchone()
log(arm, "fit rows/pos", nfit, time.time() - t)
mp = d / "model.txt"
import lightgbm as lgb
if mp.exists():
    bst = lgb.Booster(model_file=str(mp))
else:
    bst = train(fit_in, tune_in, {"learning_rate": 0.2}, 42, 8, rounds=5000, features=feats)
    bst.save_model(str(mp), num_iteration=bst.best_iteration)
    bst = lgb.Booster(model_file=str(mp))
log(arm, "trained trees", bst.current_iteration(), time.time() - t)
for g, f in (("tune", tune_f), ("holdout", hold_f)):
    if not (d / f"pred_{g}.parquet").exists():
        predict_file(bst, Path(f), d / f"pred_{g}.parquet", 500_000, features=feats)
log(arm, "predicted", time.time() - t)


def fetch(sql):
    """Materialize then fetch on one thread (ber.pipeline._fetch_numpy workaround)."""
    con.execute(f"CREATE OR REPLACE TEMP TABLE _f AS {sql}")
    con.execute("SET threads = 1")
    try:
        return con.execute("SELECT * FROM _f").fetchnumpy()
    finally:
        con.execute("SET threads = 4"); con.execute("DROP TABLE _f")


def arrays(g):
    """ber.pipeline._group_arrays equivalent (rival drop) + per-entity country and in-candidate true counts."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE s1g AS
        SELECT s.entity_id AS s1_id, row_number() OVER (ORDER BY s.entity_id) - 1 AS idx,
               coalesce(g.n_matches, 0)::DOUBLE AS n_true, n.country_norm AS country
        FROM '{C}/raw_train_s1.parquet' s LEFT JOIN '{C}/train_gt_s1.parquet' g ON g.s1_id = s.entity_id
        JOIN '{C}/norm_train_s1.parquet' n ON n.entity_id = s.entity_id
        WHERE {group_where('s.entity_id', g)}""")
    ent = fetch("SELECT n_true, country FROM s1g ORDER BY idx")
    pr = d / f"pred_{g}.parquet"
    pairs = fetch(f"""WITH pr AS (SELECT lid, rid, p, label FROM '{pr}'),
                 best AS (SELECT rid, arg_max(lid, p) AS best_lid FROM pr GROUP BY rid)
            SELECT s.idx, dense_rank() OVER (ORDER BY pr.rid) - 1 AS rc, pr.p, pr.label::BOOLEAN AS label
            FROM pr JOIN s1g s ON s.s1_id = pr.lid JOIN best b ON b.rid = pr.rid
            WHERE b.best_lid IN (SELECT s1_id FROM s1g)""")
    incand = fetch(f"""SELECT s.idx, sum(pr.label)::DOUBLE AS tp_c, count(*) AS n_c FROM '{pr}' pr JOIN s1g s ON s.s1_id = pr.lid GROUP BY 1""")
    n = len(ent["n_true"])
    tpc = np.zeros(n); tpc[incand["idx"]] = incand["tp_c"]
    ncand = np.zeros(n); ncand[incand["idx"]] = incand["n_c"]
    return {"lid_idx": pairs["idx"].astype(np.int64), "rid": pairs["rc"].astype(np.int64), "p": pairs["p"].astype(np.float64),
            "label": pairs["label"].astype(bool), "n_true": ent["n_true"]}, np.asarray(ent["country"]), tpc, ncand


(at, _, _, _), (ah, country, tpc, ncand) = arrays("tune"), arrays("holdout")
rep = choose_decision(at, ah)
prm = rep["params"]; rule = prm["rule"]
lid, rid, p, label, n_true = (ah[k] for k in ("lid_idx", "rid", "p", "label", "n_true"))
if rule == "threshold":
    sel = select(lid, rid, p, prm["threshold"])
else:
    prep = prepare(lid, rid, p, int(prm["max_k"]), len(n_true))
    pc0 = calibrate(prm, prep.is_best, p, shift=0.0)
    sel = mask_from_k(prep, choose_k(prep, _logit_shift(pc0, prm["shift"]), MissingModel.from_dict(prm["missing"]),
                                     prm["empty_weight"], Calibrator.from_dict(prm["cal_empty"]))[0])
f_ent = per_entity_f05(lid, sel, label, n_true)
F = float(f_ent.mean())
assert abs(F - rep["holdout"][rule]) < 1e-9, (F, rep["holdout"][rule])
n = len(n_true)
bc = lambda w: np.bincount(lid, weights=w.astype(np.float64), minlength=n)
tp, fp = bc(sel & label), bc(sel & ~label)
tp_arr = bc(label)                     # true pairs still present after the rival drop
dec_fn = tp_arr - tp                   # in arrays, not selected
o2o = tpc - tp_arr                     # in candidates, lost to a rival's one-to-one
miss = n_true - tpc                    # never generated


def f05(tp, fn, fp):
    """Per-entity F0.5 with the singleton rule."""
    den = 1.25 * tp + 0.25 * fn + fp
    f = np.where(den > 0, 1.25 * tp / np.maximum(den, 1e-12), 0.0)
    return np.where(n_true == 0, (fp == 0).astype(float), f)


base = f05(tp, miss + o2o + dec_fn, fp).mean()
assert abs(base - F) < 1e-9
single = n_true == 0
fixes = {
    "oracle: perfect decisions on candidates (TP = all in-candidate true pairs, no FP)": f05(tpc, miss, 0 * fp).mean(),
    "oracle: perfect decisions after rival drop": f05(tp_arr, miss + o2o, 0 * fp).mean(),
    "(a) add all blocking misses as TP": f05(tp + miss, o2o + dec_fn, fp).mean(),
    "(b) recover one-to-one losses": f05(tp + o2o, miss + dec_fn, fp).mean(),
    "(c) select all decision FNs": f05(tp + dec_fn, miss + o2o, fp).mean(),
    "(d1) remove FP on entities with matches": f05(tp, miss + o2o + dec_fn, np.where(single, fp, 0)).mean(),
    "(d2) remove FP on singletons": f05(tp, miss + o2o + dec_fn, np.where(single, 0, fp)).mean(),
}
cty = {c: float(f_ent[country == c].mean()) for c in np.unique(country)}
cty_n = {c: int((country == c).sum()) for c in np.unique(country)}
orc = f05(tpc, miss, 0 * fp)
res = dict(arm=arm, n_features=len(feats), fit_rows=int(nfit[0]), trees=bst.current_iteration(), rule=rule,
           holdout_f05=F, holdout_all_rules=rep["holdout"], tune_half_b=rep["selection"]["tune_half_b"],
           recall=float(tpc.sum() / n_true.sum()), pairs_per_s1=float(ncand.mean()), per_country=cty,
           oracle_per_country={c: float(orc[country == c].mean()) for c in cty},
           counts=dict(true=int(n_true.sum()), tp=int(tp.sum()), blocking_miss=int(miss.sum()), o2o=int(o2o.sum()),
                       dec_fn=int(dec_fn.sum()), fp_matched=int(fp[~single].sum()), fp_singleton=int(fp[single].sum()),
                       singletons_zeroed=int(((fp > 0) & single).sum())),
           fixes={k: float(v) for k, v in fixes.items()}, secs=round(time.time() - t))
(d / "result.json").write_text(json.dumps(res, indent=1, default=float))
imp = sorted(zip(feats, bst.feature_importance("gain")), key=lambda x: -x[1])[:25]
(d / "importance.json").write_text(json.dumps([(a, round(float(b))) for a, b in imp]))
log(json.dumps(res, default=float))
c = res["counts"]
note(f"""
### Arm `{arm}` — HOLDOUT macro F0.5 **{F:.5f}** (rule {rule}; {len(feats)} features, {res['trees']} trees, {nfit[0]:,} FIT-SUB rows)
- candidate recall {res['recall']:.4%}, candidate pairs per holdout S1 {res['pairs_per_s1']:.2f}
- per country: {', '.join(f'{k} {v:.5f} (n={cty_n[k]:,}; oracle {res["oracle_per_country"][k]:.5f})' for k, v in cty.items())}
- all rules on holdout: {', '.join(f'{k} {v:.5f}' for k, v in rep['holdout'].items())}
- true pairs {c['true']:,}: TP {c['tp']:,}, blocking miss {c['blocking_miss']:,}, one-to-one loss {c['o2o']:,}, decision FN {c['dec_fn']:,};
  FP {c['fp_matched']:,} on matched entities + {c['fp_singleton']:,} on singletons ({c['singletons_zeroed']:,} singletons zeroed)
- oracle / one-at-a-time fixes: {'; '.join(f'{k}: {v:.5f} ({(v - F) * 100:+.3f} pts)' for k, v in fixes.items())}
- top features by gain: {', '.join(a for a, _ in imp[:12])}
""")
con.close(); (d / "w.duckdb").unlink(missing_ok=True)
