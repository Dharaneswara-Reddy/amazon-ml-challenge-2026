"""Task 3d: a third stacking stage trained on the TUNE group only (5-fold by S1 -> OOF p3 on tune; fold-model average on
holdout). Features: competition + entity features recomputed from p2, plus the top-60 stage-2 inputs and an empty-address
flag. Decision fitted on OOF tune p3, scored on holdout.

usage: t3_stage3.py NAME PRED2_TUNE PRED2_HOLDOUT S2X_PREFIX
"""
import json, sys, zlib, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb
from common import OUT, ROOT, RUN, con, arrays, note
from t3_feats import TOP, E, esql
from ber.stage2 import competition_sql, STAGE2_FEATURES
from ber.decision import compare_decisions

name = sys.argv[1] if len(sys.argv) > 1 else "s3_v4"
p2 = {"tune": Path_t if False else None}
from pathlib import Path
p2 = {"tune": Path(sys.argv[2]) if len(sys.argv) > 2 else RUN / "pred2_tune.parquet",
      "holdout": Path(sys.argv[3]) if len(sys.argv) > 3 else RUN / "pred2_holdout.parquet"}
C = ROOT / "artifacts/cache"
c = con()
S3C = [f.replace("s2_", "s3_") for f in STAGE2_FEATURES]
E3 = [f"{e}_3" for e in E]
FE = TOP + S3C + E3 + ["r_noaddr"]
mats = {}
for g in ("tune", "holdout"):
    f = OUT / f"s3x_{name}_{g}.parquet"
    if not f.exists():
        comp = competition_sql(f"(SELECT lid, rid, p FROM read_parquet('{p2[g]}'))")
        c.execute(f"CREATE OR REPLACE TEMP TABLE comp AS {comp}")
        c.execute(f"CREATE OR REPLACE TEMP TABLE ef AS {esql(p2[g])}")
        c.execute(f"""CREATE OR REPLACE TEMP TABLE rn AS SELECT entity_id, coalesce(length(trim(addr_norm)),0)=0 AS noaddr
                      FROM read_parquet(['{C}/norm_train_s2.parquet','{C}/norm_train_s3.parquet'])""")
        sc = ", ".join(f"comp.{a} AS {b}" for a, b in zip(STAGE2_FEATURES, S3C))
        ec = ", ".join(f"ef.{a} AS {b}" for a, b in zip(E, E3))
        tc = ", ".join(f"x.{t}" for t in TOP)
        c.execute(f"""COPY (SELECT x.lid, x.rid, x.label, x.in_group, {tc}, {sc}, {ec}, rn.noaddr::FLOAT AS r_noaddr
            FROM read_parquet('{OUT}/s2x_{g}.parquet') x JOIN comp ON comp.lid = x.lid AND comp.rid = x.rid
            JOIN ef ON ef.lid = x.lid AND ef.rid = x.rid JOIN rn ON rn.entity_id = x.rid) TO '{f}' (FORMAT parquet)""")
        assert pq.ParquetFile(f).metadata.num_rows == pq.ParquetFile(OUT / f"s2x_{g}.parquet").metadata.num_rows
    t = pq.read_table(f)
    mats[g] = (np.column_stack([t[k].to_numpy().astype(np.float32) for k in FE]), t["label"].to_numpy().astype(np.int8),
               t["in_group"].to_numpy().astype(bool), t.select(["lid", "rid", "label"]))
    del t
print("features built", flush=True)
K = 5
X, y, ing, ids = mats["tune"]
fold = np.array([zlib.crc32(s.encode()) % K for s in ids["lid"].to_pylist()])
params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=5, seed=42, deterministic=True)
p_tune = np.zeros(len(y)); Xh = mats["holdout"][0]; p_hold = np.zeros(len(Xh)); iters = []
for k in range(K):
    tr = ing & (fold != k); va = ing & (fold == k)
    b = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=FE), 3000, valid_sets=[lgb.Dataset(X[va], y[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    iters.append(b.best_iteration)
    p_tune[fold == k] = b.predict(X[fold == k], num_iteration=b.best_iteration)
    p_hold += b.predict(Xh, num_iteration=b.best_iteration) / K
    if k == 0:
        imp = sorted(zip(FE, b.feature_importance("gain")), key=lambda x: -x[1]); tot = sum(v for _, v in imp)
for g, pp in (("tune", p_tune), ("holdout", p_hold)):
    tb = mats[g][3]
    pq.write_table(pa.table({"lid": tb["lid"], "rid": tb["rid"], "p": pa.array(pp.astype(np.float32)), "label": tb["label"]}),
                   OUT / f"pred_{name}_{g}.parquet")
r = compare_decisions(arrays(OUT / f"pred_{name}_tune.parquet", "tune"), arrays(OUT / f"pred_{name}_holdout.parquet", "holdout"))
res = {"name": name, "iters": iters, "holdout": {k: round(v, 5) for k, v in r["holdout"].items()},
       "tune_oof": {k: round(v, 5) for k, v in r["tune"].items()}, "top_imp": [(a, round(v / tot * 100, 2)) for a, v in imp[:10]]}
print(json.dumps(res), flush=True)
with open(OUT / "t3_results.jsonl", "a") as fh:
    fh.write(json.dumps(res) + "\n")
note(f"- stage-3 on tune (5-fold OOF) `{name}` from {p2['holdout'].name}: holdout expF **{res['holdout']['expected_f']}** "
     f"(thr {res['holdout']['threshold']}), tune-OOF expF {res['tune_oof']['expected_f']}, iters {iters}. top gain {res['top_imp'][:6]}")
