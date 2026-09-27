"""Quick experiment: does a signed house-number-diff signature improve holdout F0.5 on top of stage-2 p?

Correction model = LightGBM on [logit(p2), signed diff of first numbers, relation code, flags], cross-fitted on
tune (2 folds by S1 hash) and trained on full tune for holdout. Decision params refit on tune with the corrected
p (same fit_decision as the pipeline), then holdout scored with _group_arrays + decide. Baseline redone the same way.
"""
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from ber.config import load_config  # noqa: E402
from ber.decision import decide, fit_decision, macro_f05  # noqa: E402
from ber.pipeline import _group_arrays  # noqa: E402

RUN = ROOT / "artifacts/runs/20260925T161436_b538130_1242e7cd_v2"
OUT = ROOT / "notebooks/v3_research/out"
C = ROOT / "artifacts/cache"
cfg = load_config(root=ROOT)
con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=4")
con.execute(f"SET temp_directory='{ROOT}/artifacts/tmp/v3a'")
con.execute(f"""CREATE TEMP TABLE rn AS SELECT entity_id, addr_nums, addr_house FROM read_parquet('{C}/norm_train_s2.parquet')
                UNION ALL SELECT entity_id, addr_nums, addr_house FROM read_parquet('{C}/norm_train_s3.parquet')""")
con.execute(f"CREATE TEMP TABLE ln AS SELECT entity_id, addr_nums, addr_house FROM read_parquet('{C}/norm_train_s1.parquet')")
FEAT = """
  WITH x AS (SELECT pr.lid, pr.rid, pr.p, pr.label, split_part(l.addr_nums,' ',1) a, split_part(r.addr_nums,' ',1) b,
             l.addr_nums la, r.addr_nums ra
             FROM read_parquet('{f}') pr JOIN ln l ON l.entity_id = pr.lid JOIN rn r ON r.entity_id = pr.rid)
  SELECT lid, rid, p, label::INT AS label,
    CASE WHEN try_cast(a AS BIGINT) IS NOT NULL AND try_cast(b AS BIGINT) IS NOT NULL AND length(a) < 12 AND length(b) < 12
         THEN (b::BIGINT - a::BIGINT)::DOUBLE END AS d1,
    (a = '')::INT + 2 * (b = '')::INT AS miss_code,
    (length(a) = length(b) + 1 AND starts_with(a, b))::INT AS trunc,
    (length(a) = length(b) + 1 AND list_contains(list_transform(range(1, length(a) + 1), i -> a[1:i-1] || a[i+1:]), b))::INT AS del1,
    (length(b) > length(a) AND starts_with(b, a) AND a <> '')::INT AS plus_digits,
    len(string_split(la,' ')) AS nl, len(string_split(ra,' ')) AS nr,
    len(list_filter(string_split(ra,' '), y -> NOT list_contains(string_split(la,' '), y))) AS r_extra_nums,
    len(list_filter(string_split(la,' '), y -> NOT list_contains(string_split(ra,' '), y))) AS l_extra_nums
  FROM x"""
COLS = ["d1", "miss_code", "trunc", "del1", "plus_digits", "nl", "nr", "r_extra_nums", "l_extra_nums"]


def load(split):
    """Return dict of numpy arrays for pred2_<split> with the new number features."""
    return con.execute(FEAT.format(f=RUN / f"pred2_{split}.parquet") + " ORDER BY lid, rid").fetchnumpy()


def X(d):
    """Feature matrix: logit of stage-2 p plus the new number features."""
    p = np.clip(d["p"].astype(np.float64), 1e-7, 1 - 1e-7)
    cols = [np.log(p / (1 - p))] + [np.asarray(d[c], dtype=np.float64) for c in COLS]
    return np.column_stack(cols)


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=1.0,
              verbose=-1, seed=42, num_threads=4)
tune = load("tune")
hold = load("holdout")
print("rows tune", len(tune["p"]), "holdout", len(hold["p"]), flush=True)
Xt, yt = X(tune), tune["label"]
fold = np.array([hash(s) % 2 for s in tune["lid"]])  # python hash is salted per process, fine for a 2-fold split
p_tune = np.zeros(len(yt))
for k in (0, 1):
    m = lgb.train(PARAMS, lgb.Dataset(Xt[fold != k], yt[fold != k]), num_boost_round=300)
    p_tune[fold == k] = m.predict(Xt[fold == k])
m = lgb.train(PARAMS, lgb.Dataset(Xt, yt), num_boost_round=300)
p_hold = m.predict(X(hold))
print("feature gain:", dict(zip(["logit_p2"] + COLS, m.feature_importance("gain").round(0))), flush=True)
import pyarrow as pa, pyarrow.parquet as pq  # noqa: E401,E402
for name, d, pn in (("tune", tune, p_tune), ("holdout", hold, p_hold)):
    for tag, pv in (("base", d["p"].astype(np.float64)), ("num", pn)):
        pq.write_table(pa.table({"lid": d["lid"], "rid": d["rid"], "p": pv.astype(np.float32),
                                 "label": d["label"].astype(np.int8)}), OUT / f"p_{tag}_{name}.parquet")
del tune, hold, Xt
for tag in ("base", "num"):
    gt = _group_arrays(con, cfg, OUT / f"p_{tag}_tune.parquet", "tune")
    gh = _group_arrays(con, cfg, OUT / f"p_{tag}_holdout.parquet", "holdout")
    params = fit_decision(gt)
    sel = decide(params, gh["lid_idx"], gh["rid"], gh["p"])
    print(tag, "holdout F0.5", macro_f05(gh["lid_idx"], sel, gh["label"], gh["n_true"]),
          "FP", int((sel & ~gh["label"]).sum()), "TP", int((sel & gh["label"]).sum()), flush=True)
