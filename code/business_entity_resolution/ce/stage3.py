"""Stage 3 (D-028): combine the v5 stage-2 probability with cross-encoder scores; measure on holdout; predict test.

Protocol (no leakage): the cross-encoders never saw tune/holdout/test. Stage 3 is a small LightGBM on
[logit(p2), ce_<model>...] trained on the TUNE group with 5-fold cross-fitting by S1 id (OOF tune predictions fit
the decision rule); a final model on all of tune scores HOLDOUT (reported once) and TEST.
Pairs outside a model's scored band keep NaN for that model (LightGBM handles missing).

Usage: python stage3.py RUN_DIR OUT_DIR name1=CE_DIR1 [name2=CE_DIR2 ...]
  CE_DIRk holds ce_tune.parquet, ce_holdout.parquet[, ce_test.parquet] (lid, rid, ce).
Writes OUT_DIR/{pred2_tune,pred2_holdout[,pred2_test]}.parquet (lid, rid, p[, label]), decision_params.json,
stage3.json (holdout F0.5 of v5 alone vs stage 3 under the same protocol).
"""

import json
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.config import load_config  # noqa: E402
from ber.decision import choose_decision  # noqa: E402
from ber.duck import connect  # noqa: E402
from ber.pipeline import _group_arrays  # noqa: E402

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=1.0,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, deterministic=True,
              force_row_wise=True, num_threads=8)
ROUNDS = 400


def frame(con: duckdb.DuckDBPyConnection, pred2: Path, ce: dict[str, Path], g: str) -> pa.Table:
    """Pairs of pred2_<g> (all rows) with logit(p2) and each model's ce score (NaN outside its band)."""
    joins = " ".join(f"LEFT JOIN read_parquet('{d / f'ce_{g}.parquet'}') c_{n} ON c_{n}.lid = p.lid AND c_{n}.rid = p.rid"
                     for n, d in ce.items())
    cols = ", ".join(f"c_{n}.ce AS ce_{n}" for n in ce)
    lab = ", p.label" if g != "test" else ""
    return con.execute(f"""SELECT p.lid, p.rid, p.p AS p2{lab},
            ln(greatest(p.p, 1e-7) / greatest(1 - p.p, 1e-7)) AS lg2, {cols},
            (md5_number(p.lid) % 5)::INT AS fold
        FROM read_parquet('{pred2}') p {joins}""").to_arrow_table()


def matrix(t: pa.Table, feats: list[str]) -> np.ndarray:
    """Float32 feature matrix (NULL -> NaN)."""
    return np.column_stack([t.column(f).to_numpy(zero_copy_only=False).astype(np.float32) for f in feats])


def write(t: pa.Table, p: np.ndarray, path: Path) -> Path:
    """Write (lid, rid, p[, label]) parquet."""
    cols = {"lid": t.column("lid"), "rid": t.column("rid"), "p": pa.array(p.astype(np.float32))}
    if "label" in t.column_names:
        cols["label"] = t.column("label")
    pq.write_table(pa.table(cols), path)
    return path


def main(run: Path, out: Path, ce: dict[str, Path]) -> None:
    """Cross-fit stage 3 on tune, score holdout (+ test when every model has ce_test), compare with v5 alone."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    con = connect(cfg)
    feats = ["lg2"] + [f"ce_{n}" for n in ce]
    tu = frame(con, run / "pred2_tune.parquet", ce, "tune")
    ho = frame(con, run / "pred2_holdout.parquet", ce, "holdout")
    y = tu.column("label").to_numpy().astype(np.float32)
    x_tu, x_ho = matrix(tu, feats), matrix(ho, feats)
    fold = tu.column("fold").to_numpy()
    oof = np.zeros(len(y), dtype=np.float64)
    for k in range(5):
        m = fold == k
        b = lgb.train(PARAMS, lgb.Dataset(x_tu[~m], y[~m], feature_name=feats), ROUNDS)
        oof[m] = b.predict(x_tu[m])
    final = lgb.train(PARAMS, lgb.Dataset(x_tu, y, feature_name=feats), ROUNDS)
    final.save_model(str(out / "stage3_model.txt"))
    p_ho = final.predict(x_ho)
    arr = {"tune": _group_arrays(con, cfg, write(tu, oof, out / "pred2_tune.parquet"), "tune"),
           "holdout": _group_arrays(con, cfg, write(ho, p_ho, out / "pred2_holdout.parquet"), "holdout")}
    rep3 = choose_decision(arr["tune"], arr["holdout"])
    base = choose_decision(_group_arrays(con, cfg, run / "pred2_tune.parquet", "tune"),
                           _group_arrays(con, cfg, run / "pred2_holdout.parquet", "holdout"))
    res = {"models": list(ce), "v5_holdout": base["holdout"][base["params"]["rule"]], "v5_rule": base["params"]["rule"],
           "stage3_holdout": rep3["holdout"][rep3["params"]["rule"]], "stage3_rule": rep3["params"]["rule"],
           "stage3_all_rules": rep3["holdout"],
           "importance": dict(zip(feats, final.feature_importance("gain").round(1).tolist(), strict=True))}
    res["gain"] = res["stage3_holdout"] - res["v5_holdout"]
    (out / "decision_params.json").write_text(json.dumps(rep3["params"], default=float))
    if all((d / "ce_test.parquet").exists() for d in ce.values()):
        te = frame(con, run / "pred2_test.parquet", ce, "test")
        write(te, final.predict(matrix(te, feats)), out / "pred2_test.parquet")
        res["test_written"] = True
    (out / "stage3.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), {a.split("=")[0]: Path(a.split("=")[1]) for a in sys.argv[3:]})
