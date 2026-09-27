"""Stage 3 (D-028): combine the v5 stage-2 probability with cross-encoder scores; measure on holdout; predict test.

Protocol (no leakage): the cross-encoders never saw tune/holdout/test. Stage 3 is a small LightGBM on
[logit(p2), ce_<model>...] trained on the TUNE group with 5-fold cross-fitting by S1 id (OOF tune predictions fit
the decision rule); a final model on all of tune scores HOLDOUT (reported once) and TEST.
Pairs outside a model's scored band keep NaN for that model (LightGBM handles missing).

With --ctx, each score also gets competition context: over the S1's candidates and over the S2/S3 record's
candidate S1s (rank, gap to the best / best other, counts), plus flags for an empty right address and the right's
source. pred2_<g> holds every rival row of a group's right records, so the context is computed the same way in
tune, holdout and test.

With --s2feats=a,b,... those columns of RUN_DIR/s2_<g>.parquet (stage-2 input features) are added too.

A model's scores may come from several dirs joined by '+' (e.g. e5=first_band+extended_band, D-029).

Usage: python stage3.py RUN_DIR OUT_DIR [--ctx] [--s2feats=a,b] name1=CE_DIR1 [name2=CE_DIR2 ...]
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
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.config import load_config  # noqa: E402
from ber.decision import choose_decision  # noqa: E402
from ber.duck import connect  # noqa: E402
from ber.pipeline import _group_arrays  # noqa: E402
from ber.split import group_where  # noqa: E402

C = Path(__file__).resolve().parents[3] / "artifacts" / "cache"
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=1.0,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42, deterministic=True,
              force_row_wise=True, num_threads=8)
ROUNDS = 400


def ctx_sql(scores: list[str]) -> tuple[str, list[str]]:
    """SELECT list and names of competition features per score column over windows l (same S1) and r (same right)."""
    cols = ["count(*) OVER l AS n_l", "count(*) OVER r AS n_r"]
    for s in scores:
        for w in ("l", "r"):
            best_other = (f"CASE WHEN {s} = max({s}) OVER {w} THEN nth_value({s}, 2) OVER ({w} ORDER BY {s} DESC NULLS "
                          f"LAST ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) ELSE max({s}) OVER {w} END")
            cols += [f"rank() OVER ({w} ORDER BY {s} DESC NULLS LAST) AS {s}_{w}rk",
                     f"{s} - max({s}) OVER {w} AS {s}_{w}gap",
                     f"{s} - {best_other} AS {s}_{w}gap2"]
            if s != "lg2":
                cols.append(f"count({s}) OVER {w} AS {s}_{w}n")
    cols.append("sum(p2) OVER r AS r_psum")
    names = [c.rsplit(" AS ", 1)[1] for c in cols]
    return ", ".join(cols), names


CTX_FLAGS = ["r_ae", "r_s3"]


def ce_files(d: Path, g: str) -> str:
    """SQL list of a model's ce_<g>.parquet files; d may join several dirs with '+' (disjoint p bands of one model)."""
    return "[" + ", ".join(f"'{Path(x) / f'ce_{g}.parquet'}'" for x in str(d).split("+")) + "]"


def frame(con: duckdb.DuckDBPyConnection, pred2: Path, ce: dict[str, Path], g: str, ctx: bool = False,
          s2feats: list[str] | None = None) -> pa.Table:
    """Pairs of pred2_<g> (all rows) with logit(p2) and each model's ce score (NaN outside its band);
    with ctx, plus competition features (ctx_sql) and CTX_FLAGS; with s2feats, plus those stage-2 input columns."""
    return con.execute(frame_sql(pred2, ce, g, ctx, s2feats)).to_arrow_table()


def frame_sql(pred2: Path, ce: dict[str, Path], g: str, ctx: bool = False, s2feats: list[str] | None = None) -> str:
    """SQL of frame() (see there)."""
    joins = " ".join(f"LEFT JOIN read_parquet({ce_files(d, g)}) c_{n} ON c_{n}.lid = p.lid AND c_{n}.rid = p.rid"
                     for n, d in ce.items())
    cols = ", ".join(f"c_{n}.ce AS ce_{n}" for n in ce)
    lab = ", p.label" if g != "test" else ""
    s2 = ""
    if s2feats:
        joins += f" JOIN read_parquet('{pred2.parent / f's2_{g}.parquet'}') f ON f.lid = p.lid AND f.rid = p.rid"
        s2 = ", " + ", ".join(f"f.{c}::FLOAT AS {c}" for c in s2feats)
    base = f"""SELECT p.lid, p.rid, p.p AS p2{lab},
            ln(greatest(p.p, 1e-7) / greatest(1 - p.p, 1e-7)) AS lg2, {cols}{s2},
            (md5_number(p.lid) % 5)::INT AS fold
        FROM read_parquet('{pred2}') p {joins}"""
    if not ctx:
        return base + " ORDER BY p.lid, p.rid"  # fixed row order: LightGBM bagging is then reproducible
    split = "test" if g == "test" else "train"
    raw = ", ".join(f"'{C / f'raw_{split}_s{k}.parquet'}'" for k in (2, 3))
    extra, _ = ctx_sql(["lg2"] + [f"ce_{n}" for n in ce])
    return f"""SELECT q.*, {extra},
            (coalesce(r.business_address, '') = '')::INT AS r_ae, starts_with(q.rid, 'S3-')::INT AS r_s3
        FROM ({base}) q JOIN read_parquet([{raw}]) r ON r.entity_id = q.rid
        WINDOW l AS (PARTITION BY q.lid), r AS (PARTITION BY q.rid)
        ORDER BY q.lid, q.rid"""


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


def ho_ids(con: duckdb.DuckDBPyConnection) -> pa.Array:
    """Source-1 ids of the holdout group (train split)."""
    return con.execute(f"""SELECT entity_id FROM read_parquet('{C / "raw_train_s1.parquet"}')
                           WHERE {group_where("entity_id", "holdout")}""").to_arrow_table().column(0).combine_chunks()


def main(run: Path, out: Path, ce: dict[str, Path], ctx: bool = False, s2feats: list[str] | None = None) -> None:
    """Cross-fit stage 3 on tune, score holdout (+ test when every model has ce_test), compare with v5 alone."""
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    con = connect(cfg)
    feats = ["lg2"] + [f"ce_{n}" for n in ce]
    if ctx:
        feats += ctx_sql(feats)[1] + CTX_FLAGS
    feats += s2feats or []
    tu = frame(con, run / "pred2_tune.parquet", ce, "tune", ctx, s2feats)
    ho = frame(con, run / "pred2_holdout.parquet", ce, "holdout", ctx, s2feats)
    y = tu.column("label").to_numpy().astype(np.float32)
    x_tu, x_ho = matrix(tu, feats), matrix(ho, feats)
    fold = tu.column("fold").to_numpy()
    # rival rows of tune right records whose S1 is in the holdout group never train (keeps the holdout clean)
    keep = ~pc.is_in(tu.column("lid"), value_set=ho_ids(con)).to_numpy(zero_copy_only=False)
    oof = np.zeros(len(y), dtype=np.float64)
    for k in range(5):
        m = fold == k
        b = lgb.train(PARAMS, lgb.Dataset(x_tu[~m & keep], y[~m & keep], feature_name=feats), ROUNDS)
        oof[m] = b.predict(x_tu[m])
    final = lgb.train(PARAMS, lgb.Dataset(x_tu[keep], y[keep], feature_name=feats), ROUNDS)
    final.save_model(str(out / "stage3_model.txt"))
    p_ho = final.predict(x_ho)
    arr = {"tune": _group_arrays(con, cfg, write(tu, oof, out / "pred2_tune.parquet"), "tune"),
           "holdout": _group_arrays(con, cfg, write(ho, p_ho, out / "pred2_holdout.parquet"), "holdout")}
    rep3 = choose_decision(arr["tune"], arr["holdout"])
    base = choose_decision(_group_arrays(con, cfg, run / "pred2_tune.parquet", "tune"),
                           _group_arrays(con, cfg, run / "pred2_holdout.parquet", "holdout"))
    res = {"models": list(ce), "ctx": ctx, "s2feats": s2feats or [], "v5_holdout": base["holdout"][base["params"]["rule"]], "v5_rule": base["params"]["rule"],
           "stage3_holdout": rep3["holdout"][rep3["params"]["rule"]], "stage3_rule": rep3["params"]["rule"],
           "stage3_all_rules": rep3["holdout"],
           "importance": dict(zip(feats, final.feature_importance("gain").round(1).tolist(), strict=True))}
    res["gain"] = res["stage3_holdout"] - res["v5_holdout"]
    (out / "decision_params.json").write_text(json.dumps(rep3["params"], default=float))
    (out / "stage3.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float), flush=True)
    del tu, ho, x_tu, x_ho, arr  # free memory before the 19M-row test frame
    if all((Path(x) / "ce_test.parquet").exists() for d in ce.values() for x in str(d).split("+")):
        predict_test(con, frame_sql(run / "pred2_test.parquet", ce, "test", ctx, s2feats), final, feats, out)
        res["test_written"] = True
        (out / "stage3.json").write_text(json.dumps(res, indent=1, default=float))


def predict_test(con: duckdb.DuckDBPyConnection, sql: str, model: lgb.Booster, feats: list[str], out: Path) -> None:
    """Materialize the test frame to disk (windows need all rows), then predict it in batches -> pred2_test.parquet."""
    tmp = out / "test_frame.parquet"
    con.execute(f"COPY ({sql}) TO '{tmp}' (FORMAT parquet)")
    writer = None
    for bt in pq.ParquetFile(tmp).iter_batches(batch_size=1_000_000, columns=["lid", "rid", *feats]):
        t = pa.Table.from_batches([bt])
        ob = pa.table({"lid": t.column("lid"), "rid": t.column("rid"),
                       "p": pa.array(model.predict(matrix(t, feats)).astype(np.float32))})
        writer = writer or pq.ParquetWriter(out / "pred2_test.parquet", ob.schema)
        writer.write_table(ob)
    writer.close()
    tmp.unlink()


if __name__ == "__main__":
    rest = sys.argv[3:]
    s2f = [a.split("=", 1)[1].split(",") for a in rest if a.startswith("--s2feats=")]
    main(Path(sys.argv[1]), Path(sys.argv[2]),
         {a.split("=")[0]: Path(a.split("=")[1]) for a in rest if not a.startswith("--")}, "--ctx" in rest,
         s2f[0] if s2f else None)
