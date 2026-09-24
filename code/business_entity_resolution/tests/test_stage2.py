"""Tests for ber.stage2: competition features (numpy vs SQL vs hand-checked), stage-2 matrix, OOF protocol."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ber.features import FEATURES
from ber.model import predict_file, train
from ber.split import group_where
from ber.stage2 import (
    MODEL2_FEATURES,
    STAGE2_FEATURES,
    competition_features,
    competition_sql,
    fold_sql,
    oof_predict,
    predict_protocol,
    write_stage2_features,
)

# L1: a (S2) 0.95, b (S3) 0.6, c (S3) 0.3 ; L2: b 0.8, c 0.2   (b prefers L2, c prefers L1)
LID = np.array(["S1-1", "S1-1", "S1-1", "S1-2", "S1-2"])
RID = np.array(["S2-a", "S3-b", "S3-c", "S3-b", "S3-c"])
P = np.array([0.95, 0.6, 0.3, 0.8, 0.2])
EXPECTED = {
    "s2_p1": [0.95, 0.6, 0.3, 0.8, 0.2],
    "s2_r_rank": [1, 2, 1, 1, 2],
    "s2_r_margin": [0.95, -0.2, 0.1, 0.2, -0.1],
    "s2_r_is_best": [1, 0, 1, 1, 0],
    "s2_r_share": [1.0, 0.6 / 1.4, 0.3 / 0.5, 0.8 / 1.4, 0.2 / 0.5],
    "s2_l_rank": [1, 2, 3, 1, 2],
    "s2_l_gap": [0.0, 0.35, 0.65, 0.0, 0.6],
    "s2_l_max": [0.95, 0.95, 0.95, 0.8, 0.8],
    "s2_l_sum": [1.85, 1.85, 1.85, 1.0, 1.0],
    "s2_l_n50": [2, 2, 2, 1, 1],
    "s2_l_n90": [1, 1, 1, 0, 0],
    "s2_l_n50_best": [1, 1, 1, 1, 1],
    "s2_mutual": [1, 0, 1, 1, 0],
    "s2_l_n50_s2": [1, 1, 1, 0, 0],
    "s2_l_n50_s3": [1, 1, 1, 1, 1],
    "s2_l_n90_s2": [1, 1, 1, 0, 0],
    "s2_l_n90_s3": [0, 0, 0, 0, 0],
}


def test_competition_features_hand_checked() -> None:
    """Every stage-2 feature on a two-entity, three-record example matches hand-computed values."""
    got = competition_features(LID, RID, P, mutual_k=4)
    assert list(got) == STAGE2_FEATURES
    for name, want in EXPECTED.items():
        assert got[name] == pytest.approx(np.array(want, dtype=float)), name


def test_mutual_requires_top_k_of_entity() -> None:
    """With mutual_k=2 the R-best pair ranked 3rd in its entity is not mutual."""
    assert competition_features(LID, RID, P, mutual_k=2)["s2_mutual"].tolist() == [1, 0, 0, 1, 0]


def _sql_features(con: duckdb.DuckDBPyConnection, lid, rid, p) -> pa.Table:
    """Run competition_sql over in-memory arrays and return rows ordered by (lid, rid)."""
    con.register("t_pairs", pa.table({"lid": lid, "rid": rid, "p": pa.array(p, pa.float32())}))
    return con.execute(competition_sql("t_pairs") + " ORDER BY lid, rid").to_arrow_table()


def test_sql_matches_hand_checked(con: duckdb.DuckDBPyConnection) -> None:
    """The DuckDB implementation reproduces the hand-checked table (float32 tolerance)."""
    got = _sql_features(con, LID, RID, P)
    order = np.lexsort((RID, LID))
    for name, want in EXPECTED.items():
        assert got.column(name).to_numpy() == pytest.approx(np.array(want, float)[order], abs=1e-6), name


def test_sql_matches_numpy_with_ties(con: duckdb.DuckDBPyConnection) -> None:
    """On random pairs with many tied probabilities the SQL and numpy features agree."""
    rng = np.random.default_rng(3)
    lid = np.array([f"S1-{i:03d}" for i in rng.integers(0, 60, 800)])
    rid = np.array([f"S{2 + i % 2}-{i:04d}" for i in rng.integers(0, 250, 800)])
    keep = np.unique(np.char.add(np.char.add(lid, "|"), rid), return_index=True)[1]
    lid, rid = lid[keep], rid[keep]
    p = np.round(rng.random(len(lid)), 1).astype(np.float32)
    got = _sql_features(con, lid, rid, p)
    ref = competition_features(lid, rid, p.astype(np.float64))
    order = np.lexsort((rid, lid))
    for name in STAGE2_FEATURES:
        assert got.column(name).to_numpy() == pytest.approx(ref[name][order], abs=1e-5), name


def _ids_in_group(con: duckdb.DuckDBPyConnection, group: str, n: int, prefix: str = "S1-") -> list[str]:
    """First ``n`` synthetic S1 ids whose md5 bucket falls in ``group``."""
    rows = con.execute(
        f"""SELECT id FROM (SELECT '{prefix}' || lpad(i::VARCHAR, 6, '0') AS id FROM range(20000) r(i))
            WHERE {group_where("id", group)} ORDER BY id LIMIT {n}"""
    ).fetchall()
    return [r[0] for r in rows]


def _write_feat(path: Path, lids: list[str], rng: np.random.Generator, per: int = 3, shared: int = 0) -> pa.Table:
    """Synthetic stage-1 feature parquet: ``per`` pairs per S1 (first one true), FEATURES random + signal in [0]."""
    lid = np.repeat(np.array(lids), per)
    rid = np.array([f"S{2 + i % 2}-{i:06d}" for i in range(len(lid))])
    if shared:
        rid[per + 1 :: per][:shared] = rid[0]  # rivals: other S1s also list the first right record
    label = np.tile(np.r_[1, np.zeros(per - 1, int)], len(lids)).astype(np.int8)
    cols = {"lid": lid, "rid": rid}
    for j, f in enumerate(FEATURES):
        x = rng.random(len(lid)).astype(np.float32)
        cols[f] = x + (2 * label if j == 0 else 0)
    cols["label"] = label
    tbl = pa.table(cols)
    pq.write_table(tbl, path)
    return tbl


def _write_pred(path: Path, feat: pa.Table, p: np.ndarray) -> None:
    """Stage-1 prediction parquet (lid, rid, p, label) for the rows of ``feat``."""
    pq.write_table(
        pa.table({"lid": feat["lid"], "rid": feat["rid"], "p": pa.array(p, pa.float32()), "label": feat["label"]}),
        path,
    )


def test_write_stage2_features_adds_columns_and_uses_context(con, tmp_path: Path) -> None:
    """Output keeps every feature row, adds STAGE2_FEATURES, and rival predictions change R-level features only."""
    rng = np.random.default_rng(0)
    feat = _write_feat(tmp_path / "feat.parquet", ["S1-a", "S1-b"], rng)
    _write_pred(tmp_path / "pred.parquet", feat, np.array([0.9, 0.2, 0.1, 0.8, 0.3, 0.1]))
    n = write_stage2_features(con, tmp_path / "feat.parquet", tmp_path / "pred.parquet", tmp_path / "s2.parquet")
    out = pq.read_table(tmp_path / "s2.parquet")
    assert n == 6 and out.num_rows == 6
    assert set(MODEL2_FEATURES + ["lid", "rid", "label"]) == set(out.column_names)
    first = feat["rid"][0].as_py()
    ctx = pa.table({"lid": ["S1-zzz"], "rid": [first], "p": pa.array([0.99], pa.float32())})
    pq.write_table(ctx, tmp_path / "ctx.parquet")
    write_stage2_features(
        con, tmp_path / "feat.parquet", tmp_path / "pred.parquet", tmp_path / "s2c.parquet", tmp_path / "ctx.parquet"
    )
    with_ctx = pq.read_table(tmp_path / "s2c.parquet")
    assert with_ctx.num_rows == 6
    row = con.execute(
        f"SELECT s2_r_is_best, s2_r_rank, s2_l_max FROM read_parquet('{tmp_path / 's2c.parquet'}') WHERE rid = ?",
        [first],
    ).fetchone()
    assert row == (0.0, 2.0, pytest.approx(0.9))


def test_write_stage2_features_rejects_missing_predictions(con, tmp_path: Path) -> None:
    """A feature row without a stage-1 prediction is an error, not a silent drop."""
    rng = np.random.default_rng(0)
    feat = _write_feat(tmp_path / "feat.parquet", ["S1-a", "S1-b"], rng)
    _write_pred(tmp_path / "pred.parquet", feat.slice(0, 5), np.full(5, 0.5))
    with pytest.raises(ValueError, match="6 feature rows"):
        write_stage2_features(con, tmp_path / "feat.parquet", tmp_path / "pred.parquet", tmp_path / "s2.parquet")


def _recording_fns(scored: list[tuple[object, set[str]]]):
    """Fake train/predict functions: a 'booster' is the set of S1 ids it was trained on (or the string FULL)."""

    def fake_train(path: Path, valid_path: Path, params: dict, seed: int, threads: int) -> frozenset[str]:
        """Return the training S1 ids as the model."""
        return frozenset(pq.read_table(path, columns=["lid"]).column("lid").to_pylist())

    def fake_predict(booster: object, feat_path: Path, out_path: Path, batch_rows: int) -> int:
        """Record which S1 ids this model scored and write a constant prediction."""
        tbl = pq.read_table(feat_path, columns=["lid", "rid", "label"])
        scored.append((booster, set(tbl.column("lid").to_pylist())))
        _write_pred(out_path, tbl, np.full(tbl.num_rows, 0.5))
        return tbl.num_rows

    return fake_train, fake_predict


def test_oof_never_scores_with_a_model_that_saw_the_fold(con, tmp_path: Path) -> None:
    """Every fit pair is scored exactly once, by a model trained on none of its S1's pairs (folds by S1)."""
    rng = np.random.default_rng(1)
    fit_ids = _ids_in_group(con, "fit", 40)
    feat = _write_feat(tmp_path / "fit.parquet", fit_ids, rng)
    _write_feat(tmp_path / "tune.parquet", _ids_in_group(con, "tune", 5), rng)
    scored: list[tuple[object, set[str]]] = []
    fake_train, fake_predict = _recording_fns(scored)
    boosters = oof_predict(
        con,
        tmp_path / "fit.parquet",
        tmp_path / "tune.parquet",
        tmp_path / "oof.parquet",
        tmp_path / "work",
        {},
        1,
        1,
        1000,
        train_fn=fake_train,
        predict_fn=fake_predict,
    )
    assert len(boosters) == 2 and all(boosters)
    assert len(scored) == 2
    for booster, lids in scored:
        assert lids and not (booster & lids)
        assert booster | lids == set(fit_ids)
    out = pq.read_table(tmp_path / "oof.parquet")
    assert sorted(zip(out["lid"].to_pylist(), out["rid"].to_pylist(), strict=True)) == sorted(
        zip(feat["lid"].to_pylist(), feat["rid"].to_pylist(), strict=True)
    )
    folds = dict(
        con.execute(f"SELECT id, {fold_sql('id')} FROM (SELECT unnest(?::VARCHAR[]) AS id)", [fit_ids]).fetchall()
    )
    for booster, _ in scored:
        assert len({folds[i] for i in booster}) == 1


def test_predict_protocol_routes_rows(con, tmp_path: Path) -> None:
    """Fit-group rows go to the fold model that excludes their fold; other rows to the full model."""
    rng = np.random.default_rng(2)
    fit_ids, tune_ids = _ids_in_group(con, "fit", 20), _ids_in_group(con, "tune", 6)
    _write_feat(tmp_path / "fit.parquet", fit_ids, rng)
    _write_feat(tmp_path / "mixed.parquet", fit_ids[:10] + tune_ids, rng)
    _write_feat(tmp_path / "tune.parquet", tune_ids, rng)
    scored: list[tuple[object, set[str]]] = []
    fake_train, fake_predict = _recording_fns(scored)
    folds = oof_predict(
        con,
        tmp_path / "fit.parquet",
        tmp_path / "tune.parquet",
        tmp_path / "oof.parquet",
        tmp_path / "w",
        {},
        1,
        1,
        1000,
        train_fn=fake_train,
        predict_fn=fake_predict,
    )
    scored.clear()
    n = predict_protocol(
        con,
        tmp_path / "mixed.parquet",
        tmp_path / "m.parquet",
        folds,
        "FULL",
        1000,
        tmp_path / "w",
        predict_fn=fake_predict,
    )
    assert n == 3 * 16
    full = [lids for b, lids in scored if b == "FULL"]
    assert full == [set(tune_ids)]
    for b, lids in scored:
        if b != "FULL":
            assert lids <= set(fit_ids[:10]) and not (b & lids)
    scored.clear()
    predict_protocol(
        con,
        tmp_path / "tune.parquet",
        tmp_path / "t.parquet",
        folds,
        "FULL",
        1000,
        tmp_path / "w",
        predict_fn=fake_predict,
    )
    assert scored == [("FULL", set(tune_ids))]
    with pytest.raises(ValueError, match="full_booster"):
        predict_protocol(con, tmp_path / "tune.parquet", tmp_path / "t.parquet", folds, None, 1000, tmp_path / "w")


def test_end_to_end_with_lightgbm(con, tmp_path: Path) -> None:
    """Real LightGBM: OOF stage 1 -> stage-2 matrix -> stage-2 model on MODEL2_FEATURES -> predictions."""
    rng = np.random.default_rng(4)
    _write_feat(tmp_path / "fit.parquet", _ids_in_group(con, "fit", 300), rng, shared=20)
    tune = _write_feat(tmp_path / "tune.parquet", _ids_in_group(con, "tune", 150), rng)
    params = {"min_data_in_leaf": 5, "num_leaves": 7, "learning_rate": 0.3}
    oof_predict(
        con,
        tmp_path / "fit.parquet",
        tmp_path / "tune.parquet",
        tmp_path / "p1_fit.parquet",
        tmp_path / "w",
        params,
        1,
        1,
        500,
    )
    b1 = train(tmp_path / "fit.parquet", tmp_path / "tune.parquet", params, 1, 1, rounds=50)
    predict_file(b1, tmp_path / "tune.parquet", tmp_path / "p1_tune.parquet", 500)
    for g in ("fit", "tune"):
        write_stage2_features(
            con, tmp_path / f"{g}.parquet", tmp_path / f"p1_{g}.parquet", tmp_path / f"s2_{g}.parquet"
        )
    b2 = train(
        tmp_path / "s2_fit.parquet", tmp_path / "s2_tune.parquet", params, 1, 1, rounds=50, features=MODEL2_FEATURES
    )
    assert b2.feature_name() == MODEL2_FEATURES
    n = predict_file(b2, tmp_path / "s2_tune.parquet", tmp_path / "p2_tune.parquet", 500, features=MODEL2_FEATURES)
    assert n == tune.num_rows
    pred = pq.read_table(tmp_path / "p2_tune.parquet")
    p, y = pred["p"].to_numpy(), pred["label"].to_numpy()
    assert p[y == 1].mean() > p[y == 0].mean() + 0.5
