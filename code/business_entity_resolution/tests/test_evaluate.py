"""Tests for ber.evaluate (official macro F0.5 and blocking stats)."""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

from ber.evaluate import blocking_stats, f05, macro_f05, per_entity_scores


def pairs(rows: list[tuple[str, str]]) -> pa.Table:
    """Build an (s1_id, match_id) pyarrow table."""
    return pa.table(
        {"s1_id": [r[0] for r in rows], "match_id": [r[1] for r in rows]},
        schema=pa.schema([("s1_id", pa.string()), ("match_id", pa.string())]),
    )


def ids(values: list[str]) -> pa.Table:
    """Build an (s1_id) pyarrow table."""
    return pa.table({"s1_id": pa.array(values, pa.string())})


def test_organizer_example(con) -> None:
    """Pred [47,193,812] vs truth [47,812] -> P=2/3, R=1 -> 0.714285..."""
    pred = pairs([("S1-00001", "S2-00047"), ("S1-00001", "S2-00193"), ("S1-00001", "S3-00812")])
    gt = pairs([("S1-00001", "S2-00047"), ("S1-00001", "S3-00812")])
    res = macro_f05(con, pred, gt, ids(["S1-00001"]))
    assert res["macro_f05"] == pytest.approx(5 / 7, abs=1e-12)
    assert round(res["macro_f05"], 3) == 0.714
    assert f05(2 / 3, 1.0) == pytest.approx(5 / 7)


def test_singleton_rules(con) -> None:
    """No-GT entity: 1.0 for empty prediction, 0.0 for any prediction."""
    gt = pairs([])
    assert macro_f05(con, pairs([]), gt, ids(["S1-1"]))["macro_f05"] == 1.0
    res = macro_f05(con, pairs([("S1-1", "S2-9")]), gt, ids(["S1-1"]))
    assert res["macro_f05"] == 0.0
    assert res["singleton_accuracy"] == 0.0 and res["n_singletons"] == 1


def test_matched_entity_empty_prediction_scores_zero(con) -> None:
    """Entity with GT matches but nothing (or nothing correct) predicted scores 0."""
    gt = pairs([("S1-1", "S2-1")])
    assert macro_f05(con, pairs([]), gt, ids(["S1-1"]))["macro_f05"] == 0.0
    assert macro_f05(con, pairs([("S1-1", "S3-5")]), gt, ids(["S1-1"]))["macro_f05"] == 0.0


def test_macro_average_mixed(con) -> None:
    """Average over ALL s1_ids, including ones absent from both tables; out-of-population pairs ignored."""
    gt = pairs([("S1-a", "S2-1"), ("S1-a", "S3-1"), ("S1-b", "S2-2"), ("S1-z", "S2-9")])
    pred = pairs(
        [
            ("S1-a", "S2-1"),  # P=1, R=0.5 -> 1.25*.5/(.25+.5)=0.8333
            ("S1-b", "S2-2"),
            ("S1-b", "S2-2"),  # duplicate pair counted once -> 1.0
            ("S1-c", "S2-7"),  # singleton with a false match -> 0
            ("S1-z", "S2-9"),  # not in population -> ignored
        ]
    )
    res = macro_f05(con, pred, gt, ids(["S1-a", "S1-b", "S1-c", "S1-d"]))  # S1-d: singleton, empty -> 1
    expected = (f05(1.0, 0.5) + 1.0 + 0.0 + 1.0) / 4
    assert res["macro_f05"] == pytest.approx(expected)
    assert res["n_entities"] == 4 and res["n_singletons"] == 2
    assert res["singleton_accuracy"] == pytest.approx(0.5)
    assert res["f05_non_singletons"] == pytest.approx((f05(1.0, 0.5) + 1.0) / 2)
    assert res["micro_precision"] == pytest.approx(2 / 3)
    assert res["micro_recall"] == pytest.approx(2 / 3)


def test_inputs_as_relation_and_table_name(con) -> None:
    """Relations and table names are accepted just like pyarrow tables."""
    con.execute("CREATE TABLE gt AS SELECT 'S1-1' AS s1_id, 'S2-1' AS match_id")
    pred_rel = con.sql("SELECT 'S1-1' AS s1_id, 'S2-1' AS match_id")
    assert macro_f05(con, pred_rel, "gt", ids(["S1-1"]))["macro_f05"] == 1.0
    scores = per_entity_scores(con, pred_rel, "gt", ids(["S1-1"])).fetchall()
    assert scores == [("S1-1", 1, 1, 1, 1.0)]


def test_missing_column_raises(con) -> None:
    """A pair table without match_id is rejected clearly."""
    with pytest.raises(ValueError, match="missing columns"):
        macro_f05(con, pa.table({"s1_id": ["S1-1"]}), pairs([]), ids(["S1-1"]))


def test_blocking_stats(con) -> None:
    """Pair completeness, entity full recall, per-S1 distribution and reduction ratio."""
    gt = pairs([("S1-a", "S2-1"), ("S1-a", "S3-1"), ("S1-b", "S2-2")])
    cand = pairs([("S1-a", "S2-1"), ("S1-a", "S2-5"), ("S1-b", "S2-2"), ("S1-b", "S3-3"), ("S1-b", "S3-4")])
    st = blocking_stats(con, cand, gt, ids(["S1-a", "S1-b", "S1-c"]), n_right_records=10)
    assert st["pair_completeness"] == pytest.approx(2 / 3)
    assert st["entity_full_recall"] == pytest.approx(1 / 2)
    assert st["total_pairs"] == 5
    assert st["cands_per_s1_mean"] == pytest.approx(5 / 3)
    assert st["cands_per_s1_max"] == 3 and st["cands_per_s1_p50"] == 2
    assert st["reduction_ratio"] == pytest.approx(1 - 5 / 30)
    assert not math.isnan(st["reduction_ratio"])
