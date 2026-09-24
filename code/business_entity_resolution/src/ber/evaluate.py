"""Official metric (macro F0.5 per Source-1 entity) and blocking diagnostics, computed in DuckDB SQL.

Input convention (see ``ber.duck.TableLike``): every table argument may be a table/view name in ``con``,
a relation of ``con``, or a pyarrow Table.
- pair tables (predictions, ground truth, candidates): columns ``s1_id``, ``match_id``
- ``s1_ids``: column ``s1_id`` — the evaluation population; pairs for other S1 ids are ignored.
Pairs are de-duplicated before scoring (the metric works on sets).
"""

from __future__ import annotations

from typing import Any

import duckdb

from ber.duck import TableLike, as_view, drop_views

PAIR_COLS = ("s1_id", "match_id")
S1_COLS = ("s1_id",)


def f05(precision: float, recall: float) -> float:
    """F-beta with beta=0.5: 1.25PR / (0.25P + R); 0 when both are 0."""
    denom = 0.25 * precision + recall
    return 0.0 if denom == 0 else 1.25 * precision * recall / denom


def _per_entity_sql(s1: str, pred: str, gt: str) -> str:
    """SQL producing one row per distinct S1 id with n_gt, n_pred, n_tp and its F0.5."""
    return f"""
    WITH s1 AS (SELECT DISTINCT s1_id FROM {s1}),
    gt AS (SELECT DISTINCT s1_id, match_id FROM {gt} WHERE s1_id IN (SELECT s1_id FROM s1)),
    pr AS (SELECT DISTINCT s1_id, match_id FROM {pred} WHERE s1_id IN (SELECT s1_id FROM s1)),
    gt_n AS (SELECT s1_id, count(*) AS n_gt FROM gt GROUP BY s1_id),
    pr_n AS (SELECT s1_id, count(*) AS n_pred FROM pr GROUP BY s1_id),
    tp_n AS (SELECT s1_id, count(*) AS n_tp FROM gt JOIN pr USING (s1_id, match_id) GROUP BY s1_id),
    counts AS (
        SELECT s1.s1_id, coalesce(n_gt, 0) AS n_gt, coalesce(n_pred, 0) AS n_pred, coalesce(n_tp, 0) AS n_tp
        FROM s1 LEFT JOIN gt_n USING (s1_id) LEFT JOIN pr_n USING (s1_id) LEFT JOIN tp_n USING (s1_id)
    )
    SELECT *,
        CASE
            WHEN n_gt = 0 THEN CASE WHEN n_pred = 0 THEN 1.0 ELSE 0.0 END
            WHEN n_tp = 0 THEN 0.0
            ELSE 1.25 * (n_tp / n_pred) * (n_tp / n_gt) / (0.25 * (n_tp / n_pred) + (n_tp / n_gt))
        END AS f05
    FROM counts
    """


def per_entity_scores(
    con: duckdb.DuckDBPyConnection, pred_pairs: TableLike, gt_pairs: TableLike, s1_ids: TableLike
) -> duckdb.DuckDBPyRelation:
    """Return a materialized relation (s1_id, n_gt, n_pred, n_tp, f05) for error analysis.

    The result is stored in a temp table so it stays valid after the input views are dropped.
    """
    views = [
        as_view(con, pred_pairs, PAIR_COLS),
        as_view(con, gt_pairs, PAIR_COLS),
        as_view(con, s1_ids, S1_COLS),
    ]
    try:
        name = views[0] + "_scores"
        con.execute(f"CREATE OR REPLACE TEMP TABLE {name} AS {_per_entity_sql(views[2], views[0], views[1])}")
        return con.table(name)
    finally:
        drop_views(con, *views)


def macro_f05(
    con: duckdb.DuckDBPyConnection, pred_pairs: TableLike, gt_pairs: TableLike, s1_ids: TableLike
) -> dict[str, Any]:
    """Compute the official macro F0.5 over all ``s1_ids`` plus diagnostics.

    Per entity: no GT -> 1.0 iff prediction empty else 0.0; GT but empty/no-hit prediction -> 0.0;
    otherwise F0.5 of set precision/recall. Returns macro_f05, micro_precision, micro_recall,
    singleton_accuracy, n_entities, n_singletons, f05_singletons, f05_non_singletons, n_pred_pairs,
    n_gt_pairs, n_tp_pairs.
    """
    views = [
        as_view(con, pred_pairs, PAIR_COLS),
        as_view(con, gt_pairs, PAIR_COLS),
        as_view(con, s1_ids, S1_COLS),
    ]
    try:
        row = con.execute(
            f"""
            SELECT avg(f05), sum(n_tp), sum(n_pred), sum(n_gt), count(*), count_if(n_gt = 0),
                   avg(f05) FILTER (WHERE n_gt = 0), avg(f05) FILTER (WHERE n_gt > 0)
            FROM ({_per_entity_sql(views[2], views[0], views[1])})
            """
        ).fetchone()
    finally:
        drop_views(con, *views)
    macro, tp, npred, ngt, n, nsing, f_sing, f_non = row
    tp, npred, ngt = int(tp or 0), int(npred or 0), int(ngt or 0)
    return {
        "macro_f05": float(macro) if macro is not None else float("nan"),
        "micro_precision": tp / npred if npred else float("nan"),
        "micro_recall": tp / ngt if ngt else float("nan"),
        "singleton_accuracy": float(f_sing) if f_sing is not None else float("nan"),
        "f05_singletons": float(f_sing) if f_sing is not None else float("nan"),
        "f05_non_singletons": float(f_non) if f_non is not None else float("nan"),
        "n_entities": int(n),
        "n_singletons": int(nsing),
        "n_pred_pairs": npred,
        "n_gt_pairs": ngt,
        "n_tp_pairs": tp,
    }


def blocking_stats(
    con: duckdb.DuckDBPyConnection,
    cand_pairs: TableLike,
    gt_pairs: TableLike,
    s1_ids: TableLike,
    n_right_records: int,
) -> dict[str, Any]:
    """Blocking quality for the S1 ids in ``s1_ids``.

    Returns pair_completeness (GT pairs found in candidates / GT pairs), entity_full_recall (share of
    matched S1 entities with every GT match in candidates), candidates-per-S1 mean/p50/p90/p99/max
    (entities with no candidates count as 0), total_pairs, and reduction_ratio =
    1 - total_pairs / (n_s1 * n_right_records).
    """
    views = [
        as_view(con, cand_pairs, PAIR_COLS),
        as_view(con, gt_pairs, PAIR_COLS),
        as_view(con, s1_ids, S1_COLS),
    ]
    cand, gt, s1 = views
    try:
        row = con.execute(
            f"""
            WITH s1 AS (SELECT DISTINCT s1_id FROM {s1}),
            c AS (SELECT DISTINCT s1_id, match_id FROM {cand} WHERE s1_id IN (SELECT s1_id FROM s1)),
            g AS (SELECT DISTINCT s1_id, match_id FROM {gt} WHERE s1_id IN (SELECT s1_id FROM s1)),
            hit AS (SELECT g.s1_id, (c.s1_id IS NOT NULL) AS found
                    FROM g LEFT JOIN c USING (s1_id, match_id)),
            ent AS (SELECT s1_id, bool_and(found) AS all_found FROM hit GROUP BY s1_id),
            per AS (SELECT s1.s1_id, coalesce(k.n, 0) AS n
                    FROM s1 LEFT JOIN (SELECT s1_id, count(*) AS n FROM c GROUP BY s1_id) k USING (s1_id))
            SELECT
                (SELECT count(*) FROM hit), (SELECT count_if(found) FROM hit),
                (SELECT count(*) FROM ent), (SELECT count_if(all_found) FROM ent),
                count(*), sum(n), avg(n),
                quantile_disc(n, 0.5), quantile_disc(n, 0.9), quantile_disc(n, 0.99), max(n)
            FROM per
            """
        ).fetchone()
    finally:
        drop_views(con, *views)
    n_gt, n_found, n_ent, n_ent_full, n_s1, total, mean, p50, p90, p99, mx = row
    total = int(total or 0)
    space = int(n_s1) * int(n_right_records)
    return {
        "pair_completeness": n_found / n_gt if n_gt else float("nan"),
        "entity_full_recall": n_ent_full / n_ent if n_ent else float("nan"),
        "n_gt_pairs": int(n_gt),
        "n_gt_pairs_found": int(n_found),
        "n_s1": int(n_s1),
        "total_pairs": total,
        "cands_per_s1_mean": float(mean) if mean is not None else 0.0,
        "cands_per_s1_p50": int(p50 or 0),
        "cands_per_s1_p90": int(p90 or 0),
        "cands_per_s1_p99": int(p99 or 0),
        "cands_per_s1_max": int(mx or 0),
        "reduction_ratio": 1.0 - total / space if space else float("nan"),
    }
