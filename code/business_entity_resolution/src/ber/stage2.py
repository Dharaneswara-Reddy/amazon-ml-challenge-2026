"""Stage 2: re-score every pair with stage-1 features plus "competition" features computed from stage-1 probabilities.

Why: stage 1 scores each pair in isolation, but a pair's chance also depends on its rivals. A right record (R)
matches at most one Source 1 entity (S1), so a 0.8 that loses to a 0.99 elsewhere is weaker than a lone 0.8; an S1
that already has three 0.99 matches from S2 is less likely to have a fourth; a lone mid-score candidate on an S1
with nothing else is usually a singleton's false match. Stage 2 learns these patterns (``STAGE2_FEATURES``).

Leakage protocol (stacking). Stage-2 training rows must carry p1 from a model that never saw their labels:
  * fit group (stage-2 TRAINING rows): its S1 ids are split into ``N_FOLDS`` folds by ``fold_sql`` (md5 digits
    above the group bucket, so each group splits evenly). For fold k a stage-1 model is trained on the fit rows of
    the other folds (early-stopped on the tune group) and predicts fold k (``oof_predict``).
  * tune / holdout / test: p1 from the stage-1 model trained on the full fit group (never saw those S1s).
  * ``predict_protocol`` applies this rule row by row to any train-split feature file (fit rows -> fold model not
    trained on their fold, all other rows -> full model), e.g. for the optional rival ("context") files below.

Rival completeness. Train feature files hold only pairs whose S1 is in the group, so an R's rivals from other groups
are missing at train time, while test has all of them; R-level features (``R_LEVEL_FEATURES``) are therefore
computed from fewer rivals in training. The conditional signal is still valid (in-group rivals are a random subset),
so this works as is; for full fidelity write rival pairs with ``features.write_features(..., where_sql=
context_where(group))``, score them with ``predict_protocol`` and pass them as ``context_pred_path``.

Scale (~50M pairs): ``write_stage2_features`` is one DuckDB statement (window aggregates over the narrow
(lid, rid, p) predictions, then a hash join back to the wide feature parquet), streamed to parquet by COPY. It runs
under the connection's memory_limit and spills to temp_directory (ber.duck.connect), so memory stays bounded;
disk needs roughly the size of the feature parquet. ``competition_features`` is the numpy reference (O(pairs)
memory, ~20 float64 arrays per pair), used by tests and small data.

Integration (the lead wires this into pipeline.py; FEAT(g) = pipeline.feat_path(cfg, "train", g)):
  train_stage:
    1. booster1 = model.train(FEAT(fit), FEAT(tune), params, seed, threads)            (unchanged; save model.txt)
    2. folds = oof_predict(con, FEAT(fit), FEAT(tune), run/"pred1_fit.parquet", run, params, seed, threads,
       batch_rows)                                                                     (OOF p1 for the fit group)
    3. model.predict_file(booster1, FEAT(g), run/f"pred1_{g}.parquet", batch_rows) for g in tune, holdout
    4. write_stage2_features(con, FEAT(g), run/f"pred1_{g}.parquet", run/f"s2_{g}.parquet") for fit, tune, holdout
    5. booster2 = model.train(s2_fit, s2_tune, params, seed, threads, features=MODEL2_FEATURES); save model2.txt
    6. model.predict_file(booster2, s2_g, run/f"pred2_{g}.parquet", batch_rows, features=MODEL2_FEATURES)
       for g in tune, holdout
    7. report = decision.compare_decisions(_group_arrays(con, cfg, pred2_tune, "tune"),
       _group_arrays(con, cfg, pred2_holdout, "holdout")); store report["params"] in the manifest
       (also run compare_decisions on the pred1 files to measure what stage 2 adds).
  predict_stage:
    p1 = predict_file(booster1, FEAT(test)) -> write_stage2_features(test) -> p2 = predict_file(booster2, ...,
    features=MODEL2_FEATURES) -> mask = decision.decide(params, lid_idx, rid, p2) with lid_idx/rid as integer
    codes (DuckDB dense_rank() - 1) -> matches table (s1_id, match_id) for writer.write_outputs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq

from ber.decision import best_per_right
from ber.features import FEATURES
from ber.model import predict_file, train
from ber.split import group_where

log = logging.getLogger(__name__)

N_FOLDS = 2
MUTUAL_K = 4
HI, VHI = 0.5, 0.9

R_LEVEL_FEATURES = ["s2_r_rank", "s2_r_margin", "s2_r_is_best", "s2_r_share", "s2_l_n50_best", "s2_mutual"]
STAGE2_FEATURES = [
    "s2_p1",
    "s2_r_rank",
    "s2_r_margin",
    "s2_r_is_best",
    "s2_r_share",
    "s2_l_rank",
    "s2_l_gap",
    "s2_l_max",
    "s2_l_sum",
    "s2_l_n50",
    "s2_l_n90",
    "s2_l_n50_best",
    "s2_mutual",
    "s2_l_n50_s2",
    "s2_l_n50_s3",
    "s2_l_n90_s2",
    "s2_l_n90_s3",
]

# Entity-level context from stage-1 scores (D-027; measured +0.0003 holdout, research_notes/v5/K_india_us.md T3):
# the pair's standing among its S1 entity's candidates and its right record's alternatives.
ENTITY_FEATURES = [
    "e_l_n_best", "e_l_sum_best", "e_l_rank_best", "e_l_next_gap", "e_l_prev_gap", "e_l_n10", "e_l_n30",
    "e_ls_rank", "e_ls_n50", "e_ls_max_other", "e_r_n", "e_r_second", "e_r_other_lgap", "e_r_other_ln50", "e_p_over_lmax",
]


def entity_sql(preds: str) -> str:
    """SQL of (lid, rid, ENTITY_FEATURES) over a stage-1 prediction relation ``preds`` (lid, rid, p; all rows)."""
    return f"""
    WITH b AS (SELECT lid, rid, p::DOUBLE AS p, starts_with(rid, 'S3-') AS s3,
                 row_number() OVER (PARTITION BY rid ORDER BY p DESC, lid) AS r_rn,
                 count(*) OVER (PARTITION BY rid) AS r_n
               FROM {preds}),
    rt AS (SELECT rid, max(CASE WHEN r_rn = 1 THEN lid END) AS l1, max(CASE WHEN r_rn = 1 THEN p END) AS p1,
                  max(CASE WHEN r_rn = 2 THEN lid END) AS l2, max(CASE WHEN r_rn = 2 THEN p END) AS p2
           FROM b WHERE r_rn <= 2 GROUP BY rid),
    lt AS (SELECT lid, max(p) AS lmax, sum((p >= 0.5)::INT) AS ln50 FROM b GROUP BY lid),
    c AS (SELECT b.*, r_rn = 1 AS best,
            sum(CASE WHEN r_rn = 1 THEN 1 ELSE 0 END) OVER wl AS n_best,
            sum(CASE WHEN r_rn = 1 THEN p ELSE 0 END) OVER wl AS sum_best,
            sum((p >= 0.1)::INT) OVER wl AS n10, sum((p >= 0.3)::INT) OVER wl AS n30,
            lead(p) OVER (PARTITION BY lid ORDER BY p DESC, rid) AS nxt,
            lag(p) OVER (PARTITION BY lid ORDER BY p DESC, rid) AS prv,
            rank() OVER (PARTITION BY lid, s3 ORDER BY p DESC) AS ls_rank,
            sum((p >= 0.5)::INT) OVER (PARTITION BY lid, s3) AS ls_n50,
            max(p) OVER (PARTITION BY lid) AS lmax_all
          FROM b WINDOW wl AS (PARTITION BY lid)),
    bestp AS (SELECT lid, list(p) AS bp FROM c WHERE best GROUP BY lid),
    ls2 AS (SELECT lid, s3, max(p) AS m1, arg_max(rid, p) AS a1 FROM c GROUP BY lid, s3)
    SELECT c.lid, c.rid,
      n_best::FLOAT AS e_l_n_best, sum_best::FLOAT AS e_l_sum_best,
      (1 + len(list_filter(coalesce(bp, []), x -> x > c.p)))::FLOAT AS e_l_rank_best,
      (c.p - coalesce(nxt, 0))::FLOAT AS e_l_next_gap, (coalesce(prv, c.p) - c.p)::FLOAT AS e_l_prev_gap,
      n10::FLOAT AS e_l_n10, n30::FLOAT AS e_l_n30, ls_rank::FLOAT AS e_ls_rank, ls_n50::FLOAT AS e_ls_n50,
      (CASE WHEN ls2.a1 = c.rid THEN -1 ELSE ls2.m1 END)::FLOAT AS e_ls_max_other,
      r_n::FLOAT AS e_r_n,
      (CASE WHEN best THEN coalesce(rt.p2, 0) ELSE rt.p1 END)::FLOAT AS e_r_second,
      (CASE WHEN best THEN coalesce(l2t.lmax - rt.p2, -1) ELSE l1t.lmax - rt.p1 END)::FLOAT AS e_r_other_lgap,
      (CASE WHEN best THEN coalesce(l2t.ln50, -1) ELSE l1t.ln50 END)::FLOAT AS e_r_other_ln50,
      (c.p / greatest(lmax_all, 1e-9))::FLOAT AS e_p_over_lmax
    FROM c JOIN rt ON rt.rid = c.rid LEFT JOIN lt l1t ON l1t.lid = rt.l1 LEFT JOIN lt l2t ON l2t.lid = rt.l2
    LEFT JOIN bestp ON bestp.lid = c.lid JOIN ls2 ON ls2.lid = c.lid AND ls2.s3 = c.s3"""

MODEL2_FEATURES = FEATURES + STAGE2_FEATURES + ENTITY_FEATURES

TrainFn = Callable[..., Any]
PredictFn = Callable[..., int]


# --------------------------------------------------------------------------------------------------------------
# Competition features
# --------------------------------------------------------------------------------------------------------------


def _rank_desc(g: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Competition rank of p within each group code g (1 = highest; ties share a rank, like SQL rank())."""
    n = len(p)
    order = np.lexsort((-p, g))
    gs, ps = g[order], p[order]
    idx = np.arange(n)
    gstart = np.r_[True, gs[1:] != gs[:-1]] if n else np.zeros(0, dtype=bool)
    newval = gstart | (np.r_[True, ps[1:] != ps[:-1]] if n else gstart)
    first_run = np.maximum.accumulate(np.where(newval, idx, 0)) if n else idx
    first_grp = np.maximum.accumulate(np.where(gstart, idx, 0)) if n else idx
    rank = np.empty(n, dtype=np.int64)
    rank[order] = first_run - first_grp + 1
    return rank


def competition_features(
    lid: np.ndarray,
    rid: np.ndarray,
    p: np.ndarray,
    r_is_s3: np.ndarray | None = None,
    mutual_k: int = MUTUAL_K,
) -> dict[str, np.ndarray]:
    """Numpy reference of the stage-2 features for pairs (lid, rid) with stage-1 probability p.

    lid/rid: any sortable id arrays; r_is_s3: bool per pair (default: rid starts with "S3-"). Returns a dict with one
    float64 array per name in STAGE2_FEATURES, aligned with the input. Definitions (R = right record, L = S1):
      s2_p1 p; s2_r_rank rank of p among R's candidates; s2_r_margin p minus R's best OTHER candidate (0 if none);
      s2_r_is_best R's top pair (ties -> smaller lid); s2_r_share p / sum of R's p; s2_l_rank rank among L's
      candidates; s2_l_gap L's max p minus p; s2_l_max, s2_l_sum; s2_l_n50 / s2_l_n90 L's candidates with p >= 0.5 /
      0.9; s2_l_n50_best those >= 0.5 that are also R-best; s2_mutual R-best and s2_l_rank <= mutual_k;
      s2_l_n{50,90}_s{2,3} the same counts per source of the right record.
    """
    p = np.asarray(p, dtype=np.float64)
    _, lc = np.unique(np.asarray(lid), return_inverse=True)
    _, rc = np.unique(np.asarray(rid), return_inverse=True)
    lc, rc = lc.ravel(), rc.ravel()
    s3 = np.char.startswith(np.asarray(rid).astype(str), "S3-") if r_is_s3 is None else np.asarray(r_is_s3, bool)
    nl, nr = (int(lc.max()) + 1, int(rc.max()) + 1) if len(p) else (0, 0)

    best = best_per_right(rc, lc, p)
    r_max = np.zeros(nr)
    np.maximum.at(r_max, rc, p)
    r_other = np.zeros(nr)
    np.maximum.at(r_other, rc[~best], p[~best])
    r_sum = np.bincount(rc, weights=p, minlength=nr)[rc]
    l_rank = _rank_desc(lc, p)
    l_max = np.zeros(nl)
    np.maximum.at(l_max, lc, p)

    def per_l(m: np.ndarray) -> np.ndarray:
        """Count of pairs with mask m per S1, broadcast back to pairs."""
        return np.bincount(lc, weights=m.astype(np.float64), minlength=nl)[lc]

    hi, vhi = p >= HI, p >= VHI
    return {
        "s2_p1": p,
        "s2_r_rank": _rank_desc(rc, p).astype(np.float64),
        "s2_r_margin": np.where(best, p - r_other[rc], p - r_max[rc]),
        "s2_r_is_best": best.astype(np.float64),
        "s2_r_share": np.divide(p, r_sum, out=np.zeros_like(p), where=r_sum > 0),
        "s2_l_rank": l_rank.astype(np.float64),
        "s2_l_gap": l_max[lc] - p,
        "s2_l_max": l_max[lc],
        "s2_l_sum": np.bincount(lc, weights=p, minlength=nl)[lc],
        "s2_l_n50": per_l(hi),
        "s2_l_n90": per_l(vhi),
        "s2_l_n50_best": per_l(hi & best),
        "s2_mutual": (best & (l_rank <= mutual_k)).astype(np.float64),
        "s2_l_n50_s2": per_l(hi & ~s3),
        "s2_l_n50_s3": per_l(hi & s3),
        "s2_l_n90_s2": per_l(vhi & ~s3),
        "s2_l_n90_s3": per_l(vhi & s3),
    }


def competition_sql(src: str, mutual_k: int = MUTUAL_K) -> str:
    """DuckDB SELECT computing (lid, rid, *STAGE2_FEATURES) over relation ``src`` with columns lid, rid, p.

    ``src`` is a table/view name or a parenthesized subquery. Same definitions as ``competition_features``; all
    features are FLOAT. Windows partition by rid, then by lid, so DuckDB sorts/spills within its memory_limit.
    """
    return f"""
    WITH b AS (
      SELECT lid, rid, p::DOUBLE AS p, starts_with(rid, 'S3-') AS s3,
        rank() OVER (PARTITION BY rid ORDER BY p DESC) AS r_rank,
        row_number() OVER (PARTITION BY rid ORDER BY p DESC, lid) AS r_rn,
        lead(p::DOUBLE) OVER (PARTITION BY rid ORDER BY p DESC, lid) AS r_next,
        max(p::DOUBLE) OVER (PARTITION BY rid) AS r_max,
        sum(p::DOUBLE) OVER (PARTITION BY rid) AS r_sum
      FROM {src} AS src),
    c AS (
      SELECT *, r_rn = 1 AS r_best, rank() OVER (PARTITION BY lid ORDER BY p DESC) AS l_rank FROM b)
    SELECT lid, rid,
      p::FLOAT AS s2_p1,
      r_rank::FLOAT AS s2_r_rank,
      (p - CASE WHEN r_best THEN coalesce(r_next, 0) ELSE r_max END)::FLOAT AS s2_r_margin,
      r_best::FLOAT AS s2_r_is_best,
      (CASE WHEN r_sum > 0 THEN p / r_sum ELSE 0 END)::FLOAT AS s2_r_share,
      l_rank::FLOAT AS s2_l_rank,
      (max(p) OVER wl - p)::FLOAT AS s2_l_gap,
      (max(p) OVER wl)::FLOAT AS s2_l_max,
      (sum(p) OVER wl)::FLOAT AS s2_l_sum,
      (sum((p >= {HI})::INT) OVER wl)::FLOAT AS s2_l_n50,
      (sum((p >= {VHI})::INT) OVER wl)::FLOAT AS s2_l_n90,
      (sum((p >= {HI} AND r_best)::INT) OVER wl)::FLOAT AS s2_l_n50_best,
      (r_best AND l_rank <= {int(mutual_k)})::FLOAT AS s2_mutual,
      (sum((p >= {HI} AND NOT s3)::INT) OVER wl)::FLOAT AS s2_l_n50_s2,
      (sum((p >= {HI} AND s3)::INT) OVER wl)::FLOAT AS s2_l_n50_s3,
      (sum((p >= {VHI} AND NOT s3)::INT) OVER wl)::FLOAT AS s2_l_n90_s2,
      (sum((p >= {VHI} AND s3)::INT) OVER wl)::FLOAT AS s2_l_n90_s3
    FROM c
    WINDOW wl AS (PARTITION BY lid)
    """


def write_stage2_features(
    con: duckdb.DuckDBPyConnection,
    feat_path: Path,
    pred_path: Path,
    out_path: Path,
    context_pred_path: Path | None = None,
    mutual_k: int = MUTUAL_K,
) -> int:
    """Write the stage-2 matrix: every row of ``feat_path`` plus STAGE2_FEATURES from ``pred_path``; return rows.

    feat_path: stage-1 feature parquet (lid, rid, FEATURES[, label]). pred_path: stage-1 predictions (lid, rid, p) for
    the same pairs. context_pred_path: optional predictions for extra rival pairs (not in feat_path) that share an R
    with them; they join the windows but are not output. Raises ValueError if some feature row has no prediction.
    """
    preds = f"(SELECT lid, rid, p FROM read_parquet('{pred_path}'))"
    if context_pred_path is not None:
        preds = f"""(SELECT lid, rid, p FROM read_parquet('{pred_path}')
                     UNION ALL
                     SELECT c.lid, c.rid, c.p FROM read_parquet('{context_pred_path}') c
                     ANTI JOIN read_parquet('{pred_path}') q ON c.lid = q.lid AND c.rid = q.rid)"""
    cols = ", ".join(f"s.{c}" for c in STAGE2_FEATURES)
    ecols = ", ".join(f"e.{c}" for c in ENTITY_FEATURES)
    con.execute(
        f"""COPY (
              SELECT f.*, {cols}, {ecols}
              FROM read_parquet('{feat_path}') f
              JOIN ({competition_sql(preds, mutual_k)}) s ON s.lid = f.lid AND s.rid = f.rid
              JOIN ({entity_sql(preds)}) e ON e.lid = f.lid AND e.rid = f.rid
            ) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)"""
    )
    n_out = pq.ParquetFile(out_path).metadata.num_rows
    n_feat = pq.ParquetFile(feat_path).metadata.num_rows
    if n_out != n_feat:
        raise ValueError(f"{feat_path}: {n_feat} feature rows but {n_out} matched a stage-1 prediction")
    log.info("stage2 features written path=%s rows=%d", out_path, n_out)
    return n_out


# --------------------------------------------------------------------------------------------------------------
# Out-of-fold protocol
# --------------------------------------------------------------------------------------------------------------


def fold_sql(col: str, n_folds: int = N_FOLDS) -> str:
    """SQL expression: OOF fold in [0, n_folds) of an S1 id column (md5 digits above the group bucket)."""
    return f"((md5_number({col}) // 100) % {int(n_folds)})"


def context_where(group: str) -> str:
    """``where_sql`` for features.write_features selecting a train group's rival pairs.

    Rival pairs: S1 outside ``group``, R is a candidate of some S1 inside it (table ``candidates``, alias c).
    """
    inside = f"SELECT rid FROM candidates WHERE {group_where('lid', group)}"
    return f"(NOT {group_where('c.lid', group)} AND c.rid IN ({inside}))"


def _copy_where(con: duckdb.DuckDBPyConnection, src: Path, dst: Path, where: str) -> int:
    """Copy the rows of parquet ``src`` matching SQL ``where`` into parquet ``dst``; return the row count."""
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{src}') WHERE {where}) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)"
    )
    return pq.ParquetFile(dst).metadata.num_rows


def _protocol_parts(n_folds: int) -> list[tuple[int | None, str]]:
    """(fold or None for the full model, SQL predicate on lid) for each part of the leakage protocol."""
    parts: list[tuple[int | None, str]] = [
        (k, f"({group_where('lid', 'fit')} AND {fold_sql('lid', n_folds)} = {k})") for k in range(n_folds)
    ]
    parts.append((None, f"(NOT {group_where('lid', 'fit')})"))
    return parts


def predict_protocol(
    con: duckdb.DuckDBPyConnection,
    feat_path: Path,
    out_path: Path,
    fold_boosters: list[Any],
    full_booster: Any | None,
    batch_rows: int,
    work_dir: Path,
    predict_fn: PredictFn = predict_file,
) -> int:
    """Stage-1 predictions for a TRAIN-split feature file under the leakage protocol; return rows written.

    Rows whose S1 is in the fit group, fold k, are scored by fold_boosters[k] (trained on the other folds); all
    other rows by full_booster. Writes (lid, rid, p[, label]) to out_path. Never use on the test split (its ids are
    not train ids; score test with model.predict_file and the full model).
    """
    n_folds = len(fold_boosters)
    parts = _protocol_parts(n_folds)
    counts = con.execute(
        "SELECT " + ", ".join(f"count(*) FILTER (WHERE {w})" for _, w in parts) + f" FROM read_parquet('{feat_path}')"
    ).fetchone()
    work_dir.mkdir(parents=True, exist_ok=True)
    used = [(fold, where) for (fold, where), c in zip(parts, counts, strict=True) if c > 0]
    for fold, _ in used:
        if fold is None and full_booster is None:
            raise ValueError(f"{feat_path} has rows outside the fit group but no full_booster was given")
    if len(used) == 1:
        fold = used[0][0]
        return predict_fn(full_booster if fold is None else fold_boosters[fold], feat_path, out_path, batch_rows)
    outs = []
    for i, (fold, where) in enumerate(used):
        part_in, part_out = work_dir / f"protocol_in_{i}.parquet", work_dir / f"protocol_out_{i}.parquet"
        _copy_where(con, feat_path, part_in, where)
        predict_fn(full_booster if fold is None else fold_boosters[fold], part_in, part_out, batch_rows)
        part_in.unlink()
        outs.append(part_out)
    files = ", ".join(f"'{p}'" for p in outs)
    con.execute(f"COPY (SELECT * FROM read_parquet([{files}])) TO '{out_path}' (FORMAT parquet, COMPRESSION zstd)")
    for p in outs:
        p.unlink()
    return pq.ParquetFile(out_path).metadata.num_rows


def oof_predict(
    con: duckdb.DuckDBPyConnection,
    fit_feat_path: Path,
    valid_feat_path: Path,
    out_path: Path,
    work_dir: Path,
    params: dict,
    seed: int,
    threads: int,
    batch_rows: int,
    n_folds: int = N_FOLDS,
    train_fn: TrainFn = train,
    predict_fn: PredictFn = predict_file,
) -> list[Any]:
    """Out-of-fold stage-1 predictions for the fit group; returns the fold boosters (fold k = trained without k).

    For each fold k, trains ``train_fn(part, valid_feat_path, params, seed, threads)`` on the fit rows of the other
    folds (early stopping on the tune group), then scores every fit row with the booster that did not see its fold
    and writes (lid, rid, p, label) to out_path. Temporary parts live in work_dir and are removed.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    boosters = []
    for k in range(n_folds):
        part = work_dir / f"oof_train_{k}.parquet"
        rows = _copy_where(con, fit_feat_path, part, f"{fold_sql('lid', n_folds)} != {k}")
        log.info("oof fold=%d train_rows=%d", k, rows)
        boosters.append(train_fn(part, valid_feat_path, params, seed, threads))
        part.unlink()
    predict_protocol(con, fit_feat_path, out_path, boosters, None, batch_rows, work_dir, predict_fn)
    return boosters
