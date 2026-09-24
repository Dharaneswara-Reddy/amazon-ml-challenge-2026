"""Turn pair probabilities into per-Source-1 match sets, and tune the decision threshold for macro F0.5.

Each right record matches at most one Source 1 entity (measured on train), so a right record is attached only to
its highest-scoring candidate, and only if that probability clears the threshold.
"""

from __future__ import annotations

import numpy as np

BETA2 = 0.25


def best_per_right(rid: np.ndarray, lid: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Boolean mask marking, for each right id, its single highest-probability pair (ties broken by lid)."""
    order = np.lexsort((lid, -p, rid))
    first = np.ones(len(order), dtype=bool)
    first[1:] = rid[order][1:] != rid[order][:-1]
    mask = np.zeros(len(p), dtype=bool)
    mask[order[first]] = True
    return mask


def macro_f05(lid_idx: np.ndarray, selected: np.ndarray, label: np.ndarray, n_true: np.ndarray) -> float:
    """Macro F0.5 over all Source 1 entities.

    lid_idx: per pair, index of its Source 1 entity in [0, len(n_true)).
    selected: per pair, whether it is predicted as a match. label: per pair, whether it is a true match.
    n_true: per entity, number of true matches in ground truth (including ones missing from candidates).
    Singletons score 1.0 with no predicted match and 0.0 otherwise.
    """
    n = len(n_true)
    tp = np.bincount(lid_idx, weights=(selected & label).astype(np.float64), minlength=n)
    fp = np.bincount(lid_idx, weights=(selected & ~label).astype(np.float64), minlength=n)
    fn = n_true - tp
    denom = (1 + BETA2) * tp + BETA2 * fn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(denom > 0, (1 + BETA2) * tp / denom, 0.0)
    f = np.where(n_true == 0, (fp == 0).astype(np.float64), f)
    return float(f.mean())


def tune_threshold(
    lid_idx: np.ndarray,
    rid: np.ndarray,
    p: np.ndarray,
    label: np.ndarray,
    n_true: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, list[tuple[float, float]]]:
    """Grid-search the probability threshold maximizing macro F0.5; return (best threshold, [(t, score), ...])."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best = best_per_right(rid, lid_idx, p)
    curve = [(float(t), macro_f05(lid_idx, best & (p >= t), label, n_true)) for t in grid]
    t_best = max(curve, key=lambda x: (x[1], -x[0]))[0]
    return t_best, curve


def select(lid_idx: np.ndarray, rid: np.ndarray, p: np.ndarray, threshold: float) -> np.ndarray:
    """Final decision mask: best candidate per right record with probability at or above the threshold."""
    return best_per_right(rid, lid_idx, p) & (p >= threshold)
