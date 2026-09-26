"""Turn pair probabilities into per-Source-1 match sets, and tune the decision rule for macro F0.5.

Each right record matches at most one Source 1 entity (measured on train), so a right record is attached only to
its highest-scoring candidate (the one-to-one step). Two rules then pick matches among those candidates:

* global threshold: keep a candidate if its probability clears one tuned threshold (``select``);
* expected-F0.5 (``expected_f_select``): per Source 1 entity, sort its candidates by calibrated probability and keep
  the prefix size k (possibly 0) that maximizes the expected per-entity F0.5.

Expected-F0.5 derivation. For one entity pick the top-k candidates. With TP true picks and n_true true matches,
FP = k - TP and FN = n_true - TP, so

    F0.5 = 1.25 TP / (1.25 TP + 0.25 FN + FP) = 1.25 TP / (k + 0.25 n_true)          (k + n_true > 0)

and F0.5 = 1 when k = n_true = 0 (the singleton rule). Treat the candidates' labels Y_i ~ Bernoulli(p_i) as
independent (p = calibrated probability) and let M be the entity's true matches missing from its candidates. Then

    k = 0:  E[F] = P(n_true = 0) = prod_all (1 - p_j) * P(M = 0 | no true match among candidates)
    k >= 1: E[F] = 1.25 sum_{i<=k} p_i E[1 / (k + 0.25 (1 + X_i))],   X_i = n_true - Y_i given Y_i = 1

using TP = sum_i Y_i and linearity (the "+1" is Y_i itself). E[X_i] = mu - p_i and Var[X_i] = s2 - p_i (1 - p_i)
where mu = sum_all p_j + E[M], s2 = sum_all p_j (1 - p_j) + Var[M] (sums over ALL candidates of the entity,
including ones whose right record prefers another entity). A second-order delta method gives

    E[1 / (D + 0.25 Z)] ~= 1 / D' + 0.0625 Var[Z] / D'^3,   D' = k + 0.25 (1 + E[X_i]),

which is exact for a lone candidate and within 0.002 of brute-force enumeration on small entities (the chosen k
loses < 1e-4 expected F; tests/test_decision.py). The top-k prefix is optimal because each term p / (c - 0.25 p)
increases with p.

Everything is fitted on the tune group (``fit_decision``):
* calibration: isotonic regression (pool-adjacent-violators) of the label on the raw score, separately for pairs
  that are their right record's best candidate and for rivals, with linear interpolation between pooled-block
  centres (centred isotonic regression) so new scores map monotonically;
* missing matches M (blocking misses) per bin of an entity's sum of calibrated p;
* ``cal_empty``: per-pair p ignore entity context (a lone mid-score candidate is usually a singleton's false match),
  so the model's P(n_true = 0) is itself isotonic-calibrated against the observed singleton indicator, and the
  k >= 1 values are rescaled consistently (E[F_k] = (1 - q) E[F_k | n_true > 0]);
* two robustness knobs tuned on realized macro F0.5: a logit shift of calibrated p and a weight on the k = 0 option.

Simulation (tests/test_decision.py::simulate_group, 30k entities per group, six noise settings): the tuned global
threshold is beaten by +0.4 to +1.6 macro-F0.5 points on holdout; without ``cal_empty`` the rule LOSES 1.5 points
when 20% of entities are singletons. Always confirm on real holdout data with ``compare_decisions``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

BETA2 = 0.25
DEFAULT_MAX_K = 16


def best_per_right(rid: np.ndarray, lid: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Boolean mask marking, for each right id, its single highest-probability pair (ties broken by lid)."""
    order = np.lexsort((lid, -p, rid))
    first = np.ones(len(order), dtype=bool)
    first[1:] = rid[order][1:] != rid[order][:-1]
    mask = np.zeros(len(p), dtype=bool)
    mask[order[first]] = True
    return mask


def per_entity_f05(lid_idx: np.ndarray, selected: np.ndarray, label: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """Per-entity F0.5 (array of len(n_true)); same inputs and singleton rule as ``macro_f05``."""
    n = len(n_true)
    tp = np.bincount(lid_idx, weights=(selected & label).astype(np.float64), minlength=n)
    fp = np.bincount(lid_idx, weights=(selected & ~label).astype(np.float64), minlength=n)
    fn = n_true - tp
    denom = (1 + BETA2) * tp + BETA2 * fn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(denom > 0, (1 + BETA2) * tp / denom, 0.0)
    return np.where(n_true == 0, (fp == 0).astype(np.float64), f)


def macro_f05(lid_idx: np.ndarray, selected: np.ndarray, label: np.ndarray, n_true: np.ndarray) -> float:
    """Macro F0.5 over all Source 1 entities.

    lid_idx: per pair, index of its Source 1 entity in [0, len(n_true)).
    selected: per pair, whether it is predicted as a match. label: per pair, whether it is a true match.
    n_true: per entity, number of true matches in ground truth (including ones missing from candidates).
    Singletons score 1.0 with no predicted match and 0.0 otherwise.
    """
    return float(per_entity_f05(lid_idx, selected, label, n_true).mean())


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


# --------------------------------------------------------------------------------------------------------------
# Calibration: isotonic regression by pool-adjacent-violators, centred monotone interpolation for new scores
# --------------------------------------------------------------------------------------------------------------


def pava(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted least-squares non-decreasing fit to ``y`` (already ordered by x) by pool-adjacent-violators.

    y: values in x order; w: positive weights (default 1). Returns fitted values of the same length. The block stack
    is a Python loop, O(len(y)); ``fit_calibrator`` bins large inputs to at most a few thousand points first.
    """
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    vals: list[float] = []
    wts: list[float] = []
    lens: list[int] = []
    for yi, wi in zip(y.tolist(), w.tolist(), strict=True):
        v, ww, n = yi, wi, 1
        while vals and vals[-1] > v:
            pv, pw, pn = vals.pop(), wts.pop(), lens.pop()
            v = (pv * pw + v * ww) / (pw + ww)
            ww += pw
            n += pn
        vals.append(v)
        wts.append(ww)
        lens.append(n)
    return np.repeat(np.asarray(vals, dtype=np.float64), np.asarray(lens, dtype=np.int64))


@dataclass(frozen=True, eq=False)
class Calibrator:
    """Monotone map raw score -> calibrated match probability: linear between knots (x, y), clamped outside."""

    x: np.ndarray
    y: np.ndarray

    def __call__(self, score: np.ndarray) -> np.ndarray:
        """Calibrated probabilities (float64) for an array of raw scores."""
        return np.interp(np.asarray(score, dtype=np.float64), self.x, self.y)

    def to_dict(self) -> dict[str, list[float]]:
        """JSON-ready representation (knot lists)."""
        return {"x": self.x.tolist(), "y": self.y.tolist()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Calibrator:
        """Inverse of ``to_dict``."""
        return cls(np.asarray(d["x"], dtype=np.float64), np.asarray(d["y"], dtype=np.float64))


def fit_calibrator(score: np.ndarray, label: np.ndarray, max_bins: int = 4096) -> Calibrator:
    """Fit centred isotonic regression of ``label`` (0/1) on ``score``; return a ``Calibrator``.

    Sorted scores are cut into at most ``max_bins`` equal-count bins that never split a run of equal scores, PAVA
    runs on the bin means (weighted by counts), and each pooled block becomes one knot at its count-weighted mean
    score with the block's match rate as value. Knot x's are strictly increasing, so interpolation is monotone.
    """
    s = np.asarray(score, dtype=np.float64)
    if len(s) == 0:
        raise ValueError("fit_calibrator needs at least one scored pair")
    order = np.argsort(s, kind="stable")
    s = s[order]
    y = np.asarray(label, dtype=np.float64)[order]
    n = len(s)
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    targets = np.linspace(0, n, max_bins + 1)[:-1]
    cuts = np.unique(starts[np.minimum(np.searchsorted(starts, targets), len(starts) - 1)])
    cnt = np.diff(np.r_[cuts, n]).astype(np.float64)
    xs = np.add.reduceat(s, cuts) / cnt
    fit = pava(np.add.reduceat(y, cuts) / cnt, cnt)
    new_block = np.r_[True, fit[1:] != fit[:-1]]
    block = np.cumsum(new_block) - 1
    kx = np.bincount(block, weights=xs * cnt) / np.bincount(block, weights=cnt)
    return Calibrator(kx, fit[new_block])


# --------------------------------------------------------------------------------------------------------------
# Matches missed by blocking
# --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class MissingModel:
    """Distribution of an entity's true matches absent from its candidates (M), binned by its sum of calibrated p.

    edges: increasing interior bin edges on sum_p (len = n_bins - 1).
    mean, var: E[M], Var[M] per bin among entities with >= 1 true match among candidates (used for k >= 1, where
      the picked candidate is assumed right).
    p0: P(M = 0) per bin among entities with no true match among candidates (used for the empty prediction).
    """

    edges: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    p0: np.ndarray

    def lookup(self, sum_p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(E[M], Var[M], P(M=0)) per entity for an array of per-entity sums of calibrated p."""
        b = np.searchsorted(self.edges, np.asarray(sum_p, dtype=np.float64), side="right")
        return self.mean[b], self.var[b], self.p0[b]

    @classmethod
    def none(cls) -> MissingModel:
        """Model assuming blocking misses nothing (M = 0 always)."""
        return cls(np.empty(0), np.zeros(1), np.zeros(1), np.ones(1))

    def to_dict(self) -> dict[str, list[float]]:
        """JSON-ready representation."""
        return {k: getattr(self, k).tolist() for k in ("edges", "mean", "var", "p0")}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MissingModel:
        """Inverse of ``to_dict``."""
        return cls(*(np.asarray(d[k], dtype=np.float64) for k in ("edges", "mean", "var", "p0")))


def _shrunk_bin_mean(bins: np.ndarray, values: np.ndarray, n_bins: int, prior: float, strength: float) -> np.ndarray:
    """Per-bin mean of ``values`` shrunk toward ``prior`` by ``strength`` pseudo-observations (empty bin -> prior)."""
    s = np.bincount(bins, weights=values.astype(np.float64), minlength=n_bins)
    c = np.bincount(bins, minlength=n_bins).astype(np.float64) + strength
    return np.divide(s + strength * prior, c, out=np.full(n_bins, float(prior)), where=c > 0)


def fit_missing_model(
    lid_idx: np.ndarray,
    p: np.ndarray,
    label: np.ndarray,
    n_true: np.ndarray,
    n_bins: int = 20,
    strength: float = 20.0,
) -> MissingModel:
    """Fit a ``MissingModel`` on a labelled group (inputs as in ``macro_f05``; ``p`` calibrated).

    Only entities with at least one candidate are used (others are never decided). Bins are quantiles of the
    per-entity sum of p; bin statistics are shrunk toward the global value with ``strength`` pseudo-counts.
    """
    n = len(n_true)
    has = np.bincount(lid_idx, minlength=n) > 0
    if not has.any():
        return MissingModel.none()
    sum_p = np.bincount(lid_idx, weights=np.asarray(p, dtype=np.float64), minlength=n)[has]
    found = np.bincount(lid_idx, weights=np.asarray(label, dtype=np.float64), minlength=n)[has]
    miss = np.maximum(np.asarray(n_true, dtype=np.float64)[has] - found, 0.0)
    edges = np.unique(np.quantile(sum_p, np.linspace(0, 1, n_bins + 1)[1:-1]))
    nb = len(edges) + 1
    b = np.searchsorted(edges, sum_p, side="right")
    pos = found > 0
    m1 = float(miss[pos].mean()) if pos.any() else 0.0
    m2 = float((miss[pos] ** 2).mean()) if pos.any() else 0.0
    z0 = float((miss[~pos] == 0).mean()) if (~pos).any() else 1.0
    mean = _shrunk_bin_mean(b[pos], miss[pos], nb, m1, strength)
    var = np.maximum(_shrunk_bin_mean(b[pos], miss[pos] ** 2, nb, m2, strength) - mean**2, 0.0)
    p0 = _shrunk_bin_mean(b[~pos], miss[~pos] == 0, nb, z0, strength)
    return MissingModel(edges, mean, var, p0)


# --------------------------------------------------------------------------------------------------------------
# Expected-F0.5 selection
# --------------------------------------------------------------------------------------------------------------


def entity_stats(lid_idx: np.ndarray, p: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-entity (sum p, sum p(1-p), sum log(1-p)) over ALL candidate pairs, from calibrated ``p``.

    SQL equivalent for very large inputs: ``SELECT lid, sum(p), sum(p*(1-p)), sum(ln(1-least(p, 1-1e-12)))
    FROM preds GROUP BY lid``.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return (
        np.bincount(lid_idx, weights=p, minlength=n),
        np.bincount(lid_idx, weights=p * (1.0 - p), minlength=n),
        np.bincount(lid_idx, weights=np.log1p(-np.minimum(p, 1.0 - 1e-12)), minlength=n),
    )


def _position_in_entity(lid: np.ndarray, key: np.ndarray) -> np.ndarray:
    """0-based position of each pair within its entity when sorted by ``key`` descending (ties: input order)."""
    order = np.lexsort((-key, lid))
    ls = lid[order]
    idx = np.arange(len(ls))
    start = np.maximum.accumulate(np.where(np.r_[True, ls[1:] != ls[:-1]], idx, 0))
    pos = np.empty(len(ls), dtype=np.int64)
    pos[order] = idx - start
    return pos


@dataclass(frozen=True, eq=False)
class Prepared:
    """One-to-one step and per-entity ordering for one set of raw scores, reusable across calibrations.

    n: number of entities. lid_idx: entity of every pair. is_best: pair is its right record's best candidate.
    idx: selectable pairs (best for their right record and among the first max_k of their entity), sorted by
    position; lid/pos: their entity and 0-based position by descending score. ends[k-1]: how many selectable pairs
    have position < k. n_cand: selectable pairs per entity.
    """

    n: int
    lid_idx: np.ndarray
    is_best: np.ndarray
    idx: np.ndarray
    lid: np.ndarray
    pos: np.ndarray
    ends: np.ndarray
    n_cand: np.ndarray
    max_k: int


def prepare(
    lid_idx: np.ndarray, rid: np.ndarray, score: np.ndarray, max_k: int = DEFAULT_MAX_K, n: int | None = None
) -> Prepared:
    """Run the one-to-one step on raw ``score`` and order each entity's surviving candidates (see ``Prepared``).

    n: number of entities (default max(lid_idx) + 1; pass len(n_true) to include entities without candidates).
    """
    lid_idx = np.asarray(lid_idx, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    n = (int(lid_idx.max()) + 1 if len(lid_idx) else 0) if n is None else int(n)
    is_best = best_per_right(rid, lid_idx, score)
    best = np.flatnonzero(is_best)
    pos = _position_in_entity(lid_idx[best], score[best])
    keep = pos < max_k
    best, pos = best[keep], pos[keep]
    order = np.argsort(pos, kind="stable")
    idx, pos = best[order], pos[order]
    lid = lid_idx[idx]
    ends = np.searchsorted(pos, np.arange(1, max_k + 1), side="left")
    return Prepared(n, lid_idx, is_best, idx, lid, pos, ends, np.bincount(lid, minlength=n), max_k)


def _moments(prep: Prepared, p: np.ndarray, missing: MissingModel | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per entity: (mean of n_true, variance of n_true, model P(n_true = 0)) from calibrated ``p`` of all pairs."""
    sum_p, var_p, log_q = entity_stats(prep.lid_idx, p, prep.n)
    m_mean, m_var, m_p0 = (missing or MissingModel.none()).lookup(sum_p)
    return sum_p + m_mean, var_p + m_var, np.exp(log_q) * m_p0


def _expected_f_by_k(prep: Prepared, p: np.ndarray, mu: np.ndarray, s2: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (k, E[F0.5] per entity of its top-k prefix) for k = 1..max_k (formula in the module docstring).

    p: calibrated probabilities of all pairs; mu/s2: per-entity mean/variance of n_true. The k-independent parts
    are gathered once; iteration k touches only the selectable pairs with position < k (a prefix of prep.idx).
    """
    lk = prep.lid
    pk = np.asarray(p, dtype=np.float64)[prep.idx]
    a = BETA2 + BETA2 * (mu[lk] - pk)
    v = BETA2 * BETA2 * np.maximum(s2[lk] - pk * (1.0 - pk), 0.0)
    w = (1 + BETA2) * pk
    for k in range(1, prep.max_k + 1):
        m = int(prep.ends[k - 1])
        if m == 0:
            return
        inv = 1.0 / (k + a[:m])
        yield k, np.bincount(lk[:m], weights=w[:m] * inv * (1.0 + v[:m] * inv * inv), minlength=prep.n)


def choose_k(
    prep: Prepared,
    p: np.ndarray,
    missing: MissingModel | None = None,
    empty_weight: float = 1.0,
    cal_empty: Calibrator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per entity, the prefix size k (0..max_k) maximizing expected F0.5; returns (k_star, E[F0.5] at k_star).

    p: calibrated probabilities of all pairs (aligned with prep.lid_idx). missing: blocking-miss model (default: none).
    cal_empty: optional isotonic map from the model's P(n_true = 0) to the empirical singleton rate (``fit_decision``);
    since F0.5 = 0 whenever k >= 1 and n_true = 0, E[F_k] = (1 - q) E[F_k | n_true > 0], so the k >= 1 values are
    rescaled by (1 - q_calibrated) / (1 - q_model) to stay consistent with the calibrated q.
    empty_weight multiplies E[F0.5 | k = 0] in the comparison only (1 = pure model; > 1 favours empty predictions).
    """
    mu, s2, f_empty = _moments(prep, p, missing)
    scale = None
    if cal_empty is not None:
        q_cal = cal_empty(f_empty)
        scale = (1.0 - q_cal) / np.maximum(1.0 - f_empty, 1e-12)
        f_empty = q_cal
    best_f = empty_weight * f_empty
    ef = f_empty.copy()
    k_star = np.zeros(prep.n, dtype=np.int64)
    for k, f_k in _expected_f_by_k(prep, p, mu, s2):
        if scale is not None:
            f_k = f_k * scale
        better = (prep.n_cand >= k) & (f_k > best_f + 1e-12)
        best_f = np.where(better, f_k, best_f)
        ef = np.where(better, f_k, ef)
        k_star[better] = k
    return k_star, ef


def mask_from_k(prep: Prepared, k_star: np.ndarray) -> np.ndarray:
    """Decision mask over all pairs: each entity's first k_star selectable pairs."""
    mask = np.zeros(len(prep.lid_idx), dtype=bool)
    mask[prep.idx[prep.pos < k_star[prep.lid]]] = True
    return mask


def expected_f_select(
    lid_idx: np.ndarray,
    rid: np.ndarray,
    p: np.ndarray,
    score: np.ndarray | None = None,
    missing: MissingModel | None = None,
    max_k: int = DEFAULT_MAX_K,
    empty_weight: float = 1.0,
    cal_empty: Calibrator | None = None,
) -> np.ndarray:
    """Decision mask by per-entity expected-F0.5 maximization (one-to-one step first).

    lid_idx: per pair, entity index. rid: right ids. p: CALIBRATED probabilities. score: raw scores for the
    one-to-one step and ordering (default ``p``; pass the raw model score so flat isotonic steps do not create ties).
    missing / empty_weight / cal_empty: see ``choose_k``. At most ``max_k`` picks per entity.
    Memory: about 12 arrays of 8 bytes per pair (~100 B/pair; 5 GB at 50M pairs, 1 GB at the ~10M R-best pairs).
    """
    prep = prepare(lid_idx, rid, p if score is None else score, max_k)
    k_star, _ = choose_k(prep, np.asarray(p, dtype=np.float64), missing, empty_weight, cal_empty)
    return mask_from_k(prep, k_star)


def expected_f_curve(
    p: np.ndarray, missing_mean: float = 0.0, missing_var: float = 0.0, missing_p0: float = 1.0
) -> np.ndarray:
    """E[F0.5] for k = 0..len(p) of ONE entity whose candidates (all selectable) have calibrated probabilities ``p``.

    Diagnostic wrapper over the same code path as ``choose_k``; returns a float64 array of len(p) + 1.
    """
    p = np.sort(np.asarray(p, dtype=np.float64))[::-1]
    prep = prepare(np.zeros(len(p), dtype=np.int64), np.arange(len(p)), p, max_k=max(len(p), 1), n=1)
    sum_p, var_p, log_q = entity_stats(prep.lid_idx, p, 1)
    mu, s2 = sum_p + missing_mean, var_p + missing_var
    out = [float(np.exp(log_q[0]) * missing_p0)]
    out += [float(f[0]) for _, f in _expected_f_by_k(prep, p, mu, s2)]
    return np.asarray(out)


# --------------------------------------------------------------------------------------------------------------
# Fitting and comparing decision rules on labelled groups
# --------------------------------------------------------------------------------------------------------------

SHIFT_GRID = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
EMPTY_WEIGHT_GRID = (0.8, 1.0, 1.25, 1.5, 2.0, 3.0)


def _logit_shift(p: np.ndarray, shift: float | np.ndarray) -> np.ndarray:
    """Add ``shift`` (scalar, or one value per pair) to the logit of probabilities ``p`` (identity when 0)."""
    if np.isscalar(shift) and shift == 0:
        return p
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + shift)))


def calibrate(
    params: Mapping[str, Any], is_best: np.ndarray, p: np.ndarray, shift: float | np.ndarray | None = None
) -> np.ndarray:
    """Calibrated probabilities from raw scores ``p`` with fitted params (see ``fit_decision``).

    Pairs that are their right record's best candidate (``is_best``, e.g. ``Prepared.is_best``) go through
    ``cal_best``, rivals that lose the one-to-one step through ``cal_other``; then the logit is shifted by ``shift``
    (default ``params['shift']``; an array gives one shift per pair, e.g. per-country prior shifts).
    """
    pc = np.where(is_best, Calibrator.from_dict(params["cal_best"])(p), Calibrator.from_dict(params["cal_other"])(p))
    return _logit_shift(pc, float(params.get("shift", 0.0)) if shift is None else shift)


def fit_decision(
    tune: Mapping[str, np.ndarray],
    max_k: int = DEFAULT_MAX_K,
    n_missing_bins: int = 20,
    shift_grid: tuple[float, ...] = SHIFT_GRID,
    empty_weight_grid: tuple[float, ...] = EMPTY_WEIGHT_GRID,
) -> dict[str, Any]:
    """Fit both rules on a labelled group; return JSON-ready params.

    tune: dict with lid_idx, rid, p (raw model score), label, n_true, as built by ``pipeline._group_arrays``.
    Steps: global threshold; isotonic calibrators for R-best pairs and for rivals; missing-match model; an isotonic
    ``cal_empty`` mapping the model's P(n_true = 0) to the observed singleton rate (fixes context-free p, e.g. a
    lone mid-score candidate is usually a singleton's false match); then a grid search of two robustness knobs on
    realized tune macro F0.5: logit ``shift`` of calibrated p and ``empty_weight`` (grids of length 1 disable it).
    Output keys: rule, threshold, cal_best, cal_other, cal_empty, missing, max_k, shift, empty_weight, tune_grid.
    """
    lid, rid, p, label, n_true = (tune[k] for k in ("lid_idx", "rid", "p", "label", "n_true"))
    t, _ = tune_threshold(lid, rid, p, label, n_true)
    prep = prepare(lid, rid, p, max_k, len(n_true))
    best = prep.is_best
    cal_best = fit_calibrator(p[best], label[best])
    cal_other = fit_calibrator(p[~best], label[~best]) if (~best).any() else cal_best
    params: dict[str, Any] = {
        "rule": "expected_f",
        "threshold": t,
        "cal_best": cal_best.to_dict(),
        "cal_other": cal_other.to_dict(),
        "max_k": max_k,
        "shift": 0.0,
        "empty_weight": 1.0,
    }
    pc = calibrate(params, best, p)
    miss = fit_missing_model(lid, pc, label, n_true, n_missing_bins)
    has = np.bincount(lid, minlength=len(n_true)) > 0
    f_empty = _moments(prep, pc, miss)[2]
    cal_empty = fit_calibrator(f_empty[has], (n_true == 0)[has])
    params.update(missing=miss.to_dict(), cal_empty=cal_empty.to_dict())
    grid = []
    for shift in shift_grid:
        ps = _logit_shift(pc, shift)
        for w in empty_weight_grid:
            sel = mask_from_k(prep, choose_k(prep, ps, miss, w, cal_empty)[0])
            grid.append((float(shift), float(w), macro_f05(lid, sel, label, n_true)))
    shift, w, _ = max(grid, key=lambda g: (g[2], -abs(g[0]), -abs(g[1] - 1.0)))
    params.update(shift=shift, empty_weight=w, tune_grid=grid)
    return params


def decide(
    params: Mapping[str, Any],
    lid_idx: np.ndarray,
    rid: np.ndarray,
    p: np.ndarray,
    pair_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Apply saved decision params (from ``fit_decision``; ``params['rule']`` picks the rule) to raw scores ``p``.

    lid_idx: entity index per pair (e.g. ``dense_rank() OVER (ORDER BY lid) - 1`` in DuckDB); rid: right ids (any
    sortable dtype, integer codes are cheapest). pair_shift: optional logit shift per pair (overrides
    ``params['shift']``, expected_f rule only). Returns the boolean match mask.
    """
    if params["rule"] == "threshold":
        return select(lid_idx, rid, p, float(params["threshold"]))
    if params["rule"] != "expected_f":
        raise ValueError(f"unknown decision rule {params['rule']!r}")
    prep = prepare(lid_idx, rid, p, int(params["max_k"]))
    k_star, _ = choose_k(
        prep,
        calibrate(params, prep.is_best, p, pair_shift),
        MissingModel.from_dict(params["missing"]),
        float(params["empty_weight"]),
        Calibrator.from_dict(params["cal_empty"]),
    )
    return mask_from_k(prep, k_star)


def compare_decisions(
    tune: Mapping[str, np.ndarray],
    holdout: Mapping[str, np.ndarray],
    max_k: int = DEFAULT_MAX_K,
    n_missing_bins: int = 20,
) -> dict[str, Any]:
    """Fit on ``tune``; score the global threshold and three expected-F0.5 variants on both groups.

    Inputs: dicts as built by ``pipeline._group_arrays`` (lid_idx, rid, p raw score, label, n_true). Variants:
    ``expected_f_pure`` (calibrated p only), ``expected_f_model`` (+ missing model + cal_empty),
    ``expected_f`` (+ tuned shift/empty_weight; what ``decide`` runs). Returns ``{"threshold", "shift",
    "empty_weight", "tune": {rule: f05}, "holdout": {rule: f05}, "holdout_expected_pred" / "holdout_expected_real":
    predicted vs realized mean F0.5 of expected_f_model on holdout entities with candidates (a calibration check),
    "params": fit_decision output}``. Tune scores are in-sample; use ``choose_decision`` to pick the rule.
    """
    params = fit_decision(tune, max_k, n_missing_bins)
    miss = MissingModel.from_dict(params["missing"])
    cal_empty = Calibrator.from_dict(params["cal_empty"])
    shift, w = params["shift"], params["empty_weight"]
    out: dict[str, Any] = {"threshold": params["threshold"], "shift": shift, "empty_weight": w}
    for name, g in (("tune", tune), ("holdout", holdout)):
        lid, rid, p, label, n_true = (g[k] for k in ("lid_idx", "rid", "p", "label", "n_true"))
        prep = prepare(lid, rid, p, max_k, len(n_true))
        pc0 = calibrate(params, prep.is_best, p, shift=0.0)
        k_model, ef = choose_k(prep, pc0, miss, 1.0, cal_empty)
        masks = {
            "threshold": select(lid, rid, p, params["threshold"]),
            "expected_f_pure": mask_from_k(prep, choose_k(prep, pc0)[0]),
            "expected_f_model": mask_from_k(prep, k_model),
            "expected_f": mask_from_k(prep, choose_k(prep, _logit_shift(pc0, shift), miss, w, cal_empty)[0]),
        }
        out[name] = {k: macro_f05(lid, m, label, n_true) for k, m in masks.items()}
        if name == "holdout":
            has = np.bincount(lid, minlength=len(n_true)) > 0
            real = per_entity_f05(lid, masks["expected_f_model"], label, n_true)
            out["holdout_expected_pred"] = float(ef[has].mean()) if has.any() else float("nan")
            out["holdout_expected_real"] = float(real[has].mean()) if has.any() else float("nan")
    out["params"] = dict(params)
    return out


def _subset(g: Mapping[str, np.ndarray], keep_entity: np.ndarray) -> dict[str, np.ndarray]:
    """Restrict group arrays to the entities where keep_entity is True, re-indexing lid_idx compactly."""
    new_idx = np.cumsum(keep_entity) - 1
    pair_keep = keep_entity[g["lid_idx"]]
    return {
        "lid_idx": new_idx[g["lid_idx"][pair_keep]],
        "rid": g["rid"][pair_keep],
        "p": g["p"][pair_keep],
        "label": g["label"][pair_keep],
        "n_true": g["n_true"][keep_entity],
    }


def choose_decision(
    tune: Mapping[str, np.ndarray], holdout: Mapping[str, np.ndarray], max_k: int = DEFAULT_MAX_K
) -> dict[str, Any]:
    """Pick threshold vs expected-F0.5 WITHOUT touching the holdout (decisions D-004, D-009).

    The tune entities are split in two halves (even / odd entity index): the rule parameters are fitted on half A
    and the rule is chosen by macro F0.5 on half B. Parameters are then refitted on the full tune group and both
    rules are reported on the holdout for information only. Returns compare_decisions(tune, holdout) plus
    ``params["rule"]`` and ``selection`` (half-B scores).
    """
    half = np.arange(len(tune["n_true"])) % 2 == 0
    sel = compare_decisions(_subset(tune, half), _subset(tune, ~half), max_k)
    winner = max(("threshold", "expected_f"), key=lambda r: sel["holdout"][r])
    full = compare_decisions(tune, holdout, max_k)
    full["params"]["rule"] = winner
    full["selection"] = {"rule": winner, "tune_half_b": sel["holdout"]}
    return full


def em_prior_shift(
    p: np.ndarray, train_prior: float, max_iter: int = 500, tol: float = 1e-8
) -> tuple[float, np.ndarray]:
    """Estimate the test positive rate and re-weighted probabilities by EM (Saerens, Latinne & Decaestecker 2002).

    p: calibrated P(match) from a model trained where the positive rate among these pairs was ``train_prior``.
    Iterates prior' = mean(adjusted p) with adjusted p = a·p / (a·p + b·(1 − p)), a = prior'/train_prior,
    b = (1 − prior')/(1 − train_prior). Returns (estimated test prior, adjusted probabilities). Decision D-010.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    prior = float(train_prior)
    adj = p
    for _ in range(max_iter):
        a, b = prior / train_prior, (1 - prior) / (1 - train_prior)
        adj = a * p / (a * p + b * (1 - p))
        new = float(adj.mean())
        if abs(new - prior) < tol:
            prior = new
            break
        prior = new
    return prior, adj


def shift_probabilities(p: np.ndarray, train_prior: float, test_prior: float) -> np.ndarray:
    """Re-weight calibrated probabilities from the train prior to a new prior (the EM fixed-point map)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-9, 1 - 1e-9)
    a, b = test_prior / train_prior, (1 - test_prior) / (1 - train_prior)
    return a * p / (a * p + b * (1 - p))
