"""Tests for ber.decision: metric, one-to-one selection, threshold tuning, calibration, expected-F0.5 selection."""

import itertools
import json

import numpy as np
import pytest

from ber.decision import (
    Calibrator,
    MissingModel,
    best_per_right,
    compare_decisions,
    decide,
    expected_f_curve,
    expected_f_select,
    fit_calibrator,
    fit_missing_model,
    macro_f05,
    pava,
    select,
    tune_threshold,
)


def test_organizer_example_scores_0714() -> None:
    """Predict [S2-00047, S2-00193, S3-00812] vs truth [S2-00047, S3-00812] gives 0.714."""
    lid = np.array([0, 0, 0])
    selected = np.array([True, True, True])
    label = np.array([True, False, True])
    assert macro_f05(lid, selected, label, np.array([2.0])) == pytest.approx(0.7142857, abs=1e-6)


def test_singletons_and_missed_entities() -> None:
    """Singleton: empty → 1.0, any match → 0.0. Matched entity with empty prediction → 0.0."""
    lid = np.array([0, 1, 2])
    label = np.array([False, False, True])
    n_true = np.array([0.0, 0.0, 1.0, 3.0])
    selected = np.array([False, True, True])
    assert macro_f05(lid, selected, label, n_true) == pytest.approx((1.0 + 0.0 + 1.25 / 1.25 + 0.0) / 4)


def test_best_per_right_keeps_one_pair_per_right_record() -> None:
    """Each right id keeps only its highest-probability left candidate; ties go to the smaller lid."""
    rid = np.array(["r1", "r1", "r2", "r2", "r3"])
    lid = np.array([0, 1, 0, 1, 1])
    p = np.array([0.4, 0.9, 0.5, 0.5, 0.2])
    assert best_per_right(rid, lid, p).tolist() == [False, True, True, False, True]


def test_select_applies_threshold_after_one_to_one() -> None:
    """A right record's runner-up is never selected even if above threshold."""
    rid = np.array(["r1", "r1"])
    lid = np.array([0, 1])
    p = np.array([0.95, 0.9])
    assert select(lid, rid, p, 0.5).tolist() == [True, False]


def test_tune_threshold_prefers_precision() -> None:
    """With a confident true pair and a mid-probability false pair on a singleton, the tuned threshold drops the FP."""
    lid = np.array([0, 1])
    rid = np.array(["r1", "r2"])
    p = np.array([0.9, 0.6])
    label = np.array([True, False])
    n_true = np.array([1.0, 0.0])
    t, curve = tune_threshold(lid, rid, p, label, n_true)
    assert 0.6 < t <= 0.9
    assert max(s for _, s in curve) == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------------------------
# Calibration, expected-F0.5 selection, and a simulation against the global threshold
# ---------------------------------------------------------------------------------------------------------------


def test_pava_known_example_and_properties() -> None:
    """PAVA pools violators, is non-decreasing, preserves the weighted mean, and leaves sorted input unchanged."""
    assert pava(np.array([1, 3, 2, 4, 3, 5])).tolist() == [1, 2.5, 2.5, 3.5, 3.5, 5]
    rng = np.random.default_rng(0)
    y, w = rng.random(500), rng.random(500) + 0.1
    fit = pava(y, w)
    assert np.all(np.diff(fit) >= -1e-12)
    assert np.average(fit, weights=w) == pytest.approx(np.average(y, weights=w))
    assert pava(np.sort(y)).tolist() == pytest.approx(np.sort(y).tolist())


def test_calibrator_is_monotone_and_recovers_probabilities() -> None:
    """Isotonic calibration of miscalibrated scores is monotone on a fine grid and close to the true curve."""
    rng = np.random.default_rng(1)
    s = rng.random(200_000)
    y = rng.random(len(s)) < s**3  # scores overstate the match probability
    cal = fit_calibrator(s, y, max_bins=512)
    grid = np.linspace(-0.1, 1.1, 2001)
    out = cal(grid)
    assert np.all(np.diff(out) >= 0)
    assert 0 <= out.min() and out.max() <= 1
    mid = (grid > 0.05) & (grid < 0.95)
    assert np.abs(out[mid] - grid[mid] ** 3).max() < 0.05
    again = Calibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert again(grid) == pytest.approx(out)


def _exact_expected_f(p: np.ndarray, missing_pmf: dict[int, float]) -> np.ndarray:
    """Brute-force E[F0.5] for k = 0..m (top-k by p) with independent labels and missing count M ~ pmf."""
    p = np.sort(p)[::-1]
    out = np.zeros(len(p) + 1)
    for ys in itertools.product([0, 1], repeat=len(p)):
        y = np.array(ys)
        pr = float(np.prod(np.where(y == 1, p, 1 - p)))
        for mm, pm in missing_pmf.items():
            n_true = y.sum() + mm
            for k in range(len(p) + 1):
                f = 1.0 if k == 0 and n_true == 0 else 1.25 * y[:k].sum() / (k + 0.25 * n_true)
                out[k] += pr * pm * f
    return out


def test_expected_f_approximation_matches_brute_force() -> None:
    """Second-order expected-F0.5 is within 0.005 of enumeration and its chosen k loses < 0.001 expected F."""
    rng = np.random.default_rng(2)
    worst_err = worst_regret = 0.0
    for trial in range(200):
        p = rng.beta(0.6, 0.6, int(rng.integers(1, 7)))
        pmf = {0: 1.0} if trial % 2 == 0 else {0: 0.7, 1: 0.2, 2: 0.1}
        mean = sum(k * v for k, v in pmf.items())
        var = sum(k * k * v for k, v in pmf.items()) - mean**2
        exact = _exact_expected_f(p, pmf)
        approx = expected_f_curve(p, mean, var, pmf[0])
        worst_err = max(worst_err, float(np.abs(exact - approx).max()))
        worst_regret = max(worst_regret, float(exact.max() - exact[np.argmax(approx)]))
    assert worst_err < 0.005
    assert worst_regret < 0.001
    assert expected_f_curve(np.array([0.7]))[1] == pytest.approx(0.7)  # lone candidate: exact


def test_expected_f_empty_for_weak_entities_and_keeps_confident_matches() -> None:
    """Low-confidence candidates -> empty prediction; confident ones kept; a runner-up right record never picked."""
    lid = np.array([0, 0, 1, 1, 1, 1, 2, 3])
    rid = np.array(["a", "b", "c", "d", "e", "f", "g", "c"])
    p = np.array([0.3, 0.2, 0.97, 0.95, 0.9, 0.1, 0.45, 0.5])
    mask = expected_f_select(lid, rid, p)
    assert mask.tolist() == [False, False, True, True, True, False, False, False]
    curve = expected_f_curve(np.array([0.3, 0.2]))
    assert curve.argmax() == 0 and curve[0] == pytest.approx(0.7 * 0.8)


def test_expected_f_adapts_the_bar_per_entity() -> None:
    """A mid candidate is added next to strong matches but a weak one is not; the bar is per entity."""
    assert expected_f_curve(np.array([0.6])).argmax() == 1
    assert expected_f_curve(np.array([0.99, 0.99, 0.99, 0.2])).argmax() == 3
    assert expected_f_curve(np.array([0.99, 0.99, 0.99, 0.85])).argmax() == 4
    lid = np.array([0, 1])
    assert expected_f_select(lid, np.array(["a", "b"]), np.array([0.45, 0.55])).tolist() == [False, True]


def test_missing_model_separates_entities() -> None:
    """Blocking misses: E[M] per bin among found-something entities, P(M=0) among found-nothing ones."""
    lid = np.array([0, 1, 2, 2, 3])
    p = np.array([0.1, 0.1, 0.9, 0.9, 0.9])
    label = np.array([False, False, True, True, True])
    n_true = np.array([0.0, 3.0, 3.0, 1.0])
    m = fit_missing_model(lid, p, label, n_true, n_bins=2, strength=0.0)
    mean, var, p0 = m.lookup(np.array([0.1, 1.8]))
    assert p0[0] == pytest.approx(0.5)  # entities 0 (singleton) and 1 (all 3 missed)
    assert mean[1] == pytest.approx(0.5) and var[1] == pytest.approx(0.25)  # entity 2 misses 1, entity 3 misses 0
    again = MissingModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert again.lookup(np.array([1.8]))[0] == pytest.approx(mean[1:])


def simulate_group(rng: np.random.Generator, n_ent: int, singleton: float = 0.056) -> dict:
    """Synthetic labelled group: known true sets, 7% blocking misses, rival S1s sharing right records, noisy scores.

    Non-singletons have 1 + Binomial(6, 0.45) matches (mean 3.7); each entity gets Poisson(5) false candidates, 15%
    of them hard (score around 0.5) and 30% reusing a true right record of another entity; an entity-level random
    effect makes some entities harder. Returns the dict layout of pipeline._group_arrays with raw scores as p.
    """
    n_true = np.where(rng.random(n_ent) < singleton, 0, 1 + rng.binomial(6, 0.45, n_ent))
    eff = rng.normal(0, 0.8, n_ent)
    t_l = np.repeat(np.arange(n_ent), n_true)
    t_l = t_l[rng.random(len(t_l)) < 0.93]
    t_s = 3.0 + eff[t_l] + rng.normal(0, 1.8, len(t_l))
    f_l = np.repeat(np.arange(n_ent), rng.poisson(5.0, n_ent))
    f_s = np.where(rng.random(len(f_l)) < 0.15, 0.0, -4.0) + rng.normal(0, 1.8, len(f_l))
    f_r = len(t_l) + np.arange(len(f_l))
    steal = rng.random(len(f_l)) < 0.3
    f_r[steal] = rng.integers(0, len(t_l), steal.sum())
    ok = ~(steal & (t_l[np.minimum(f_r, len(t_l) - 1)] == f_l))
    s = np.r_[t_s, f_s[ok]]
    return {
        "lid_idx": np.r_[t_l, f_l[ok]].astype(np.int64),
        "rid": np.r_[np.arange(len(t_l)), f_r[ok]].astype(np.int64),
        "p": 1 / (1 + np.exp(-s)),
        "label": np.r_[np.ones(len(t_l), bool), np.zeros(ok.sum(), bool)],
        "n_true": n_true.astype(np.float64),
    }


@pytest.mark.parametrize("singleton", [0.056, 0.2])
def test_simulation_expected_f_beats_global_threshold(singleton: float) -> None:
    """On simulated tune/holdout groups, expected-F0.5 fitted on tune beats the tuned global threshold on holdout.

    Also checks the model's predicted mean F0.5 is close to the realized one, and that ``decide`` on JSON-round-tripped
    params reproduces the reported score of the chosen rule.
    """
    rng = np.random.default_rng(0)
    tune, holdout = simulate_group(rng, 12_000, singleton), simulate_group(rng, 12_000, singleton)
    res = compare_decisions(tune, holdout)
    h = res["holdout"]
    assert h["expected_f_model"] > h["threshold"], h
    assert h["expected_f"] > h["threshold"], h
    assert abs(res["holdout_expected_pred"] - res["holdout_expected_real"]) < 0.03
    params = json.loads(json.dumps(res["params"]))
    sel = decide(params, holdout["lid_idx"], holdout["rid"], holdout["p"])
    assert macro_f05(holdout["lid_idx"], sel, holdout["label"], holdout["n_true"]) == pytest.approx(h[params["rule"]])
