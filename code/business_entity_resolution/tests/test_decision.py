"""Tests for ber.decision: metric, one-to-one selection, threshold tuning."""

import numpy as np
import pytest

from ber.decision import best_per_right, macro_f05, select, tune_threshold


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
