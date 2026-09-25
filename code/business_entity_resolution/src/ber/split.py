"""Deterministic Source-1 groups for fitting, stacking folds, threshold tuning, and held-out scoring (decision D-004).

Groups are disjoint ranges of md5(entity_id) mod 100, so every pair of one Source 1 entity lands in the same group
(grouped split, no leakage across groups) and the assignment is identical on every machine and independent of file
order. v2 layout: two stacking folds fitA/fitB (together "fit"), tune for early stopping and the decision rule,
holdout touched only to score an experiment; buckets 40-99 are unused spare.
"""

from __future__ import annotations

GROUPS: dict[str, tuple[int, int]] = {
    "fitA": (0, 15),
    "fitB": (15, 30),
    "fit": (0, 30),
    "tune": (30, 35),
    "holdout": (35, 40),
}
FOLDS = ("fitA", "fitB")


def bucket_sql(col: str) -> str:
    """SQL expression mapping an id column to a stable bucket in [0, 100)."""
    return f"(md5_number({col}) % 100)"


def group_where(col: str, *groups: str) -> str:
    """SQL predicate selecting ids of the given named groups."""
    parts = [f"({bucket_sql(col)} >= {GROUPS[g][0]} AND {bucket_sql(col)} < {GROUPS[g][1]})" for g in groups]
    return "(" + " OR ".join(parts) + ")"


def other_fold(fold: str) -> str:
    """The stacking fold whose statistics/models are applied out-of-fold to ``fold``."""
    return {"fitA": "fitB", "fitB": "fitA"}[fold]
