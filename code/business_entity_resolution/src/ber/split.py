"""Deterministic Source-1 groups for fitting, threshold tuning, and held-out scoring.

Groups are disjoint ranges of md5(entity_id) mod 100, so every pair of one Source 1 entity lands in the same group
(grouped split, no leakage across groups) and the assignment is identical on every machine.
"""

from __future__ import annotations

GROUPS: dict[str, tuple[int, int]] = {"fit": (0, 10), "tune": (10, 15), "holdout": (15, 20)}


def bucket_sql(col: str) -> str:
    """SQL expression mapping an id column to a stable bucket in [0, 100)."""
    return f"(md5_number({col}) % 100)"


def group_where(col: str, *groups: str) -> str:
    """SQL predicate selecting ids of the given named groups."""
    parts = [f"({bucket_sql(col)} >= {GROUPS[g][0]} AND {bucket_sql(col)} < {GROUPS[g][1]})" for g in groups]
    return "(" + " OR ".join(parts) + ")"
