"""Shared helpers for v5 France-shift research (paths, DuckDB connection, slice definitions)."""
import sys
from pathlib import Path

import duckdb

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
RUN = ROOT / "artifacts/runs/20260926T063247_1a0d20f_99304c0b_v4"
CACHE = ROOT / "artifacts/cache"
TMP = ROOT / "artifacts/v5_tmp"
OUT = ROOT / "notebooks/v5_france/out"
NOTES = ROOT / "research_notes/v5/J_france_shift.md"
FEAT = {g: CACHE / f"feat_v4_train_{g}.parquet" for g in ("fit", "tune", "holdout", "fitA", "fitB")}
FEAT["test"] = CACHE / "feat_v4_test.parquet"
SUB6 = ROOT / "submissions/06_20260926-1234/matching_results.tsv"
# the "clean near-identical top candidate" slice (numbers identical, same core name, best block candidate of the right)
CLEAN = ("f.sg_num_changed = 0 AND f.sg_num_l_only = 0 AND f.sg_num_r_only = 0 AND f.sg_num_eq >= 1 "
         "AND f.core_ratio >= 0.95 AND f.r_rank = 1")


def connect(threads: int = 5, mem: str = "3GB") -> duckdb.DuckDBPyConnection:
    """DuckDB connection with the shared-laptop limits and spill dir under artifacts/v5_tmp."""
    TMP.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{mem}'; SET threads={threads}; SET temp_directory='{TMP}/spill'; "
              "SET preserve_insertion_order=false")
    return c


def note(text: str) -> None:
    """Append a finding to the research notes file (incremental, survives session cut-offs)."""
    with open(NOTES, "a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n\n")
    print(text)


def auc(y, s):
    """ROC AUC via ranks (ties averaged); y in {0,1}."""
    import numpy as np
    from scipy.stats import rankdata
    y = np.asarray(y).astype(bool); r = rankdata(s)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
