"""Shared paths and a memory-capped DuckDB connection for the F (generator signature) scripts (f*.py).

Named f_common to avoid clashing with common.py of the parallel v3 end-to-end proof agent in this directory.
"""
import sys
from pathlib import Path

import duckdb

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
C = ROOT / "artifacts/cache"
RUN = ROOT / "artifacts/runs/20260925T161436_b538130_1242e7cd_v2"
OUT = ROOT / "notebooks/v3_proof/out"
TMP = ROOT / "artifacts/v3_proof_tmp/f_spill"
NOTES = ROOT / "research_notes/v3/F_generator_signature.md"


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with the agreed limits (2.5 GB, 4 threads, temp files on disk under artifacts/)."""
    TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='2500MB'; SET threads=4; SET temp_directory='{TMP}'; SET preserve_insertion_order=false")
    return con


def raw_right(split: str = "train") -> str:
    """SQL for the union of raw S2 and S3 rows of a split."""
    return (f"(SELECT * FROM read_parquet('{C}/raw_{split}_s2.parquet') UNION ALL "
            f"SELECT * FROM read_parquet('{C}/raw_{split}_s3.parquet'))")
