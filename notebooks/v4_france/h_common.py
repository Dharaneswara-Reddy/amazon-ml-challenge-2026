"""Shared paths and a memory-capped DuckDB connection for the v4 France research scripts (h*.py)."""
import sys
from pathlib import Path

import duckdb

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
C = ROOT / "artifacts/cache"
RUN = ROOT / "artifacts/runs/20260926T032546_e2e7494_678411c5_v3"
TMP = ROOT / "artifacts/v4_tmp"
NOTES = ROOT / "research_notes/v4/H_france.md"


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with the agreed limits (3 GB, 5 threads, spill under artifacts/v4_tmp)."""
    (TMP / "spill").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='3GB'; SET threads=5; SET temp_directory='{TMP}/spill'; "
                "SET preserve_insertion_order=false")
    return con


def raw_right(split: str = "test") -> str:
    """SQL for the union of raw S2 and S3 rows of a split."""
    return (f"(SELECT * FROM read_parquet('{C}/raw_{split}_s2.parquet') UNION ALL "
            f"SELECT * FROM read_parquet('{C}/raw_{split}_s3.parquet'))")


def norm_right(split: str = "test") -> str:
    """SQL for the union of normalized S2 and S3 rows of a split."""
    return (f"(SELECT * FROM read_parquet('{C}/norm_{split}_s2.parquet') UNION ALL "
            f"SELECT * FROM read_parquet('{C}/norm_{split}_s3.parquet'))")
