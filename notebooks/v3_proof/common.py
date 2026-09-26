"""Shared paths and helpers for the v3 end-to-end proof (research only; imports the frozen ber package read-only)."""
import os, sys, time, json
from pathlib import Path

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
C = ROOT / "artifacts/cache"
T = ROOT / "artifacts/v3_proof_tmp"
DB = T / "p.duckdb"
NOTE = ROOT / "research_notes/v3/E_proof_end_to_end.md"
SUBSETS = {"fitsub": (0, 5), "tune": (30, 35), "holdout": (35, 40)}
PASSES = ["px", "hs", "n2", "na", "phx"]


def connect(db=DB, mem="5GB", threads=8, read_only=False):
    """DuckDB connection with the agreed memory/thread/spill limits."""
    import duckdb
    con = duckdb.connect(str(db), read_only=read_only)
    (T / "spill").mkdir(parents=True, exist_ok=True)
    con.execute(f"SET memory_limit='{mem}'; SET threads={threads}; SET preserve_insertion_order=false; "
                f"SET temp_directory='{T / 'spill'}'; SET max_temp_directory_size='40GiB'")
    return con


def bucket_pred(col, name):
    """SQL predicate: md5 bucket of ``col`` inside subset ``name``."""
    a, b = SUBSETS[name]
    return f"(md5_number({col}) % 100 >= {a} AND md5_number({col}) % 100 < {b})"


def note(text):
    """Append a block of markdown to the results file immediately (survives session cut-offs)."""
    with open(NOTE, "a") as f:
        f.write(text.rstrip() + "\n")


def log(*a):
    """Timestamped print, flushed."""
    print(time.strftime("%H:%M:%S"), *a, flush=True)
