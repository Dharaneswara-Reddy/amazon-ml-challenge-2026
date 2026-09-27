"""Shared paths/helpers for the v4 transfer study (train on one country, evaluate on another)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
CACHE = ROOT / "artifacts/cache"
TMP = ROOT / "artifacts/v4_tmp"
TMP.mkdir(parents=True, exist_ok=True)
NOTES = ROOT / "research_notes/v4/I_transfer.md"
FEAT = {g: CACHE / f"feat_v3_train_{g}.parquet" for g in ("fit", "tune", "holdout")}
S1 = CACHE / "norm_train_s1.parquet"
R = [CACHE / "norm_train_s2.parquet", CACHE / "norm_train_s3.parquet"]

TD_LO = ["nd_l_logodds_max", "nd_l_logodds_sum", "nd_r_logodds_max", "nd_r_logodds_sum"]
SG_LO = ["sg_ins_lo_max", "sg_ins_lo_sum", "sg_repl_lo_max", "sg_repl_lo_sum", "sg_rtgt_lo_max", "sg_del_lo_min",
         "sg_edit_lo_max", "sg_a_ins_lo_max", "sg_a_del_lo_max"]
VOCAB_COLS = TD_LO + SG_LO
FREQ_COLS = ["l_name_freq", "r_name_freq", "r_core_l_freq"]


def connect(mem: str = "3GB"):
    """DuckDB connection with the study's memory/thread/temp limits."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET memory_limit='{mem}'; SET threads=6; SET temp_directory='{TMP}'; SET preserve_insertion_order=true")
    return con


def bucket(col: str) -> str:
    """md5 bucket SQL of an id column (same as ber.split)."""
    return f"(md5_number({col}) % 100)"


def note(text: str) -> None:
    """Append a block of text to the findings file immediately."""
    with open(NOTES, "a") as fh:
        fh.write(text.rstrip() + "\n\n")
