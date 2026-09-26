"""Shared helpers for the v5 India+US experiments: config, cached decision arrays per group and pred file."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))
from ber.config import load_config
from ber.duck import connect
from ber.pipeline import _group_arrays, _fetch_numpy
from ber import io

RUN = ROOT / "artifacts/runs/20260926T063247_1a0d20f_99304c0b_v4"
OUT = ROOT / "artifacts/v5_iu"
cfg = load_config(override=ROOT / "notebooks/v5_iu/cfg.yaml")
_con = None


def con():
    """Lazy DuckDB connection with the v5 limits (3GB, 5 threads, spill in artifacts/v5_tmp)."""
    global _con
    if _con is None:
        _con = connect(cfg)
    return _con


def arrays(pred: Path, group: str, extra_ids: bool = False) -> dict:
    """_group_arrays for a pred parquet, cached as npz in artifacts/v5_iu (keyed by path+group)."""
    key = hashlib.md5(f"{pred}|{group}|{extra_ids}".encode()).hexdigest()[:10]
    f = OUT / f"arr_{Path(pred).stem}_{group}_{key}.npz"
    if f.exists():
        return dict(np.load(f, allow_pickle=True))
    a = _group_arrays(con(), cfg, Path(pred), group)
    if extra_ids:
        c = con()
        ids = _fetch_numpy(c, f"""WITH pr AS (SELECT lid, rid, p FROM read_parquet('{pred}')),
                 best AS (SELECT rid, arg_max(lid, p) AS best_lid FROM pr GROUP BY rid)
            SELECT s.idx, dense_rank() OVER (ORDER BY pr.rid) - 1 AS rc, pr.rid AS rid_s, pr.p
            FROM pr JOIN s1g s ON s.s1_id = pr.lid JOIN best b ON b.rid = pr.rid
            WHERE b.best_lid IN (SELECT s1_id FROM s1g)""")
        # align to a's order: sort both by (idx, rc)
        o1 = np.lexsort((a["rid"], a["lid_idx"]))
        o2 = np.lexsort((ids["rc"], ids["idx"]))
        rid_s = np.empty(len(o1), dtype=object); rid_s[o1] = ids["rid_s"][o2]
        a["rid_s"] = rid_s
        s1 = _fetch_numpy(c, f"""SELECT s.idx, s.s1_id, n.country_norm AS c FROM s1g s
            JOIN read_parquet('{io.cache_dir(cfg)}/norm_train_s1.parquet') n ON n.entity_id = s.s1_id ORDER BY s.idx""")
        a["s1_id"] = s1["s1_id"]; a["country"] = s1["c"]
    np.savez(f, **a)
    return a


def params():
    """v4 decision params (fitted on tune)."""
    return json.loads((RUN / "decision_params.json").read_text())


def note(text: str) -> None:
    """Append a block of text to the findings file (incremental, survives cut-offs)."""
    p = ROOT / "research_notes/v5/K_india_us.md"
    with p.open("a") as fh:
        fh.write(text.rstrip() + "\n\n")
