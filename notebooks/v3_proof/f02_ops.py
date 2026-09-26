"""Compute raw-text edit operations for the f01 samples (distractor-parent pairs and true pairs).

Output: out/f02_ops_{dist,true}.parquet, one row per pair with flat op columns and op vocab lists.
"""
from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from f_common import OUT
from sigdiff import pair_ops


def flat(row) -> dict:
    """Flatten pair_ops output of one (lid, rid, ln, la, rn, ra) row into scalar / list columns."""
    ln, la, rn, ra = row
    d = pair_ops(ln or "", la or "", rn or "", ra or "")
    o = {"nonlatin": d["r_name_nonlatin"], "name_exact": d["name_exact"], "case_only": d["name_case_only"],
         "addr_empty": d["r_addr_empty"], "l_ntok": d["name_ntok"][0], "r_ntok": d["name_ntok"][1]}
    n = d["name"]
    if n is None:
        n = {"typo": None, "abbrev": None, "trunc": None, "replaced": [], "inserted": [], "deleted": [], "reorder": None,
             "n_common": None}
    o.update(n_typo=n["typo"], n_abbrev=n["abbrev"], n_trunc=n["trunc"], n_concat=n.get("concat", 0),
             n_reorder=n["reorder"], n_common=n["n_common"], n_repl=len(n["replaced"]), n_ins=len(n["inserted"]),
             n_del=len(n["deleted"]), n_repl_w=[f"{x}>{y}" for x, y in n["replaced"]],
             n_ins_w=[y for y, _, _ in n["inserted"]], n_ins_pos=[p for _, p, _ in n["inserted"]],
             n_ins_dup=[dup for _, _, dup in n["inserted"]], n_del_w=list(n["deleted"]))
    a = d["addr"]
    if a is None:
        o.update(a_typo=None, a_abbrev=None, a_repl=None, a_ins=None, a_del=None, a_repl_w=[], a_ins_w=[], a_del_w=[],
                 nl=None, nr=None, n_eq=None, l_only=None, r_only=None, pad=None, rel=[], off=[], li=[], ri=[])
    else:
        m = d["num"]
        o.update(a_typo=a["typo"], a_abbrev=a["abbrev"], a_repl=len(a["replaced"]), a_ins=len(a["inserted"]),
                 a_del=len(a["deleted"]), a_repl_w=[f"{x}>{y}" for x, y in a["replaced"]],
                 a_ins_w=[y for y, _, _ in a["inserted"]], a_del_w=list(a["deleted"]),
                 nl=m["nl"], nr=m["nr"], n_eq=m["n_eq"], l_only=m["l_only"], r_only=m["r_only"], pad=m["pad"],
                 rel=[c for (c, _), _, _ in m["pairs"]], off=[v for (_, v), _, _ in m["pairs"]],
                 li=[i for _, i, _ in m["pairs"]], ri=[j for _, _, j in m["pairs"]])
    return o


def run(tag: str) -> None:
    """Compute ops for out/f01_<tag>.parquet and write out/f02_ops_<tag>.parquet."""
    t = pq.read_table(OUT / f"f01_{tag}.parquet")
    rows = list(zip(*(t.column(c).to_pylist() for c in ("ln", "la", "rn", "ra"))))
    with Pool(3) as pool:
        recs = pool.map(flat, rows, chunksize=2000)
    ops = pa.Table.from_pylist(recs)
    keep = [c for c in t.column_names if c not in ()]
    for c in keep:
        ops = ops.append_column(c, t.column(c))
    pq.write_table(ops, OUT / f"f02_ops_{tag}.parquet")
    print(tag, ops.num_rows, flush=True)


if __name__ == "__main__":
    for tag in ("dist", "true"):
        run(tag)
