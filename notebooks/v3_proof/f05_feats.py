"""Signature features for every (lid, rid) with p2 >= 1e-3 of a v2 stage-2 prediction file (tune / holdout).

Vocabulary log-odds (inserted words, replacement pairs / targets, deleted words) are learned ONLY from the fit-group
samples of f02 (dist_near vs true). Output: out/f05_feat_<split>.parquet = lid, rid, p, label + features.
usage: python f05_feats.py tune|holdout
"""
import json
import math
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from f_common import C, OUT, RUN, connect, raw_right
from sigdiff import pair_ops

S_OFF = {3, 4, 5, 7, 9, 11, 13, 21}  # distractor offsets (learned from fit sample, finding 1), +1/+2 kept apart
MINC = 10


def learn_vocab() -> dict:
    """Smoothed log-odds (distractor vs true) of name edit tokens from the fit-group f02 samples."""
    cols = ["n_ins_w", "n_del_w", "n_repl_w", "a_ins_w", "a_del_w"]
    dt = pq.read_table(OUT / "f02_ops_dist.parquet", columns=cols + ["best_s"])
    keep = [s >= 1.7 for s in dt.column("best_s").to_pylist()]
    d = {c: [x for x, k in zip(dt.column(c).to_pylist(), keep) if k] for c in cols}
    tt = pq.read_table(OUT / "f02_ops_true.parquet", columns=cols)
    t = {c: tt.column(c).to_pylist() for c in cols}
    prior = math.log(len(t["n_ins_w"]) / len(d["n_ins_w"]))
    voc = {}
    for col, key in (("n_ins_w", "ins"), ("n_del_w", "del"), ("n_repl_w", "repl"), ("a_ins_w", "ains"), ("a_del_w", "adel")):
        cd = Counter(w for ws in d[col] for w in ws)
        ct = Counter(w for ws in t[col] for w in ws)
        voc[key] = {w: round(math.log((cd[w] + 1) / (ct[w] + 1)) + prior, 3)
                    for w in set(cd) | set(ct) if cd[w] + ct[w] >= MINC}
        if key == "repl":
            cd2 = Counter(w.split(">")[1] for ws in d[col] for w in ws)
            ct2 = Counter(w.split(">")[1] for ws in t[col] for w in ws)
            voc["rtgt"] = {w: round(math.log((cd2[w] + 1) / (ct2[w] + 1)) + prior, 3)
                           for w in set(cd2) | set(ct2) if cd2[w] + ct2[w] >= MINC}
    return voc


VOC = None


def init(v):
    """Pool initializer: share the vocab dict."""
    global VOC
    VOC = v


def feats(row) -> list:
    """Feature vector of one (ln, la, rn, ra) pair."""
    ln, la, rn, ra = row
    d = pair_ops(ln or "", la or "", rn or "", ra or "")
    n = d["name"]
    nan = float("nan")
    if n is None:
        nf = [nan] * 16
    else:
        ins = [VOC["ins"].get(w, 0.0) for w, _, _ in n["inserted"]]
        rep = [VOC["repl"].get(f"{x}>{y}", VOC["rtgt"].get(y, 0.0)) for x, y in n["replaced"]]
        rtg = [VOC["rtgt"].get(y, 0.0) for _, y in n["replaced"]]
        dl = [VOC["del"].get(w, 0.0) for w in n["deleted"]]
        nf = [n["typo"], n["abbrev"], n["trunc"], n.get("concat", 0), n["reorder"], len(n["replaced"]), len(n["inserted"]),
              len(n["deleted"]), n["n_common"], max(ins, default=0.0), sum(ins), max(rep, default=0.0), sum(rep),
              max(rtg, default=0.0), min(dl, default=0.0), max(ins + rep, default=0.0)]
    a = d["addr"]
    if a is None:
        af = [nan] * 7
        mf = [nan] * 17
    else:
        ai = [VOC["ains"].get(w, 0.0) for w, _, _ in a["inserted"]]
        ad = [VOC["adel"].get(w, 0.0) for w in a["deleted"]]
        af = [a["typo"], a["abbrev"], len(a["replaced"]), len(a["inserted"]), len(a["deleted"]), max(ai, default=0.0),
              max(ad, default=0.0)]
        m = d["num"]
        rel = [c for (c, _), _, _ in m["pairs"]]
        off = [v for (c, v), _, _ in m["pairs"] if c == "offset"]
        first = next(((v, i) for (c, v), i, _ in m["pairs"] if c in ("offset", "add_digit", "transpose")), (nan, -1))
        mf = [m["nl"], m["nr"], m["n_eq"], m["l_only"], m["r_only"], len(rel),
              sum(v in S_OFF for v in off), sum(v in (1, 2) for v in off), sum(v in (-1, -2) for v in off),
              sum(v not in S_OFF and v not in (1, 2, -1, -2) for v in off),
              rel.count("trunc_last") + rel.count("drop_first") + rel.count("drop_inner"), rel.count("add_digit"),
              rel.count("transpose"), max(-1000.0, min(1000.0, first[0])) if first[0] == first[0] else nan, first[1],
              m["pad"], float(len(rel) == 1 and len(off) == 1 and off[0] in S_OFF)]
    return [float(d["r_name_nonlatin"]), float(d["name_exact"]), float(d["name_case_only"]), float(d["r_addr_empty"]),
            d["name_ntok"][0], d["name_ntok"][1]] + nf + af + mf


COLS = (["r_nonlatin", "name_exact", "case_only", "r_addr_empty", "l_ntok", "r_ntok",
         "n_typo", "n_abbrev", "n_trunc", "n_concat", "n_reorder", "n_repl", "n_ins", "n_del", "n_common",
         "ins_lo_max", "ins_lo_sum", "repl_lo_max", "repl_lo_sum", "rtgt_lo_max", "del_lo_min", "edit_lo_max",
         "a_typo", "a_abbrev", "a_repl", "a_ins", "a_del", "a_ins_lo_max", "a_del_lo_max",
         "num_nl", "num_nr", "num_eq", "num_l_only", "num_r_only", "num_changed", "off_in_S", "off_p12", "off_m12",
         "off_other", "num_trunc", "num_add_digit", "num_transpose", "first_off", "first_off_idx", "num_pad",
         "sole_off_in_S"])


def main(split: str) -> None:
    """Compute features for pred2_<split> (train group) in 250k-row chunks and write one parquet."""
    vf = OUT / "f05_vocab.json"
    if not vf.exists():
        vf.write_text(json.dumps(learn_vocab()))
    voc = json.loads(vf.read_text())
    con = connect()
    src = "train"
    con.execute(f"CREATE TEMP TABLE pr AS SELECT lid, rid, p, label FROM read_parquet('{RUN}/pred2_{split}.parquet') WHERE p >= 1e-3")
    con.execute(f"""CREATE TEMP TABLE rr AS SELECT entity_id, business_name, business_address FROM {raw_right(src)}
                    WHERE entity_id IN (SELECT DISTINCT rid FROM pr)""")
    con.execute(f"""CREATE TEMP TABLE ll AS SELECT entity_id, business_name, business_address, country
                    FROM read_parquet('{C}/raw_{src}_s1.parquet') WHERE entity_id IN (SELECT DISTINCT lid FROM pr)""")
    rel = con.execute("""SELECT pr.lid, pr.rid, pr.p, pr.label, l.country, l.business_name ln, l.business_address la,
                         r.business_name rn, r.business_address ra
                         FROM pr JOIN ll l ON l.entity_id = pr.lid JOIN rr r ON r.entity_id = pr.rid""")
    reader = rel.fetch_record_batch(250_000)
    writer = None
    with Pool(3, initializer=init, initargs=(voc,)) as pool:
        for i, b in enumerate(reader):
            rows = list(zip(*(b.column(c).to_pylist() for c in ("ln", "la", "rn", "ra"))))
            F = np.array(pool.map(feats, rows, chunksize=2000), dtype=np.float32)
            t = pa.table({"lid": b.column("lid"), "rid": b.column("rid"), "p": b.column("p"), "label": b.column("label"),
                          "country": b.column("country"), **{c: F[:, j] for j, c in enumerate(COLS)}})
            if writer is None:
                writer = pq.ParquetWriter(OUT / f"f05_feat_{split}.parquet.tmp", t.schema)
            writer.write_table(t)
            print(split, "chunk", i, flush=True)
    writer.close()
    (OUT / f"f05_feat_{split}.parquet.tmp").rename(OUT / f"f05_feat_{split}.parquet")


if __name__ == "__main__":
    main(sys.argv[1])
