"""France check: do the distractor signatures appear in TEST (no labels)?

For train and test, sample right records (md5(rid)%100 < 4), take their best candidate by jw_core + jw_addr
(>= 1.7, near-duplicate) and compute the diff ops. Compare per country the share of near pairs with a sole
number offset in S, the offset histogram, vocabulary inserts (location words / org suffixes) and legal-form swaps.
Train pairs are unlabeled here on purpose (same procedure as test); train labels give the reference mix.
"""
from collections import Counter
from multiprocessing import Pool

from f_common import C, OUT, connect, raw_right
from sigdiff import pair_ops

S_OFF = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}
LOC = {"east", "west", "north", "south", "uptown", "downtown", "midtown", "eastgate", "westgate", "riverside", "southside",
       "northside", "summit", "central", "greater", "metro", "lakeside", "valley", "coastal", "harbor", "highland"}
ORG = {"group", "holdings", "partners", "industries", "overseas", "ventures", "exports", "infratech", "enterprises", "public"}


def ops(row):
    """(country, is_true or None, sigdiff ops) for one pair."""
    c, lab, ln, la, rn, ra = row
    return c, lab, pair_ops(ln or "", la or "", rn or "", ra or "")


def sample(split: str):
    """Near-duplicate best-candidate pairs of sampled right records of a split."""
    con = connect()
    lab = (f"(c.rid IN (SELECT match_id FROM read_parquet('{C}/train_gt_pairs.parquet') g WHERE g.s1_id = c.lid))"
           if split == "train" else "NULL::BOOLEAN")
    return con.execute(f"""WITH c AS (SELECT rid, arg_max(lid, coalesce(jw_core,0)+coalesce(jw_addr,0)) lid,
                                 max(coalesce(jw_core,0)+coalesce(jw_addr,0)) s
                          FROM read_parquet('{C}/cand_{split}.parquet') WHERE md5_number(rid) % 100 < 4 GROUP BY rid HAVING s >= 1.7)
        SELECT l.country, {lab}, l.business_name, l.business_address, r.business_name, r.business_address
        FROM c JOIN read_parquet('{C}/raw_{split}_s1.parquet') l ON l.entity_id = c.lid
        JOIN {raw_right(split)} r ON r.entity_id = c.rid""").fetchall()


def summarize(split, res):
    """Print per-country signature statistics."""
    by = {}
    for c, lab, d in res:
        key = (c, {True: "true", False: "nonmatch", None: "unlabeled"}[lab])
        s = by.setdefault(key, Counter())
        s["n"] += 1
        m = d["num"]
        if m is not None:
            offs = [v for (k, v), _, _ in m["pairs"] if k == "offset"]
            if len(m["pairs"]) == 1 and len(offs) == 1:
                s[f"off{offs[0]:+d}" if -3 <= offs[0] <= 25 else "off_other"] += 1
                s["sole_offset_in_S"] += offs[0] in S_OFF
                s["sole_offset_in_S_ge3"] += offs[0] in S_OFF and offs[0] >= 3
            s["num_changed"] += len(m["pairs"]) > 0
        n = d["name"]
        if n is not None:
            ins = {w for w, _, _ in n["inserted"]}
            s["ins_loc"] += bool(ins & LOC)
            s["ins_org"] += bool(ins & ORG)
            s["repl_legal_swap"] += any(x in {"llc", "inc", "ltd", "corp", "co", "private", "pvt", "limited", "sarl", "sas", "sa", "eurl", "sci", "snc"}
                                        for x, _ in n["replaced"])
            for w in ins:
                s["W_" + w] += 1
    for key in sorted(by):
        s = by[key]
        n = s["n"]
        line = {k: round(s[k] / n, 4) for k in ("num_changed", "sole_offset_in_S", "sole_offset_in_S_ge3", "ins_loc", "ins_org", "repl_legal_swap")}
        hist = {k: s[k] for k in sorted((k for k in s if k.startswith("off")), key=lambda x: (len(x), x)) if s[k] >= max(3, n // 2000)}
        words = [(k[2:], v) for k, v in s.most_common() if k.startswith("W_")][:30]
        print(split, key, "n", n, line)
        print("    offsets:", hist)
        print("    top inserted name words:", words, flush=True)


if __name__ == "__main__":
    for split in ("train", "test"):
        rows = sample(split)
        with Pool(2) as pool:
            res = pool.map(ops, rows, chunksize=2000)
        summarize(split, res)
