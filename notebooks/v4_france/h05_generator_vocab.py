"""Isolate the French distractor NAME-edit vocabulary: rid-best D1 pairs (number-anchored distractors vs their parent)
against T3+T4 true copies; then estimate how many T4 / accepted French pairs carry a generator word (name-only
distractors that the number signature cannot see). Same procedure on train for reference (labels known).
"""
import math
import multiprocessing as mp
import sys
from collections import Counter

from h_common import C, TMP, connect, norm_right

sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber import signature  # noqa: E402

N = 150_000


def ops(row):
    """(group, label, inserted name tokens, replaced pairs) of one pair."""
    g, lab, ln, rn = row
    d = signature.token_diff(signature.name_tokens(ln or ""), signature.name_tokens(rn or ""))
    return g, lab, [w for w, _, _ in d["inserted"]], [f"{x}>{y}" for x, y in d["replaced"]], [y for _, y in d["replaced"]]


def fetch(con, path, split):
    """Sampled D1-best / T3 / T4 / accepted (rid-best p>=0.69) pairs with raw-ish normalized names."""
    num = ("coalesce(sg_num_changed,1)=0 AND coalesce(sg_num_l_only,1)=0 AND coalesce(sg_num_r_only,1)=0 "
           "AND coalesce(sg_num_eq,0)>=1")
    grp = f"""CASE WHEN sg_sole_off_in_s=1 AND r_rank=1 THEN 'D1b'
        WHEN {num} AND r_rank=1 AND p>=0.5 AND sg_n_repl=0 AND sg_n_ins=0 THEN 'T3'
        WHEN {num} AND r_rank=1 AND p>=0.5 THEN 'T4'
        WHEN r_rank=1 AND p>=0.69 AND sg_num_changed=1 AND sg_first_off IN (1,2) THEN 'P12'
        WHEN r_rank=1 AND p>=0.69 THEN 'ACC' END"""
    return con.execute(f"""WITH s AS (SELECT lid, rid, label, {grp} g FROM read_parquet('{path}')),
        k AS (SELECT *, row_number() OVER (PARTITION BY g ORDER BY hash(lid||rid)) k FROM s WHERE g IS NOT NULL)
        SELECT g, coalesce(label,-1), l.name_norm, r.name_norm FROM k
        JOIN read_parquet('{C}/norm_{split}_s1.parquet') l ON l.entity_id=k.lid JOIN {norm_right(split)} r ON r.entity_id=k.rid
        WHERE k.k <= {N}""").fetchall()


def run(con, path, split, tag):
    rows = fetch(con, path, split)
    with mp.get_context("fork").Pool(4) as pool:
        res = pool.map(ops, rows, chunksize=5000)
    n = Counter(); cnt = {}
    for g, lab, ins, rep, tgt in res:
        n[g] += 1
        c = cnt.setdefault(g, {"ins": Counter(), "repl": Counter(), "rtgt": Counter()})
        c["ins"].update(set(ins)); c["repl"].update(set(rep)); c["rtgt"].update(set(tgt))
    print(tag, "groups", dict(n))
    voc = {}
    for k in ("ins", "repl", "rtgt"):
        d, t = cnt["D1b"][k], cnt["T3"][k] + cnt["T4"][k]
        nd, nt = n["D1b"], n["T3"] + n["T4"]
        voc[k] = {w: math.log((d[w] + 1) / nd) - math.log((t[w] + 1) / nt) for w in set(d) | set(t) if d[w] + t[w] >= 20}
        top = sorted(voc[k].items(), key=lambda x: -x[1])[:40]
        print(f"{tag} {k} top generator words (lo, share of D1b pairs, share of T pairs):",
              [(w, round(v, 1), round(d[w] / nd, 4), round(t[w] / nt, 5)) for w, v in top])
    gen = {k: {w for w, v in voc[k].items() if v >= 3.0} for k in voc}
    for g in ("D1b", "T3", "T4", "P12", "ACC"):
        hit = Counter()
        for gg, lab, ins, rep, tgt in res:
            if gg != g:
                continue
            h = bool(set(ins) & gen["ins"]) or bool(set(rep) & gen["repl"]) or bool(set(tgt) & gen["rtgt"])
            hit[(h, lab)] += 1
        tot = sum(hit.values())
        print(f"{tag} {g}: share with a generator word (lo>=3): {sum(v for (h, _), v in hit.items() if h) / max(tot, 1):.4f}",
              "by (hit,label):", dict(hit), flush=True)
    return voc


if __name__ == "__main__":
    con = connect()
    run(con, f"{TMP}/pairs_train.parquet", "train", "TRAIN")
    run(con, f"{TMP}/pairs_fr.parquet", "test", "FR")
