"""Pseudo labels (signature-anchored rules of h02) for France test and, identically, for train holdout+tune;
learn signature vocab + token log-odds from them; compare the train pseudo vocab against a true-label vocab.

Rules: y=1 'T4' nums all equal & rid-best & p>=0.5 ; y=0 'D' sole offset in S>=3 (D1) or all pairs of a right whose
best pair is D1 (D2) ; y=0 'E' p<0.01. Writes artifacts/v4_tmp/fr_pseudo.parquet (all labelled France pairs) and
sigvocab_fr.json / logodds_fr.json (+ *_trpseudo.json, *_trtrue.json for the validation).
"""
import json
import math
import multiprocessing as mp
import sys
import time
from collections import Counter

import numpy as np

from h_common import C, TMP, connect, norm_right

NUMEQ = ("coalesce(sg_num_changed,1)=0 AND coalesce(sg_num_l_only,1)=0 AND coalesce(sg_num_r_only,1)=0 "
         "AND coalesce(sg_num_eq,0)>=1")
LABEL_SQL = f"""CASE WHEN {NUMEQ} AND r_rank = 1 AND p >= 0.5 THEN 1
    WHEN sg_sole_off_in_s = 1 OR rid_d1 THEN 0 WHEN p < 0.01 THEN 0 END"""
RULE_SQL = f"""CASE WHEN {NUMEQ} AND r_rank = 1 AND p >= 0.5 THEN (CASE WHEN coalesce(sg_n_repl,1)=0 AND
    coalesce(sg_n_ins,1)=0 THEN 'T3' ELSE 'T4' END) WHEN sg_sole_off_in_s = 1 THEN 'D1' WHEN rid_d1 THEN 'D2'
    WHEN p < 0.01 THEN 'E' END"""
CONF = {"T3": 0.9995, "T4": 0.998, "D1": 0.995, "D2": 0.995, "E": 0.999}  # min train precision over India/US
PER_RULE = 350_000


def labelled(con, path: str, name: str) -> None:
    """Create table {name}(lid, rid, label, y, rule, p) of rule-labelled pairs from a pair table."""
    con.execute(f"""CREATE OR REPLACE TABLE {name} AS
      WITH t AS (SELECT x.*, coalesce(d.rid_d1, false) AS rid_d1 FROM read_parquet('{path}') x
                 LEFT JOIN (SELECT rid, true AS rid_d1 FROM read_parquet('{path}') WHERE r_rank=1 AND sg_sole_off_in_s=1) d
                 USING (rid))
      SELECT lid, rid, label, {LABEL_SQL} AS y, {RULE_SQL} AS rule, p FROM t WHERE ({LABEL_SQL}) IS NOT NULL""")


def sample_rows(con, name: str, split: str, extra: str = "") -> list[tuple]:
    """Sample up to PER_RULE pairs per rule group (T, D, E) and fetch normalized texts + pseudo and true labels."""
    return con.execute(f"""WITH s AS (SELECT *, row_number() OVER (PARTITION BY left(rule,1) ORDER BY hash(lid || rid)) k
                                     FROM {name} {extra})
        SELECT l.name_norm, l.addr_norm, r.name_norm, r.addr_norm, s.y, coalesce(s.label, -1), l.name_core, r.name_core
        FROM s JOIN read_parquet('{C}/norm_{split}_s1.parquet') l ON l.entity_id = s.lid
        JOIN {norm_right(split)} r ON r.entity_id = s.rid WHERE s.k <= {PER_RULE}""").fetchall()


def _count(chunk):
    """Signature edit-token counts and textdiff unexplained-token counts of a chunk, for pseudo (y) and true labels."""
    sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
    from ber import signature, textdiff
    keys = ("ins", "del", "repl", "rtgt", "ains", "adel")
    out = {}
    for lab_idx in (4, 5):
        sig = {s: {k: Counter() for k in keys} for s in (0, 1)}
        n = Counter()
        tok = {s: (Counter(), Counter()) for s in (0, 1)}  # (unexplained, total)
        for row in chunk:
            y = row[lab_idx]
            if y < 0:
                continue
            n[y] += 1
            toks = signature.edit_tokens(signature.pair_ops(row[0] or "", row[1] or "", row[2] or "", row[3] or ""))
            for k, ws in toks.items():
                sig[y][k].update(ws)
            lt = row[6].split() if row[6] else []
            rt = row[7].split() if row[7] else []
            cl, cr = textdiff.align_tokens(lt, rt)
            for t, c in zip(lt + rt, cl + cr):
                t = textdiff._norm_token(t)
                if t:
                    tok[y][1][t] += 1
                    if c == "unexplained":
                        tok[y][0][t] += 1
        out[lab_idx] = (sig, n, tok)
    return out


def learn(rows, lab_idx: int, workers: int = 4):
    """(sigvocab, token log-odds) from rows using label column lab_idx (4 = pseudo y, 5 = true label)."""
    chunks = [rows[i:i + 20_000] for i in range(0, len(rows), 20_000)]
    keys = ("ins", "del", "repl", "rtgt", "ains", "adel")
    sig = {s: {k: Counter() for k in keys} for s in (0, 1)}
    n = Counter()
    tok = {s: (Counter(), Counter()) for s in (0, 1)}
    with mp.get_context("fork").Pool(workers) as pool:
        for part in pool.imap_unordered(_count, chunks):
            s_, n_, t_ = part[lab_idx]
            n.update(n_)
            for y in (0, 1):
                for k in keys:
                    sig[y][k].update(s_[y][k])
                tok[y][0].update(t_[y][0])
                tok[y][1].update(t_[y][1])
    prior = math.log(max(n[1], 1) / max(n[0], 1))
    vocab = {k: {w: round(math.log((sig[0][k][w] + 1) / (sig[1][k][w] + 1)) + prior, 3)
                 for w in set(sig[0][k]) | set(sig[1][k]) if sig[0][k][w] + sig[1][k][w] >= 10} for k in keys}
    lo = {}
    for t in set(tok[0][1]) | set(tok[1][1]):
        npos, nneg = tok[1][1][t], tok[0][1][t]
        if npos + nneg < 20:
            continue
        lo[t] = math.log(((tok[0][0][t] + 1) / (nneg + 2)) / ((tok[1][0][t] + 1) / (npos + 2)))
    return vocab, lo, dict(n), sig


def compare(a: dict, b: dict, name: str) -> None:
    """Correlation / agreement of two vocab dicts on shared tokens (per sub-vocab)."""
    for k in ("ins", "repl", "rtgt", "del", "ains"):
        sh = [w for w in a.get(k, {}) if w in b.get(k, {})]
        if len(sh) < 5:
            continue
        x = np.array([a[k][w] for w in sh]); y = np.array([b[k][w] for w in sh])
        big = [w for w in sh if b[k][w] >= 3]
        agree = np.mean([a[k][w] >= 2 for w in big]) if big else float("nan")
        print(f"  {name} {k}: shared {len(sh)} corr {np.corrcoef(x, y)[0, 1]:.3f} slope {np.polyfit(y, x, 1)[0]:.2f} "
              f"| true>=3: {len(big)} of which pseudo>=2: {agree:.3f}", flush=True)


if __name__ == "__main__":
    con = connect()
    t0 = time.time()
    labelled(con, f"{TMP}/pairs_fr.parquet", "fr")
    con.execute(f"""COPY (SELECT lid, rid, y::TINYINT AS y, CASE rule {' '.join(f"WHEN '{r}' THEN {c}" for r, c in CONF.items())} END
                   AS confidence, rule, p FROM fr) TO '{TMP}/fr_pseudo.parquet' (FORMAT parquet)""")
    print("fr pseudo:", con.execute("SELECT rule, y, count(*), count(DISTINCT lid), count(DISTINCT rid) FROM fr GROUP BY ALL ORDER BY 1").fetchall(), flush=True)
    labelled(con, f"{TMP}/pairs_train.parquet", "tr")
    print("train pseudo precision:", con.execute("SELECT rule, count(*), avg((label = y)::INT) FROM tr GROUP BY 1 ORDER BY 1").fetchall(), flush=True)
    # train: pseudo-label vocab vs true-label vocab on the SAME sampled pairs
    rows = sample_rows(con, "tr", "train")
    print("train sample", len(rows), time.time() - t0, flush=True)
    vp, lp, n_p, _ = learn(rows, 4)
    # true-label vocab: same texts, but all labelled pairs of those rights would be needed; use the sample's true labels
    vt, lt, n_t, _ = learn(rows, 5)
    print("train n pseudo", n_p, "n true", n_t, time.time() - t0, flush=True)
    fit = json.loads((C / "learned/sigvocab_fit.json").read_text())
    compare(vp, vt, "trainPseudo~trainTrue(same pairs)")
    compare(vp, fit, "trainPseudo~sigvocab_fit")
    sh = [w for w in lp if w in lt]
    print("  token logodds pseudo~true corr", np.corrcoef([lp[w] for w in sh], [lt[w] for w in sh])[0, 1], len(sh))
    json.dump(vp, open(TMP / "sigvocab_trpseudo.json", "w")); json.dump(lp, open(TMP / "logodds_trpseudo.json", "w"))
    del rows
    rows = sample_rows(con, "fr", "test")
    print("fr sample", len(rows), time.time() - t0, flush=True)
    vf, lf, n_f, sigc = learn(rows, 4)
    print("fr n", n_f, {k: len(v) for k, v in vf.items()}, len(lf), time.time() - t0, flush=True)
    json.dump(vf, open(TMP / "sigvocab_fr.json", "w"), ensure_ascii=False)
    json.dump(lf, open(TMP / "logodds_fr.json", "w"), ensure_ascii=False)
    compare(vf, fit, "fr~sigvocab_fit")
    for k in ("ins", "repl", "rtgt", "del", "ains", "adel"):
        top = sorted(vf[k].items(), key=lambda x: -x[1])[:30]
        bot = sorted(vf[k].items(), key=lambda x: x[1])[:15]
        print(f"FR {k} top distractor:", [(w, v, sigc[0][k][w], sigc[1][k][w]) for w, v in top])
        print(f"FR {k} top true-noise:", [(w, v, sigc[0][k][w], sigc[1][k][w]) for w, v in bot], flush=True)
    new = {k: [w for w in vf[k] if w not in fit.get(k, {})] for k in vf}
    print("tokens new vs train vocab:", {k: len(v) for k, v in new.items()})
    print("FR token logodds top:", sorted(lf.items(), key=lambda x: -x[1])[:40])
    print("FR token logodds bottom:", sorted(lf.items(), key=lambda x: x[1])[:20])
