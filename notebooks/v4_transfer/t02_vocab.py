"""Learn vocabularies (signature edit vocab + textdiff token log-odds) from chosen pairs/labels in DuckDB, and apply
them to the extracted token lists (t01) to recompute the 13 vocab-dependent feature columns.

Vocab variants (all learned from FIT-group in_group pairs only):
  IN_A / IN_B / IN      India, true labels, buckets 0-14 / 15-29 / 0-29
  USS                   US, self-labels from the number signature (no US labels)
  INS_A / INS_B / INS   India, self-labels (same rule), for a consistent self-labelled vocab in every country
  MRG                   IN overlaid with USS (target-country tokens override)
Output: artifacts/v4_tmp/vocab_{name}.json and vf_{group}_{variant}.parquet (lid, rid, VOCAB_COLS).
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from common import CACHE, R, S1, SG_LO, TD_LO, TMP, VOCAB_COLS, connect

SIG_KEYS = ("ins", "del", "repl", "rtgt", "ains", "adel")
# self-label rule (no labels): distractor = the sole changed number moved by an offset in S; true = all numbers equal
NEG_RULE = "sg_sole_off_in_s = 1"
POS_RULE = "sg_num_changed = 0 AND sg_num_l_only = 0 AND sg_num_r_only = 0 AND sg_num_eq >= 1"


def learn(con, where: str, label: str) -> dict:
    """Signature vocab + token log-odds from tok_fit rows matching ``where`` with label expression ``label``."""
    src = f"(SELECT *, ({label})::INT AS y FROM read_parquet('{TMP}/tok_fit.parquet') WHERE {where}) t"
    con.execute(f"CREATE OR REPLACE TEMP TABLE lt AS SELECT * FROM {src} WHERE y IS NOT NULL")
    n_pos, n_neg = con.execute("SELECT sum(y), sum(1 - y) FROM lt").fetchone()
    prior = math.log(max(n_pos, 1) / max(n_neg, 1))
    sig = {}
    col = {"del": '"del"'}
    for k in SIG_KEYS:
        rows = con.execute(
            f"""SELECT tok, sum(1 - y) neg, sum(y) pos FROM (SELECT unnest({col.get(k, k)}) tok, y FROM lt)
                GROUP BY tok HAVING count(*) >= 10"""
        ).fetchall()
        sig[k] = {w: round(math.log((ng + 1) / (ps + 1)) + prior, 3) for w, ng, ps in rows}
    # token log-odds (textdiff.learn_token_logodds, min_count 20, alpha 1): totals over all name_core tokens
    norm = "regexp_replace(lower(x), '[^a-z0-9]', '', 'g')"
    tot = con.execute(
        f"""WITH p AS (SELECT l.name_core lc, r.name_core rc, lt.y FROM lt
                       JOIN read_parquet('{S1}') l ON l.entity_id = lt.lid
                       JOIN read_parquet({[str(p) for p in R]}) r ON r.entity_id = lt.rid),
                 toks AS (SELECT {norm} t, y FROM (SELECT unnest(string_split(lc, ' ')) x, y FROM p WHERE lc <> ''
                          UNION ALL SELECT unnest(string_split(rc, ' ')) x, y FROM p WHERE rc <> ''))
            SELECT t, sum(y) pos, sum(1 - y) neg FROM toks WHERE t <> '' GROUP BY t"""
    ).fetchall()
    une = dict(
        ((t, (p, n)) for t, p, n in con.execute(
            """SELECT t, sum(y), sum(1 - y) FROM (SELECT unnest(list_concat(tdl, tdr)) t, y FROM lt)
               WHERE t <> '' GROUP BY t"""
        ).fetchall())
    )
    lo = {}
    for t, np_, nn in tot:
        if np_ + nn < 20:
            continue
        up, un = une.get(t, (0, 0))
        lo[t] = math.log(((un + 1) / (nn + 2)) / ((up + 1) / (np_ + 2)))
    return {"sig": sig, "lo": lo, "n_pos": int(n_pos), "n_neg": int(n_neg)}


def apply_batch(tab: pa.Table, v: dict) -> dict[str, np.ndarray]:
    """13 vocab columns of a batch of token rows under vocab ``v`` (same semantics as ber.signature / textdiff)."""
    sig = {k: v["sig"].get(k, {}) for k in SIG_KEYS}
    lo = v["lo"]
    n = len(tab)
    out = {c: np.zeros(n, dtype=np.float32) for c in VOCAB_COLS}
    cols = {c: tab.column(c).to_pylist() for c in (*SIG_KEYS, "tdl", "tdr", "name_none", "addr_none")}
    vi, vd, vr, vt, va, vad = (sig[k] for k in SIG_KEYS)
    nan = float("nan")
    for i in range(n):
        if cols["name_none"][i]:
            for c in SG_LO[:7]:
                out[c][i] = nan
        else:
            ins = [vi.get(w, 0.0) for w in cols["ins"][i]]
            rep = [vr.get(xy, vt.get(xy.split(">", 1)[1], 0.0)) for xy in cols["repl"][i]]
            rtg = [vt.get(y, 0.0) for y in cols["rtgt"][i]]
            dl = [vd.get(w, 0.0) for w in cols["del"][i]]
            out["sg_ins_lo_max"][i] = max(ins, default=0.0)
            out["sg_ins_lo_sum"][i] = sum(ins)
            out["sg_repl_lo_max"][i] = max(rep, default=0.0)
            out["sg_repl_lo_sum"][i] = sum(rep)
            out["sg_rtgt_lo_max"][i] = max(rtg, default=0.0)
            out["sg_del_lo_min"][i] = min(dl, default=0.0)
            out["sg_edit_lo_max"][i] = max(ins + rep, default=0.0)
        if cols["addr_none"][i]:
            out["sg_a_ins_lo_max"][i] = nan
            out["sg_a_del_lo_max"][i] = nan
        else:
            out["sg_a_ins_lo_max"][i] = max((va.get(w, 0.0) for w in cols["ains"][i]), default=0.0)
            out["sg_a_del_lo_max"][i] = max((vad.get(w, 0.0) for w in cols["adel"][i]), default=0.0)
        ll = [lo.get(t, 0.0) for t in cols["tdl"][i]]
        rl = [lo.get(t, 0.0) for t in cols["tdr"][i]]
        out["nd_l_logodds_max"][i] = max(ll) if ll else 0.0
        out["nd_l_logodds_sum"][i] = sum(ll)
        out["nd_r_logodds_max"][i] = max(rl) if rl else 0.0
        out["nd_r_logodds_sum"][i] = sum(rl)
    return out


def apply(group: str, variant: str, v: dict, where=None) -> None:
    """Write vf_{group}_{variant}.parquet for the rows of tok_{group} (optionally filtered by a row predicate)."""
    t0 = time.time()
    pf = pq.ParquetFile(TMP / f"tok_{group}.parquet")
    need = ["lid", "rid", "country", "bkt", *SIG_KEYS, "tdl", "tdr", "name_none", "addr_none"]
    w = None
    for b in pf.iter_batches(batch_size=300_000, columns=need):
        tab = pa.Table.from_batches([b])
        if where is not None:
            tab = tab.filter(where(tab))
        f = apply_batch(tab, v)
        out = pa.table({"lid": tab.column("lid"), "rid": tab.column("rid"), **{c: f[c] for c in VOCAB_COLS}})
        if w is None:
            w = pq.ParquetWriter(TMP / f"vf_{group}_{variant}.parquet", out.schema, compression="zstd")
        w.write_table(out)
    w.close()
    print("applied", group, variant, round(time.time() - t0), "s", flush=True)


def load_pipeline_vocab(name: str) -> dict:
    """The pipeline's pooled India+US vocab (for validating the recomputation)."""
    return {"sig": json.loads((CACHE / f"learned/sigvocab_{name}.json").read_text()),
            "lo": json.loads((CACHE / f"learned/logodds_{name}.json").read_text())}


def vocab(name: str) -> dict:
    """Load a learned vocab variant."""
    return json.loads((TMP / f"vocab_{name}.json").read_text())


if __name__ == "__main__":
    import pyarrow.compute as pc

    step = sys.argv[1]
    if step == "validate":  # recompute with the pipeline vocab on 300k US holdout rows; compare to the feature file
        v = load_pipeline_vocab("fit")
        tab = pq.ParquetFile(TMP / "tok_holdout.parquet").iter_batches(batch_size=300_000).__next__()
        tab = pa.Table.from_batches([tab])
        f = apply_batch(tab, v)
        con = connect()
        con.register("ids", pa.table({"lid": tab.column("lid"), "rid": tab.column("rid"), "i": np.arange(len(tab))}))
        ref = con.execute(
            f"SELECT i, {', '.join(VOCAB_COLS)} FROM read_parquet('{CACHE}/feat_v3_train_holdout.parquet') "
            "JOIN ids USING (lid, rid) ORDER BY i").fetch_arrow_table()
        for c in VOCAB_COLS:
            a, b = f[c], ref.column(c).to_numpy(zero_copy_only=False)
            same = np.isclose(a, b, atol=2e-3, equal_nan=True).mean()
            print(f"{c:20s} match {same:.5f}")
    elif step == "learn":
        con = connect()
        ind, us = "country = 'india'", "country = 'us'"
        self_lab = f"CASE WHEN {NEG_RULE} THEN 0 WHEN {POS_RULE} THEN 1 END"
        specs = {
            "IN_A": (f"{ind} AND bkt < 15", "label"), "IN_B": (f"{ind} AND bkt >= 15", "label"), "IN": (ind, "label"),
            "USS": (us, self_lab),
            "INS_A": (f"{ind} AND bkt < 15", self_lab), "INS_B": (f"{ind} AND bkt >= 15", self_lab),
            "INS": (ind, self_lab),
            "US": (us, "label"),  # reference only: US vocab with TRUE labels (upper bound of C2)
        }
        for name, (where, lab) in specs.items():
            t0 = time.time()
            v = learn(con, where, lab)
            (TMP / f"vocab_{name}.json").write_text(json.dumps(v))
            print(name, "pos", v["n_pos"], "neg", v["n_neg"], "sig tokens", sum(map(len, v["sig"].values())),
                  "lo tokens", len(v["lo"]), round(time.time() - t0), "s", flush=True)
        # self-label quality on US fit (uses labels ONLY to report precision of the rule)
        r = con.execute(
            f"""SELECT CASE WHEN {NEG_RULE} THEN 'neg' WHEN {POS_RULE} THEN 'pos' ELSE 'none' END s, country,
                       count(*), avg(label) FROM read_parquet('{TMP}/tok_fit.parquet') GROUP BY 1, 2 ORDER BY 2, 1"""
        ).fetchall()
        print("selflabel quality", r, flush=True)
        a, b = vocab("IN"), vocab("USS")
        mrg = {"sig": {k: {**a["sig"].get(k, {}), **b["sig"].get(k, {})} for k in SIG_KEYS},
               "lo": {**a["lo"], **b["lo"]}}
        (TMP / "vocab_MRG.json").write_text(json.dumps(mrg))
    elif step == "learn2":  # stricter self-positive: numbers equal AND core-name ratio >= 0.6 (87-91% true)
        con = connect()
        lab2 = f"CASE WHEN {NEG_RULE} THEN 0 WHEN {POS_RULE} AND core_ratio >= 0.6 THEN 1 END"
        for name, where in {"USS2": "country = 'us'", "INS2_B": "country = 'india' AND bkt >= 15",
                            "INS2": "country = 'india'"}.items():
            v = learn(con, where, lab2)
            (TMP / f"vocab_{name}.json").write_text(json.dumps(v))
            print(name, "pos", v["n_pos"], "neg", v["n_neg"], "sig tokens", sum(map(len, v["sig"].values())), flush=True)
    elif step == "apply":
        ind = lambda t: pc.equal(t.column("country"), "india")  # noqa: E731
        indA = lambda t: pc.and_(ind(t), pc.less(t.column("bkt"), 15))  # noqa: E731
        us = lambda t: pc.equal(t.column("country"), "us")  # noqa: E731
        usA = lambda t: pc.and_(us(t), pc.less(t.column("bkt"), 15))  # noqa: E731
        jobs = sys.argv[2].split(",")
        table = {
            # B: India training rows get the OTHER India fold's vocab; India tune + US rows get full India vocab
            "fit_INB": ("fit", "IN_B", indA), "tune_IN": ("tune", "IN", None), "holdout_IN": ("holdout", "IN", None),
            "fit_USIN": ("fit", "IN", usA),
            "holdout_USS": ("holdout", "USS", None), "holdout_MRG": ("holdout", "MRG", None),
            "fit_USUSS": ("fit", "USS", usA),
            "fit_INSB": ("fit", "INS_B", indA), "tune_INS": ("tune", "INS", None), "tune_USS": ("tune", "USS", None),
            "holdout_US": ("holdout", "US", None),
            "fit_INS2B": ("fit", "INS2_B", indA), "tune_INS2": ("tune", "INS2", None),
            "holdout_USS2": ("holdout", "USS2", None), "tune_USS2": ("tune", "USS2", None),
            "fit_USUSS2": ("fit", "USS2", usA),
        }
        for j in jobs:
            g, vn, wh = table[j]
            apply(g, vn if j != "fit_USIN" and j != "fit_USUSS" else vn, vocab(vn), wh)
            if j in ("fit_USIN", "fit_USUSS", "fit_USUSS2"):
                (TMP / f"vf_{g}_{vn}.parquet").rename(TMP / f"vf_fit_{j[4:]}.parquet")
