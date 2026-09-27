"""Re-score France test pairs with the French vocabularies (faithful v3 path: features -> model1 -> stage-2
competition features -> model2). Variants: 'base' (France subset of feat_v3_test, unchanged; reproduces pred2_test)
and 'fr' (sg_* signature features and nd_*_logodds_* recomputed with merged train+French vocabularies for France
pairs with p2 >= 0.003; lower pairs keep their values). Outputs artifacts/v4_tmp/pred2_fr_{base,fr}.parquet.
"""
import json
import multiprocessing as mp
import sys
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from h_common import C, RUN, TMP, connect, norm_right

sys.path.insert(0, "/home/toji339/Documents/Amazon_ML_Hackathon/code/business_entity_resolution/src")
from ber import signature, textdiff  # noqa: E402
from ber.model import predict_file  # noqa: E402
from ber.stage2 import MODEL2_FEATURES, write_stage2_features  # noqa: E402

SIG_SCALE = 1 / 1.18  # pseudo-vocab values run ~1.18x the production vocab (h04 train comparison)
LO_COLS = ["nd_l_logodds_max", "nd_l_logodds_sum", "nd_r_logodds_max", "nd_r_logodds_sum"]
VOC = LO = None


def merged():
    """Train vocab overridden by French pseudo vocab (scaled), and the same for token log-odds."""
    tr = json.loads((C / "learned/sigvocab_fit.json").read_text())
    fr = json.loads((TMP / "sigvocab_fr.json").read_text())
    voc = {k: {**tr.get(k, {}), **{w: round(v * SIG_SCALE, 3) for w, v in fr.get(k, {}).items()}} for k in tr}
    lo = {**json.loads((C / "learned/logodds_fit.json").read_text()), **json.loads((TMP / "logodds_fr.json").read_text())}
    return voc, lo


def _init(voc, lo):
    global VOC, LO
    VOC, LO = voc, lo


def _feats(chunk):
    """Recomputed SIG_FEATURES + logodds nd columns for a chunk of (ln, la, rn, ra, lc, rc)."""
    ln, la, rn, ra, lc, rc = (list(x) for x in zip(*chunk))
    f = signature.signature_features(ln, la, rn, ra, VOC)
    nd = textdiff.name_diff_features(lc, rc, None, LO)
    return {**f, **{c: nd[c] for c in LO_COLS}}


def recompute(con):
    voc, lo = merged()
    json.dump(voc, open(TMP / "sigvocab_merged.json", "w")); json.dump(lo, open(TMP / "logodds_merged.json", "w"))
    t = con.execute(f"""SELECT x.lid, x.rid, l.name_norm, l.addr_norm, r.name_norm, r.addr_norm, l.name_core, r.name_core
        FROM read_parquet('{TMP}/pairs_fr.parquet') x JOIN read_parquet('{C}/norm_test_s1.parquet') l ON l.entity_id = x.lid
        JOIN {norm_right()} r ON r.entity_id = x.rid WHERE x.p >= 0.003""").fetchall()
    print("pairs to recompute", len(t), flush=True)
    ids = [(a, b) for a, b, *_ in t]
    rows = [tuple("" if v is None else v for v in r[2:]) for r in t]
    del t
    chunks = [rows[i:i + 20000] for i in range(0, len(rows), 20000)]
    with mp.get_context("fork").Pool(4, initializer=_init, initargs=(voc, lo)) as pool:
        parts = pool.map(_feats, chunks)
    cols = list(parts[0])
    tab = pa.table({"lid": [a for a, _ in ids], "rid": [b for _, b in ids],
                    **{c: np.concatenate([p[c] for p in parts]).astype(np.float32) for c in cols}})
    pq.write_table(tab, TMP / "fr_newfeat.parquet")
    return cols


def build(con, cols):
    fcols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{C}/feat_v3_test.parquet')").fetchall()]
    fr = f"(SELECT entity_id FROM read_parquet('{C}/raw_test_s1.parquet') WHERE country='France')"
    con.execute(f"""COPY (SELECT f.* FROM read_parquet('{C}/feat_v3_test.parquet') f SEMI JOIN {fr} s ON s.entity_id = f.lid)
                   TO '{TMP}/feat_fr_base.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    sel = ", ".join(f"coalesce(n.{c}, f.{c}) AS {c}" if c in cols else f"f.{c}" for c in fcols)
    con.execute(f"""COPY (SELECT {sel} FROM read_parquet('{TMP}/feat_fr_base.parquet') f
                   LEFT JOIN read_parquet('{TMP}/fr_newfeat.parquet') n USING (lid, rid))
                   TO '{TMP}/feat_fr_fr.parquet' (FORMAT parquet, COMPRESSION zstd)""")


def score(con, variant):
    b1 = lgb.Booster(model_file=str(RUN / "model1.txt"))
    b2 = lgb.Booster(model_file=str(RUN / "model2.txt"))
    feat, p1, s2, p2 = (TMP / f"{x}_fr_{variant}.parquet" for x in ("feat", "pred1", "s2", "pred2"))
    predict_file(b1, feat, p1, 500_000)
    write_stage2_features(con, feat, p1, s2)
    predict_file(b2, s2, p2, 500_000, features=MODEL2_FEATURES)


if __name__ == "__main__":
    con = connect()
    t0 = time.time()
    steps = sys.argv[1:] or ["recompute", "build", "base", "fr"]
    cols = signature.SIG_FEATURES + LO_COLS
    if "recompute" in steps:
        cols = recompute(con); print("recomputed", time.time() - t0, flush=True)
    if "build" in steps:
        build(con, cols); print("built", time.time() - t0, flush=True)
    for v in ("base", "fr"):
        if v in steps:
            score(con, v); print("scored", v, time.time() - t0, flush=True)
