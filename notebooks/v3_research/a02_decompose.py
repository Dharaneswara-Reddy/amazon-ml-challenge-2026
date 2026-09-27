"""Decompose holdout macro-F0.5 loss into blocking misses, one-to-one losses, decision FNs and FPs; oracle fixes.

Reads out/hold_pairs.parquet + out/hold_ent.parquet (a01). Writes out/hold_ent_counts.parquet.
"""
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
OUT = ROOT / "notebooks/v3_research/out"
C = ROOT / "artifacts/cache"
con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=4")
con.execute(f"CREATE VIEW hp AS SELECT * FROM read_parquet('{OUT}/hold_pairs.parquet')")
con.execute(f"CREATE VIEW he AS SELECT * FROM read_parquet('{OUT}/hold_ent.parquet')")
con.execute(f"""CREATE TEMP TABLE gt AS SELECT g.s1_id, g.match_id FROM read_parquet('{C}/train_gt_pairs.parquet') g
                WHERE g.s1_id IN (SELECT s1_id FROM he)""")
con.execute(f"CREATE TEMP TABLE gtall AS SELECT * FROM read_parquet('{C}/train_gt_pairs.parquet')")
# per true pair: status
con.execute("""CREATE TEMP TABLE tp_status AS
  SELECT gt.s1_id, gt.match_id,
    CASE WHEN hp.rid IS NULL THEN 'block_miss'
         WHEN NOT hp.kept THEN 'o2o_rival_out'
         WHEN NOT hp.is_best THEN 'o2o_rival_in'
         WHEN hp.sel THEN 'tp'
         ELSE 'dec_fn' END AS status, hp.p, hp.pcal
  FROM gt LEFT JOIN hp ON hp.lid = gt.s1_id AND hp.rid = gt.match_id""")
print(con.execute("SELECT status, count(*), round(avg(p),3) FROM tp_status GROUP BY 1 ORDER BY 2 DESC").fetchall())
# FPs: selected non-label; is the rid a true match of another S1?
con.execute("""CREATE TEMP TABLE fp AS
  SELECT hp.idx, hp.lid, hp.rid, hp.p, hp.pcal, g.s1_id AS true_owner
  FROM hp LEFT JOIN gtall g ON g.match_id = hp.rid WHERE hp.sel AND NOT hp.label""")
print(con.execute("""SELECT (true_owner IS NULL) AS distractor, count(*) FROM fp GROUP BY 1""").fetchall())

e = con.execute("""
  SELECT he.idx, he.n_true, he.country, he.k_star,
    coalesce(sum((t.status='tp')::INT),0) tp, coalesce(sum((t.status='block_miss')::INT),0) fn_block,
    coalesce(sum((t.status LIKE 'o2o%')::INT),0) fn_o2o, coalesce(sum((t.status='dec_fn')::INT),0) fn_dec
  FROM he LEFT JOIN tp_status t ON t.s1_id = he.s1_id GROUP BY ALL ORDER BY he.idx""").fetchnumpy()
fpc = con.execute("""SELECT he.idx, count(fp.rid) fp, coalesce(sum((fp.true_owner IS NOT NULL)::INT),0) fp_conf
  FROM he LEFT JOIN fp ON fp.idx = he.idx GROUP BY 1 ORDER BY 1""").fetchnumpy()
n_true = e["n_true"]
tp, fb, fo, fd = (e[k].astype(float) for k in ("tp", "fn_block", "fn_o2o", "fn_dec"))
fp = fpc["fp"].astype(float)
fp_conf = fpc["fp_conf"].astype(float)
assert np.allclose(tp + fb + fo + fd, n_true)


def F(tp, fp, fn, n_true):
    denom = 1.25 * tp + 0.25 * fn + fp
    f = np.where(denom > 0, 1.25 * tp / np.maximum(denom, 1e-12), 0.0)
    return np.where(n_true == 0, (fp == 0).astype(float), f)


base = F(tp, fp, fb + fo + fd, n_true)
print("base", base.mean())
single = n_true == 0
fp_single = np.where(single, fp, 0)
fp_multi = np.where(single, 0, fp)
res = {}
res["fix FP on singletons"] = F(tp, fp - fp_single, fb + fo + fd, n_true).mean()
res["fix FP on matched entities"] = F(tp, fp - fp_multi, fb + fo + fd, n_true).mean()
res["  of which: FP rid is another S1's true match"] = F(tp, fp - np.where(single, 0, fp_conf), fb + fo + fd, n_true).mean()
res["fix all FP"] = F(tp, 0 * fp, fb + fo + fd, n_true).mean()
res["fix decision FN"] = F(tp + fd, fp, fb + fo, n_true).mean()
res["fix one-to-one FN"] = F(tp + fo, fp, fb + fd, n_true).mean()
res["fix blocking FN"] = F(tp + fb, fp, fo + fd, n_true).mean()
res["fix all FN"] = F(tp + fb + fo + fd, fp, 0 * tp, n_true).mean()
res["perfect decision on current candidates (all FP, dec FN, o2o FN)"] = F(tp + fd + fo, 0 * fp, fb, n_true).mean()
for k, v in res.items():
    print(f"{k:70s} {v:.5f}  gain {v - base.mean():+.5f}")
# waterfall
steps = [("FP singletons", dict(fp=fp - fp_single)), ("FP matched", dict(fp=0 * fp)),
         ("dec FN", dict(tp=tp + fd, fn=fb + fo)), ("o2o FN", dict(tp=tp + fd + fo, fn=fb)),
         ("block FN", dict(tp=tp + fd + fo + fb, fn=0 * fb))]
cur = dict(tp=tp, fp=fp, fn=fb + fo + fd)
prev = base.mean()
for name, ch in steps:
    cur.update(ch)
    v = F(cur["tp"], cur["fp"], cur["fn"], n_true).mean()
    print(f"waterfall {name:15s} -> {v:.5f} (+{v - prev:.5f})")
    prev = v
print("n entities", len(n_true), "singletons", single.sum(), "singletons with FP", (fp_single > 0).sum(),
      "matched ents", (~single).sum(), "matched with any FP", (fp_multi > 0).sum())
print("totals: tp", tp.sum(), "fp", fp.sum(), "fp_single", fp_single.sum(), "fp_conf(matched)",
      np.where(single, 0, fp_conf).sum(), "fn_block", fb.sum(), "fn_o2o", fo.sum(), "fn_dec", fd.sum())
# loss by entity class
loss = 1 - base
print("total loss points", loss.sum() / len(loss))
for name, m in [("singleton", single), ("n=1", n_true == 1), ("n=2", n_true == 2), ("n=3-4", (n_true >= 3) & (n_true <= 4)),
                ("n>=5", n_true >= 5)]:
    print(f"{name:10s} share_ent {m.mean():.3f} meanF {base[m].mean():.4f} loss_pts {loss[m].sum() / len(loss):.5f}")
import pyarrow as pa, pyarrow.parquet as pq  # noqa: E401,E402
pq.write_table(pa.table({"idx": e["idx"], "n_true": n_true, "country": e["country"], "k_star": e["k_star"], "tp": tp,
                         "fp": fp, "fp_conf": fp_conf, "fn_block": fb, "fn_o2o": fo, "fn_dec": fd, "f": base}),
               OUT / "hold_ent_counts.parquet")
con.execute("COPY tp_status TO '%s' (FORMAT parquet)" % (OUT / "tp_status.parquet"))
con.execute("COPY fp TO '%s' (FORMAT parquet)" % (OUT / "fp.parquet"))
