"""P04: the learned cut of B §Q3(g). A small LightGBM on cheap union features, trained on union rows of S1 in fit
buckets 5-29 only (never FIT-SUB/TUNE/HOLDOUT, so its p is out-of-sample everywhere it is used), then
keep = legacy-linear top-K per right OR p >= thr. Reports candidate recall / pairs for several cut rules and writes
the chosen candidate set (string ids, cand_train-compatible columns + new-pass columns) for ber.features.
Usage: python p04_cut.py [K=1] [thr=1e-3]"""
import numpy as np, lightgbm as lgb, pyarrow as pa, pyarrow.parquet as pq
from common import *

K = int(sys.argv[1]) if len(sys.argv) > 1 else 1
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
U = T / "union"
CF = ["in_v2", "n_pass", "tok_score", "q3_score", *(f"s_{p}" for p in PASSES), "n_pass_new", "jw_core", "jw_addr",
      "jacc_addr", "num_match", "jw_strip", "jw_phn", "jacc_name", "n_num", "r_addr_len", "r_nnum", "block_score"]
con = connect()
t = time.time()
sel = ", ".join(f"coalesce({c}::FLOAT, -1) AS {c}" for c in CF)
model_path = T / "cut_model.txt"
if not model_path.exists():
    d = con.execute(f"""SELECT y::INT AS y, {sel} FROM '{U}/b*.parquet'
                        WHERE bkt >= 5 AND bkt < 30 AND hash(rid) % 4 = 0 ORDER BY lid, rid""").fetchnumpy()
    X = np.column_stack([d[c] for c in CF]); y = d["y"]
    log("cut train rows", len(y), "pos", int(y.sum()), time.time() - t)
    prm = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=50, bagging_fraction=0.5,
               bagging_freq=1, feature_fraction=0.8, verbose=-1, num_threads=8, seed=0, deterministic=True, force_row_wise=True)
    bst = lgb.train(prm, lgb.Dataset(X, y, feature_name=CF), num_boost_round=300)
    bst.save_model(str(model_path)); del X, d
    log("cut model trained", time.time() - t)
bst = lgb.Booster(model_file=str(model_path))
for b in range(8):
    pf = pq.ParquetFile(U / f"b{b}.parquet")
    if (U / f"p{b}.parquet").exists():
        continue
    w = None
    for bt in pf.iter_batches(batch_size=2_000_000, columns=["lid", "rid", *CF]):
        X = np.column_stack([np.nan_to_num(bt.column(c).to_numpy(zero_copy_only=False).astype(np.float32), nan=-1) for c in CF])
        p = bst.predict(X, num_threads=8).astype(np.float32)
        ob = pa.RecordBatch.from_arrays([bt.column("lid"), bt.column("rid"), pa.array(p)], names=["lid", "rid", "cut_p"])
        w = w or pq.ParquetWriter(U / f"p{b}.parquet", ob.schema)
        w.write_batch(ob)
    w.close()
    log("scored bucket", b, time.time() - t)
# rank by legacy linear score (jw_core + jw_addr + jacc_addr + 0.3 * shared number) per right
con.execute(f"""CREATE OR REPLACE TABLE ucut AS SELECT u.lid, u.rid, u.y, u.in_v2, u.bkt, p.cut_p,
      row_number() OVER (PARTITION BY u.rid ORDER BY coalesce(u.jw_core,0) + coalesce(u.jw_addr,0) + coalesce(u.jacc_addr,0)
                         + 0.3 * (u.n_num > 0)::INT DESC, u.lid) AS lrank
    FROM '{U}/b*.parquet' u JOIN '{U}/p*.parquet' p USING (lid, rid)""")
log("ucut", time.time() - t)
rules = {"v2 (cand_train K=3)": "in_v2", "union (v2 + 5 passes, uncut)": "TRUE", "linear top-3": "lrank <= 3",
         "top-1 OR p>=1e-3": "lrank <= 1 OR cut_p >= 1e-3", "top-2 OR p>=1e-3": "lrank <= 2 OR cut_p >= 1e-3",
         "top-1 OR p>=1e-4": "lrank <= 1 OR cut_p >= 1e-4", "top-3 OR p>=1e-3": "lrank <= 3 OR cut_p >= 1e-3"}
grp = "CASE WHEN bkt >= 35 AND bkt < 40 THEN 'holdout' WHEN bkt >= 30 AND bkt < 35 THEN 'tune' WHEN bkt < 5 THEN 'fitsub' END"
ntrue = dict(con.execute(f"""SELECT {grp.replace('bkt', 'l.bkt')} g, count(*) FROM gt JOIN L l ON l.iid = gt.lid GROUP BY 1""").fetchall())
nent = dict(con.execute(f"SELECT {grp} g, count(*) FROM L GROUP BY 1").fetchall())
nright = con.execute("SELECT count(DISTINCT rid) FROM ucut").fetchone()[0]
rows = ["| cut rule | HOLDOUT recall | TUNE recall | FIT-SUB recall | holdout pairs / S1 | pairs / right (relevant rights) |", "|---|---|---|---|---|---|"]
for name, cond in rules.items():
    r = dict((g, (h, n)) for g, h, n in con.execute(f"SELECT {grp} g, sum(y::INT), count(*) FROM ucut WHERE {cond} GROUP BY 1").fetchall())
    tot = con.execute(f"SELECT count(*) FROM ucut WHERE {cond}").fetchone()[0]
    rows.append(f"| {name} | {r['holdout'][0] / ntrue['holdout']:.4%} | {r['tune'][0] / ntrue['tune']:.4%} | {r['fitsub'][0] / ntrue['fitsub']:.4%} "
                f"| {r['holdout'][1] / nent['holdout']:.2f} | {tot / nright:.2f} |")
    log(rows[-1])
note(f"""
## Step 1-2: candidate generation (measured on all train rights, all 2.2M train S1 on the left)
New passes px/hs/n2/na/phx exactly as B §Q5 (caps 30/50/30/30/30, top-5, phx all <= 30), run over all 10.32M train
rights (p02). Union restricted to the {nright:,} right records that have >= 1 union candidate in FIT-SUB/TUNE/HOLDOUT
(all their candidates kept). Cut model: LightGBM (300 trees) on {len(CF)} cheap features, trained on union rows of S1
buckets 5-29 only (out-of-sample for all three subsets). True pairs: HOLDOUT {ntrue['holdout']:,}, TUNE {ntrue['tune']:,},
FIT-SUB {ntrue['fitsub']:,}; entities HOLDOUT {nent['holdout']:,}.

""" + "\n".join(rows))
# write chosen candidate set with string ids for ber.features
cond = f"lrank <= {K} OR cut_p >= {THR}"
V2F = ["in_tok", "in_q3", "in_sn_name", "in_sn_name_addr", "in_sn_addr_name", "in_sn_skel_nums", "in_sn_nums_addr", "in_sn_addr_skel", "in_nameonly"]
dst = T / f"cand_v3_k{K}_t{THR:g}.parquet"
con.execute(f"""COPY (SELECT l.entity_id AS lid, r.entity_id AS rid, {', '.join('u.' + f for f in V2F)},
      u.tok_score, u.q3_score, u.jw_core, u.jw_addr, u.jacc_addr, u.num_match, u.n_pass, u.block_score,
      {', '.join(f'(u.s_{p} IS NOT NULL)::TINYINT AS in_{p}, coalesce(u.s_{p}, 0)::FLOAT AS s_{p}' for p in PASSES)},
      u.n_pass_new, u.in_v2::TINYINT AS in_v2, u.jw_strip, u.jw_phn, c.cut_p, c.lrank
    FROM ucut c JOIN '{U}/b*.parquet' u USING (lid, rid) JOIN L l ON l.iid = u.lid JOIN R r ON r.iid = u.rid
    WHERE {cond.replace('lrank', 'c.lrank').replace('cut_p', 'c.cut_p')}) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)""")
log("wrote", dst, con.execute(f"SELECT count(*) FROM '{dst}'").fetchone(), time.time() - t)
