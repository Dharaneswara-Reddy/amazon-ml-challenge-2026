"""B11: recall@K per right of (a) legacy linear score, (b) LightGBM on cheap features, over the expanded union."""
import duckdb, numpy as np, lightgbm as lgb, json
S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb", read_only=True); con.execute("SET memory_limit='2GB'; SET threads=4")
cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{S}/union_feat.parquet'").fetchall()]
feat = [c for c in cols if c not in ("lid", "rid", "y")]
d = con.execute(f"SELECT dense_rank() OVER (ORDER BY rid) rc, (hash(rid || 'x') % 2)::INT AS half, y, {', '.join(f'coalesce({c}, -1)::FLOAT AS {c}' for c in feat)} FROM '{S}/union_feat.parquet'").fetchnumpy()
ngt_half = dict(con.execute("SELECT (hash(rid || 'x') % 2)::INT, count(*) FROM gt GROUP BY 1").fetchall())
X = np.column_stack([d[c] for c in feat]); y = d["y"]; rc = d["rc"]; half = d["half"]
def recall_at_k(mask, score, kmax=10):
    r, s, l = rc[mask], score[mask], y[mask]
    o = np.lexsort((-s, r)); rs = r[o]
    st = np.r_[0, np.flatnonzero(rs[1:] != rs[:-1]) + 1]
    rank = np.arange(len(o)) - np.repeat(st, np.diff(np.r_[st, len(o)]))
    lab = l[o]; n = ngt_half[int(half[mask][0])]; nr = len(st)
    return [(k, round(lab[rank < k].sum() / n, 4), round(min(k, 99) * 0 + (rank < k).sum() / nr, 2)) for k in range(1, kmax + 1)]
te = half == 1; tr = ~te
print("features", feat)
print("union ceiling (test half):", y[te].sum() / ngt_half[1])
leg = np.nan_to_num(d["jw_core"]) + np.nan_to_num(d["jw_addr"]) + np.nan_to_num(d["jacc_addr"]) + 0.3 * (d["n_num"] > 0)
print("legacy linear:", recall_at_k(te, leg))
prm = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=50, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, verbose=-1, num_threads=4, seed=0)
bst = lgb.train(prm, lgb.Dataset(X[tr], y[tr], feature_name=feat), num_boost_round=300)
p = bst.predict(X[te]); sc = np.zeros(len(y)); sc[te] = p
print("lgbm:", recall_at_k(te, sc))
imp = sorted(zip(feat, bst.feature_importance("gain")), key=lambda x: -x[1]); print([(f, round(g / 1e3)) for f, g in imp])
# global threshold alternative: keep pairs with p >= t (plus top-1 per right) -> pairs per right vs recall
for t in [0.001, 0.003, 0.01, 0.03]:
    keep = (p >= t)
    print(f"thr {t}: recall {y[te][keep].sum() / ngt_half[1]:.4f} pairs/right {keep.sum() / len(np.unique(rc[te])):.2f}")
# hybrid: keep legacy top-k per right OR lgbm p >= t
r_te, leg_te, y_te = rc[te], leg[te], y[te]
o = np.lexsort((-leg_te, r_te)); rs = r_te[o]
st = np.r_[0, np.flatnonzero(rs[1:] != rs[:-1]) + 1]
rank = np.empty(len(o), int); rank[o] = np.arange(len(o)) - np.repeat(st, np.diff(np.r_[st, len(o)]))
nr = len(st)
for t in [1e-4, 3e-4, 1e-3]:
    keep = p >= t
    print(f"thr {t}: recall {y_te[keep].sum() / ngt_half[1]:.4f} pairs/right {keep.sum() / nr:.2f}")
for k in [1, 2, 3]:
    for t in [1e-4, 3e-4, 1e-3, 3e-3]:
        keep = (rank < k) | (p >= t)
        print(f"legacy top-{k} OR p>={t}: recall {y_te[keep].sum() / ngt_half[1]:.4f} pairs/right {keep.sum() / nr:.2f}")
