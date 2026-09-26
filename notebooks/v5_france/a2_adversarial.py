"""A2: adversarial validation. LightGBM classifiers separating (a) France test vs train pairs, (b) France test vs
India+US test pairs, (c) India+US test vs train (reference); AUC, top features by gain, univariate shift (|AUC-0.5|
of each feature alone, medians). Also (a') after dropping the top shifted features, how separable is France still."""
import json
import numpy as np, pyarrow.parquet as pq, lightgbm as lgb
from common import auc as roc_auc_score
from common import *
from ber.features import FEATURES
def load(p, where=None):
    """Parquet -> {col: np.ndarray}, optionally filtered by a country predicate."""
    t = pq.read_table(p)
    d = {k: t.column(k).to_numpy(zero_copy_only=False) for k in t.column_names}
    if where is not None:
        m = where(d["country"]); d = {k: v[m] for k, v in d.items()}
    return d
class F(dict):
    """dict of arrays with df[list] -> matrix and len()."""
    def __getitem__(self, k):
        if isinstance(k, list):
            return np.column_stack([dict.__getitem__(self, c).astype(np.float32) for c in k])
        return dict.__getitem__(self, k)
    def __len__(self):
        return len(dict.__getitem__(self, "lid"))
fr = F(load(TMP / "adv_test.parquet", lambda c: c == "France"))
iu = F(load(TMP / "adv_test.parquet", lambda c: c != "France"))
tr = F(load(TMP / "adv_train.parquet"))

def adv(a, b, feats, name, rounds=150):
    """Train a/b separator on a 70/30 split; return AUC and gain-sorted features."""
    x = np.vstack([a[feats], b[feats]])
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    rng = np.random.default_rng(0); m = rng.random(len(y)) < 0.7
    bst = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
                     "feature_fraction": 0.8, "verbose": -1, "num_threads": 5, "seed": 1},
                    lgb.Dataset(x[m], y[m], feature_name=feats), rounds)
    auc = roc_auc_score(y[~m], bst.predict(x[~m]))
    g = bst.feature_importance("gain"); g = g / g.sum()
    top = sorted(zip(feats, g), key=lambda t: -t[1])
    return auc, top

def uni(a, b, feats):
    """Univariate AUC of each feature for a vs b, with medians / p90."""
    out = []
    for f in feats:
        va, vb = a[f].astype(np.float64), b[f].astype(np.float64)
        va, vb = np.nan_to_num(va, nan=-9e9), np.nan_to_num(vb, nan=-9e9)
        auc = roc_auc_score(np.r_[np.ones(len(va)), np.zeros(len(vb))], np.r_[va, vb])
        out.append((f, abs(auc - 0.5), auc, np.median(va), np.median(vb), np.quantile(va, .9), np.quantile(vb, .9)))
    return sorted(out, key=lambda t: -t[1])

res = {}
for name, a, b in (("FR_vs_train", fr, tr), ("FR_vs_INUS_test", fr, iu), ("INUS_test_vs_train", iu, tr)):
    auc, top = adv(a, b, FEATURES, name)
    u = uni(a, b, FEATURES)
    res[name] = {"auc": auc, "top_gain": top[:25], "uni": u[:30]}
    print(f"\n### {name}: AUC {auc:.4f}")
    print("top gain:", ", ".join(f"{f} {g:.3f}" for f, g in top[:20]))
    print("univariate |AUC-.5| (feat, auc, med_a, med_b, p90_a, p90_b):")
    for r in u[:30]:
        print(f"  {r[0]:22s} auc={r[2]:.3f} med {r[3]:.3g} vs {r[4]:.3g}  p90 {r[5]:.3g} vs {r[6]:.3g}")
# how concentrated: drop top-k gain features of FR_vs_train and refit
drop = []
for k in (5, 10, 20):
    drop = [f for f, _ in res["FR_vs_train"]["top_gain"][:k]]
    feats = [f for f in FEATURES if f not in drop]
    auc, top = adv(fr, tr, feats, f"drop{k}", rounds=100)
    print(f"\nFR_vs_train after dropping top-{k} gain features: AUC {auc:.4f}; next: " + ", ".join(f"{f} {g:.3f}" for f, g in top[:8]))
    res[f"drop{k}"] = {"auc": auc, "dropped": drop, "top_gain": top[:10]}
json.dump(res, open(OUT / "a2_adversarial.json", "w"), default=float, indent=1)
