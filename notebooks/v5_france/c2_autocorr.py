"""C2: correct the C1 chance baseline for the peaked house-number distribution: for independent numbers the chance
count at offset d is proportional to A(d) = sum_h pL(h) pR(h + d); ratio = A(0) / mean_{d in BASE} A(d). Estimated
chance at d = 0 = C1 baseline x ratio. Validated on train (labels), applied to France / US test."""
import json
import numpy as np
from common import *
c = connect()
BASE = [d for d in range(-30, -2)] + [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
def hist(split, src, where):
    """Normalized histogram of integer house numbers 0..5000."""
    files = f"'{CACHE}/norm_{split}_s1.parquet'" if src == 1 else f"['{CACHE}/norm_{split}_s2.parquet', '{CACHE}/norm_{split}_s3.parquet']"
    r = c.execute(f"""SELECT h, count(*) FROM (SELECT try_cast(addr_house AS INT) h FROM read_parquet({files}) WHERE {where})
                      WHERE h BETWEEN 0 AND 5000 GROUP BY 1""").fetchall()
    v = np.zeros(5040); 
    for h, n in r: v[h] = n
    return v / v.sum()
c1 = json.load(open(OUT / "c1_collision.json"))
res = {}
for name, split, lw, rw in (("train10", "train", "hash(entity_id) % 10 = 0", "true"),
                            ("France", "test", "country_norm = 'france'", "country_norm = 'france'"),
                            ("US_test", "test", "country_norm = 'us'", "country_norm = 'us'")):
    pl, pr = hist(split, 1, lw), hist(split, 2, rw)
    A = {d: float(np.dot(pl[:5000], np.roll(pr, -d)[:5000])) for d in range(-30, 31)}
    ratio = A[0] / np.mean([A[d] for d in BASE])
    print(f"\n== {name}: A(0)/mean A(base) = {ratio:.3f}; A(1)/A(0)={A[1]/A[0]:.3f} A(-3)/A(0)={A[-3]/A[0]:.3f} A(-10)/A(0)={A[-10]/A[0]:.3f}")
    print("street city_eq    n(d=0)   true(d=0)  chance_est  est_true  est_true_share  (train: real true share)")
    for r in c1[name]:
        ch = r["base"] * ratio; est = r["n0"] - ch
        real = f"{r['t0'] / r['n0']:.3f}" if r["t0"] is not None else "-"
        print(f"{r['sb']:6s} {r['ce']:7d} {r['n0']:9,d} {r['t0'] if r['t0'] is not None else '-':>10} {ch:11.0f} {est:9.0f} {est / r['n0']:14.3f}   {real}")
    res[name] = {"ratio": ratio}
json.dump(res, open(OUT / "c2_autocorr.json", "w"), indent=1)
