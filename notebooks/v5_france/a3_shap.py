"""A3: SHAP (LightGBM pred_contrib) of the v4 stage-1 and stage-2 models on the clean slice
(numbers identical, core_ratio>=.95, r_rank=1): France rejected (p2<0.1) / France accepted (p2>=0.9) vs train
(tune+holdout) true / false, and train high-frequency (r_name_freq>=200) true / false. Mean contribution per feature
per group (logit units), sorted by FR_rej - TR_true."""
import json
import numpy as np, lightgbm as lgb
from common import *
from ber.features import FEATURES
from ber.stage2 import MODEL2_FEATURES
c = connect()
N = 12000
cols = ", ".join(f"s.{x}" for x in MODEL2_FEATURES)
# stage-2 files carry FEATURES + stage-2 columns; select the slice by joining pred2 (raw p2)
q_test = f"""SELECT 'FR_rej' g, s.lid, s.rid, p.p, {cols} FROM '{RUN}/s2_test.parquet' s
      JOIN '{CACHE}/raw_test_s1.parquet' r ON r.entity_id = s.lid AND r.country = 'France'
      JOIN '{RUN}/pred2_test.parquet' p USING (lid, rid)
      WHERE {CLEAN.replace('f.', 's.')}"""
c.execute(f"CREATE TABLE fr AS {q_test}")
c.execute(f"""CREATE TABLE a AS
  (SELECT * FROM fr WHERE p < 0.1 USING SAMPLE {N} ROWS (reservoir, 1))
  UNION ALL (SELECT * REPLACE ('FR_acc' AS g) FROM fr WHERE p >= 0.9 USING SAMPLE {N} ROWS (reservoir, 1))""")
c.execute("DROP TABLE fr")
for grp in ("tune", "holdout"):
    c.execute(f"""CREATE OR REPLACE TABLE t_{grp} AS SELECT s.label, s.lid, s.rid, p.p, {cols} FROM '{RUN}/s2_{grp}.parquet' s
        JOIN '{RUN}/pred2_{grp}.parquet' p USING (lid, rid) WHERE s.in_group = 1 AND {CLEAN.replace('f.', 's.')}""")
c.execute("CREATE TABLE tr AS SELECT * FROM t_tune UNION ALL SELECT * FROM t_holdout")
print("train clean slice n / true rate / p<.1 share:", c.execute("SELECT count(*), avg(label), avg((p<0.1)::INT) FROM tr").fetchall())
print("train clean slice r_name_freq>=200:", c.execute("SELECT count(*), avg(label), avg((p<0.1)::INT), avg(p) FROM tr WHERE r_name_freq>=200").fetchall())
for g, w in (("TR_true", "label = 1"), ("TR_false", "label = 0"), ("TR_hf_true", "label = 1 AND r_name_freq >= 200"),
             ("TR_hf_false", "label = 0 AND r_name_freq >= 200")):
    c.execute(f"INSERT INTO a SELECT '{g}', lid, rid, p, * EXCLUDE (label, lid, rid, p) FROM tr WHERE {w} USING SAMPLE {N} ROWS (reservoir, 1)")
d = c.execute("SELECT * FROM a").fetchnumpy()
groups = d["g"]
b1 = lgb.Booster(model_file=str(RUN / "model1.txt")); b2 = lgb.Booster(model_file=str(RUN / "model2.txt"))
x1 = np.column_stack([d[f].astype(np.float32) for f in FEATURES])
x2 = np.column_stack([d[f].astype(np.float32) for f in MODEL2_FEATURES])
c1 = b1.predict(x1, pred_contrib=True); c2 = b2.predict(x2, pred_contrib=True)
G = ["FR_rej", "FR_acc", "TR_true", "TR_false", "TR_hf_true", "TR_hf_false"]
res = {}
for name, cc, feats in (("stage1", c1, FEATURES), ("stage2", c2, MODEL2_FEATURES)):
    m = {g: cc[groups == g].mean(0) for g in G}
    print(f"\n=== {name}: mean logit by group:", {g: round(float(m[g].sum()), 2) for g in G}, "n:", {g: int((groups == g).sum()) for g in G})
    diff = m["FR_rej"][:-1] - m["TR_true"][:-1]
    order = np.argsort(-np.abs(diff))[:30]
    print(f"{'feature':24s} " + " ".join(f"{g:>10s}" for g in G) + "   FRrej-TRtrue  | median value FR_rej / TR_true / TR_hf_true")
    rows = []
    for j in order:
        f = feats[j]
        med = [float(np.median(d[f][groups == g])) for g in ("FR_rej", "TR_true", "TR_hf_true")]
        print(f"{f:24s} " + " ".join(f"{m[g][j]:10.3f}" for g in G) + f"   {diff[j]:+.3f}  | {med[0]:.3g} / {med[1]:.3g} / {med[2]:.3g}")
        rows.append((f, float(diff[j]), [float(m[g][j]) for g in G], med))
    res[name] = rows
json.dump(res, open(OUT / "a3_shap.json", "w"), indent=1)
