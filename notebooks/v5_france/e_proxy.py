"""E: measure shift-robust feature variants with a single-stage proxy used identically in every arm.
Protocol: LightGBM (ber DEFAULT_PARAMS, lr 0.2, seed 42, 5 threads) on the in_group rows of 35% of the v4 FIT
entities (hash bucket), early stopping on tune_in; score the full tune / holdout files; decision = ber
choose_decision (fit on tune, rule chosen on tune halves); report holdout macro F0.5 (all, India, US, templated
names). France: score all French test pairs, decide with the fitted params (no EM shift, as v4), and report
label-free checks. Usage: e_proxy.py ARM   (ARM in BASE, NOCUT, PCT, NOADMIN, COMBO, ...)."""
import json, sys, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, lightgbm as lgb
from common import *
from ber.features import FEATURES
from ber.model import DEFAULT_PARAMS
from ber.config import load_config
from ber.pipeline import _group_arrays
from ber.decision import choose_decision, decide, prepare, choose_k, calibrate, MissingModel, Calibrator, _subset, macro_f05

ARM = sys.argv[1]
PCT_FEATS = ["l_name_freq", "r_name_freq", "r_core_l_freq", "r_same_name_cands", "l_n_cand", "r_n_cand",
             "l_rank", "r_rank", "lrank", "block_score"]
DROP = {"BASE": [], "NOCUT": ["cut_p"], "NOADMIN": ["admin_eq", "admin_diff"], "PCT": [],
        "COMBO": ["cut_p", "admin_eq", "admin_diff"], "NOCROWD": ["l_name_freq", "r_name_freq", "r_core_l_freq",
        "r_same_name_cands", "l_n_cand", "r_n_cand", "l_rank", "lrank"]}[ARM]
PCT = ARM in ("PCT", "COMBO")
feats = [f for f in FEATURES if f not in DROP]
W = TMP / f"arm_{ARM}"; W.mkdir(exist_ok=True)
c = connect()
t0 = time.time()
log = open(OUT / f"e_{ARM}.log", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()

# per-(world, country) reference quantiles for the percentile transform (adversarial samples: same sampling in both worlds)
REF = {}
if PCT:
    for world, path in (("train", TMP / "adv_train.parquet"), ("test", TMP / "adv_test.parquet")):
        t = pq.read_table(path, columns=["country", *PCT_FEATS])
        cc = np.array([s.lower() for s in t.column("country").to_pylist()])
        for ctry in np.unique(cc):
            m = cc == ctry
            REF[(world, ctry)] = {f: np.sort(t.column(f).to_numpy(zero_copy_only=False).astype(np.float64)[m]) for f in PCT_FEATS}

def transform(d, world):
    """Batch dict (numpy columns incl. 'country') -> float32 matrix in `feats` order with the arm's transforms."""
    cols = {}
    for f in feats:
        v = d[f].astype(np.float64)
        if PCT and f in PCT_FEATS:
            out = np.full(len(v), np.nan)
            for ctry in np.unique(d["country"]):
                m = d["country"] == ctry
                ref = REF[(world, ctry.lower())][f]
                lo = np.searchsorted(ref, v[m], "left"); hi = np.searchsorted(ref, v[m], "right")
                out[m] = (lo + hi) / (2.0 * len(ref))
            v = out
        cols[f] = v
    return np.column_stack([cols[f].astype(np.float32) for f in feats])

def batches(sql, n=400_000):
    """Stream a DuckDB query as dicts of numpy arrays."""
    rdr = c.execute(sql).fetch_record_batch(n)
    for b in rdr:
        yield {k: b.column(i).to_numpy(zero_copy_only=False) for i, k in enumerate(b.schema.names)}

COLS = ", ".join(f"f.{x}" for x in FEATURES)
TRAIN_SQL = lambda src, where: f"""SELECT f.lid, f.rid, f.label, lower(n.country_norm) country, {COLS} FROM read_parquet('{src}') f
    JOIN '{CACHE}/norm_train_s1.parquet' n ON n.entity_id = f.lid WHERE {where}"""

def load_matrix(sql):
    """Whole query into (x, y)."""
    xs, ys = [], []
    for d in batches(sql):
        xs.append(transform(d, "train")); ys.append(d["label"].astype(np.int8))
    return np.vstack(xs), np.concatenate(ys)

if not (W / "model.txt").exists():
    x, y = load_matrix(TRAIN_SQL(RUN / "fit_in.parquet", "hash(f.lid) % 100 < 35"))
    xv, yv = load_matrix(TRAIN_SQL(RUN / "tune_in.parquet", "true"))
    say(f"[{ARM}] train rows {len(y):,} valid {len(yv):,} feats {len(feats)}  load {time.time()-t0:.0f}s")
    params = {**DEFAULT_PARAMS, "learning_rate": 0.2, "seed": 42, "num_threads": 5}
    dt = lgb.Dataset(x, y, feature_name=feats, free_raw_data=True); dv = lgb.Dataset(xv, yv, reference=dt, free_raw_data=True)
    del x, xv
    b = lgb.train(params, dt, 3000, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
    b.save_model(str(W / "model.txt"), num_iteration=b.best_iteration)
    say(f"[{ARM}] best_iter {b.best_iteration} valid logloss {b.best_score['valid_0']['binary_logloss']:.5f}  {time.time()-t0:.0f}s")
    del dt, dv
b = lgb.Booster(model_file=str(W / "model.txt"))

def score(sql, out, world, keep=("lid", "rid", "label")):
    """Score a query to parquet (lid, rid, p[, label])."""
    if out.exists():
        return
    wr = None
    for d in batches(sql):
        p = b.predict(transform(d, world)).astype(np.float32)
        cols = {k: d[k] for k in keep if k in d}; cols["p"] = p
        if "label" in cols: cols["label"] = cols["label"].astype(np.int8)
        t = pa.table(cols)
        wr = wr or pq.ParquetWriter(out, t.schema); wr.write_table(t)
    wr.close()

for g in ("tune", "holdout"):
    score(TRAIN_SQL(FEAT[g], "true"), W / f"pred_{g}.parquet", "train")
FR_SQL = f"""SELECT f.lid, f.rid, 'France' country, {COLS} FROM '{FEAT['test']}' f
    JOIN '{CACHE}/raw_test_s1.parquet' s ON s.entity_id = f.lid AND s.country = 'France'"""
score(FR_SQL, W / "pred_fr.parquet", "test", keep=("lid", "rid"))
say(f"[{ARM}] scored  {time.time()-t0:.0f}s")

# ---- in-domain: decision on tune, holdout macro F0.5 (all / per country / templated names)
cfg = load_config()
arr = {g: _group_arrays(c, cfg, W / f"pred_{g}.parquet", g) for g in ("tune", "holdout")}
rep = choose_decision(arr["tune"], arr["holdout"])
params = rep["params"]; rule = params["rule"]
res = {"arm": ARM, "rule": rule, "holdout": rep["holdout"], "tune_half_b": rep["selection"]["tune_half_b"]}
# per-entity attributes of holdout S1 in _group_arrays order (ORDER BY entity_id within the group)
from ber.split import group_where
ent = c.execute(f"""SELECT lower(n.country_norm) c, coalesce(fq.f, 0) f FROM '{CACHE}/raw_train_s1.parquet' s
    JOIN '{CACHE}/norm_train_s1.parquet' n ON n.entity_id = s.entity_id
    LEFT JOIN (SELECT name_core, count(*) f FROM '{CACHE}/norm_train_s1.parquet' GROUP BY 1) fq ON fq.name_core = n.name_core
    WHERE {group_where('s.entity_id', 'holdout')} ORDER BY s.entity_id""").fetchnumpy()
h = arr["holdout"]
mask = decide(params, h["lid_idx"], h["rid"], h["p"]) if rule == "expected_f" else None
if mask is None:
    from ber.decision import select
    mask = select(h["lid_idx"], h["rid"], h["p"], params["threshold"])
from ber.decision import per_entity_f05
pe = per_entity_f05(h["lid_idx"], mask, h["label"], h["n_true"])
res["holdout_decided"] = float(pe.mean())
for name, m in (("india", ent["c"] == "india"), ("us", ent["c"] == "us"), ("namefreq>=50", ent["f"] >= 50),
                ("namefreq>=200", ent["f"] >= 200)):
    res[f"holdout_{name}"] = (float(pe[m].mean()), int(m.sum()))
say(f"[{ARM}] HOLDOUT rule={rule} F0.5={res['holdout_decided']:.5f}  " +
    "  ".join(f"{k[8:]}={v[0]:.5f}(n={v[1]:,})" for k, v in res.items() if k.startswith("holdout_") and isinstance(v, tuple)))
say(f"[{ARM}] holdout by rule: {json.dumps(rep['holdout'])}")

# ---- France: decide (no EM shift) + label-free checks
c.execute(f"""CREATE OR REPLACE TABLE fp AS SELECT row_number() OVER () - 1 i, x.lid, x.rid, x.p,
    dense_rank() OVER (ORDER BY x.lid) - 1 li, dense_rank() OVER (ORDER BY x.rid) - 1 rc FROM '{W}/pred_fr.parquet' x""")
d = c.execute("SELECT li, rc, p FROM fp ORDER BY i").fetchnumpy()
li, rc, p = d["li"].astype(np.int64), d["rc"].astype(np.int64), d["p"].astype(np.float64)
prep = prepare(li, rc, p, int(params["max_k"]))
pc = calibrate(params, prep.is_best, p)
k_star, ef = choose_k(prep, pc, MissingModel.from_dict(params["missing"]), float(params["empty_weight"]),
                      Calibrator.from_dict(params["cal_empty"]))
from ber.decision import mask_from_k
fm = mask_from_k(prep, k_star) if rule == "expected_f" else None
if fm is None:
    from ber.decision import select
    fm = select(li, rc, p, params["threshold"])
res["france_expected_f05"] = float(ef.mean())
c.register("sel", pa.table({"i": np.flatnonzero(fm).astype(np.int64)}))
c.execute("CREATE OR REPLACE TABLE facc AS SELECT fp.lid, fp.rid, fp.p FROM fp JOIN sel USING (i)")
c.execute(f"COPY facc TO '{W}/fr_accepted.parquet' (FORMAT parquet)")
nS1 = 259452
q = c.execute(f"""WITH k AS (SELECT lid, count(*) k FROM facc GROUP BY 1)
    SELECT count(*) FROM facc""").fetchone()[0]
nonempty = c.execute("SELECT count(DISTINCT lid) FROM facc").fetchone()[0]
res["france_matches_per_s1"] = q / nS1; res["france_empty_share"] = 1 - nonempty / nS1
# offsets / distractor checks on accepted pairs
chk = c.execute(f"""SELECT count(*) FILTER (WHERE f.sg_num_changed = 1 AND f.sg_first_off IN (1, 2)),
        count(*) FILTER (WHERE f.sg_num_changed = 1 AND f.sg_first_off IN (-1, -2)),
        count(*) FILTER (WHERE f.sg_num_changed = 1 AND f.sg_first_off IN (3,4,5,7,9,11,13,21))
    FROM facc a JOIN '{FEAT['test']}' f USING (lid, rid)""").fetchone()
res["france_p12"], res["france_m12"], res["france_genoff"] = chk
# #6 agreement
c.execute(f"""CREATE OR REPLACE TABLE acc6 AS SELECT source1_entity_id lid, unnest(string_split(nullif(coalesce(matched_entity_ids,''),''), ',')) rid
    FROM read_csv('{SUB6}', delim='\t', header=true, all_varchar=true) s JOIN '{CACHE}/raw_test_s1.parquet' r ON r.entity_id = s.source1_entity_id AND r.country = 'France'""")
res["france_added_vs6"] = c.execute("SELECT count(*) FROM facc a ANTI JOIN acc6 b USING (lid, rid)").fetchone()[0]
res["france_removed_vs6"] = c.execute("SELECT count(*) FROM acc6 b ANTI JOIN facc a USING (lid, rid)").fetchone()[0]
# same-name cells (C4): accepted at d = 0 per cell, and accepted at baseline offsets (near-certain FPs)
cells = c.execute(f"""SELECT sb, ce, d = 0 z, count(*) FROM '{TMP}/fr_cells.parquet' x JOIN facc a USING (lid, rid) GROUP BY 1, 2, 3 ORDER BY 3 DESC, 1, 2""").fetchall()
res["france_cells"] = cells
# rejected clean slice of v4 (p2 < 0.1): how does p move, how many accepted now
rs = c.execute(f"""WITH s AS (SELECT f.lid, f.rid FROM '{FEAT['test']}' f JOIN '{CACHE}/raw_test_s1.parquet' r ON r.entity_id = f.lid AND r.country = 'France'
        JOIN '{RUN}/pred2_test.parquet' p2 USING (lid, rid) WHERE {CLEAN} AND p2.p < 0.1)
    SELECT count(*), avg(x.p), quantile_cont(x.p, 0.5), avg((x.p >= 0.5)::INT), count(a.rid)
    FROM s JOIN '{W}/pred_fr.parquet' x USING (lid, rid) LEFT JOIN facc a USING (lid, rid)""").fetchone()
res["france_rejslice"] = rs
say(f"[{ARM}] FRANCE expF0.5(model)={res['france_expected_f05']:.4f} matches/S1={res['france_matches_per_s1']:.3f} empty={res['france_empty_share']:.4f} "
    f"+1/+2={chk[0]} -1/-2={chk[1]} genoff={chk[2]} added_vs6={res['france_added_vs6']} removed_vs6={res['france_removed_vs6']}")
say(f"[{ARM}] France same-name cells accepted (street, city_eq, d==0, n): {cells}")
say(f"[{ARM}] France v4-rejected clean slice: n={rs[0]} mean p={rs[1]:.3f} median p={rs[2]:.4f} share p>=.5={rs[3]:.3f} accepted={rs[4]}")
json.dump(res, open(OUT / f"e_{ARM}.json", "w"), indent=1, default=float)
say(f"[{ARM}] done {time.time()-t0:.0f}s")
