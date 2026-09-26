"""Task 2: error analysis of v4 holdout decisions (FN, one-to-one, n=1 entities, FP). No pandas: duckdb over arrow."""
import json
import numpy as np, pyarrow as pa, duckdb
from common import RUN, OUT, arrays, params, note
from ber.decision import decide, per_entity_f05, prepare, calibrate, choose_k, MissingModel, Calibrator

h = arrays(RUN / "pred2_holdout.parquet", "holdout", extra_ids=True)
P = params()
li, rid, p, lab, nt = h["lid_idx"], h["rid"], h["p"], h["label"], h["n_true"]
n = len(nt)
sel = decide(P, li, rid, p)
prep = prepare(li, rid, p, int(P["max_k"]), n)
pc = calibrate(P, prep.is_best, p)
k_star, ef = choose_k(prep, pc, MissingModel.from_dict(P["missing"]), float(P["empty_weight"]), Calibrator.from_dict(P["cal_empty"]))
f = per_entity_f05(li, sel, lab, nt)
s3 = np.array([s.startswith("S3-") for s in h["rid_s"]])
pairs = pa.table(dict(li=li, rc=rid, p=p, pc=pc, lab=lab, sel=sel, best=prep.is_best, s3=s3, rid_s=h["rid_s"].astype(str)))
ent = pa.table(dict(li=np.arange(n), nt=nt, f=f, k=k_star, ef=ef, ncand=prep.n_cand, country=h["country"].astype(str),
                    s1_id=h["s1_id"].astype(str)))
pa.parquet.write_table(pairs, OUT / "holdout_pairs_v4.parquet"); pa.parquet.write_table(ent, OUT / "holdout_ent_v4.parquet")
db = duckdb.connect(); db.execute("SET threads=4"); db.register("pairs", pairs); db.register("ent0", ent)
db.execute("""CREATE TABLE ent AS SELECT e.*, coalesce(x.tp,0) tp, coalesce(x.fp,0) fp, coalesce(x.nsel,0) nsel,
  coalesce(x.cand_true,0) cand_true, coalesce(x.npairs,0) npairs, coalesce(x.maxp,0) maxp FROM ent0 e LEFT JOIN
  (SELECT li, sum((sel AND lab)::INT) tp, sum((sel AND NOT lab)::INT) fp, sum(sel::INT) nsel, sum(lab::INT) cand_true,
   count(*) npairs, max(p) maxp FROM pairs GROUP BY li) x USING (li)""")
db.execute("CREATE TABLE pj AS SELECT p.*, e.nt, e.k, e.ncand, e.country, e.tp ent_tp, e.nsel ent_nsel FROM pairs p JOIN ent e USING (li)")
q = lambda s: db.execute(s).fetchall()
out = {}
out["FN_split"] = q("""SELECT best, k=0 AS ent_k0, count(*) FROM pj WHERE lab AND NOT sel GROUP BY ALL ORDER BY ALL""")
# FN not R-best: who won the right record? (in-group rival, labelled false)
out["FN_notbest_winner"] = q("""WITH w AS (SELECT rc, arg_max(lab, p) wlab, arg_max(sel, p) wsel, max(p) wp FROM pairs GROUP BY rc)
  SELECT w.wlab, w.wsel, count(*), round(avg(w.wp - pj.p),3) gap FROM pj JOIN w USING (rc) WHERE pj.lab AND NOT pj.sel AND NOT pj.best GROUP BY ALL""")
out["FN_best_by_src_k0"] = q("SELECT s3, k=0, count(*) FROM pj WHERE lab AND NOT sel AND best GROUP BY ALL ORDER BY ALL")
out["TP_by_src"] = q("SELECT s3, count(*) FROM pj WHERE lab AND sel GROUP BY ALL ORDER BY ALL")
out["FN_best_by_nt"] = q("SELECT least(nt,6), count(*) FROM pj WHERE lab AND NOT sel AND best GROUP BY ALL ORDER BY ALL")
out["FN_best_raw_p_bins"] = q("""SELECT CASE WHEN p<.01 THEN '<.01' WHEN p<.1 THEN '<.1' WHEN p<.3 THEN '<.3' WHEN p<.5 THEN '<.5'
  WHEN p<.7 THEN '<.7' WHEN p<.9 THEN '<.9' ELSE '>=.9' END b, count(*), sum((k=0)::INT) FROM pj WHERE lab AND NOT sel AND best GROUP BY ALL ORDER BY ALL""")
out["FN_best_country"] = q("SELECT country, count(*) FROM pj WHERE lab AND NOT sel AND best GROUP BY ALL")
out["FP"] = q("""SELECT nt=0 singleton_ent, best, count(*), round(avg(p),3), sum((p>0.99)::INT) FROM pj WHERE sel AND NOT lab GROUP BY ALL ORDER BY ALL""")
out["FP_by_src"] = q("SELECT s3, count(*) FROM pj WHERE sel AND NOT lab GROUP BY ALL ORDER BY ALL")
out["entity_loss_by_kind"] = q(f"""SELECT CASE WHEN f>=1-1e-9 THEN 'perfect'
   WHEN nt>0 AND nsel=0 AND cand_true=0 THEN 'no_true_in_cands_emptypred'
   WHEN nt>0 AND nsel=0 THEN 'empty_pred_true_in_cands'
   WHEN nt=0 THEN 'fp_on_singleton'
   WHEN fp>0 AND tp=0 THEN 'all_wrong'
   WHEN fp>0 THEN 'tp_and_fp'
   WHEN tp=cand_true THEN 'only_blocking/o2o_missing'
   ELSE 'partial_fn_in_cands' END kind, count(*), round(sum(1-f)/{n}*100,4) loss_pts FROM ent GROUP BY ALL ORDER BY 3 DESC""")
out["n1_entities_by_kind"] = q(f"""SELECT CASE WHEN f>=1-1e-9 THEN 'perfect' WHEN cand_true=0 THEN 'true_not_in_cands'
   WHEN nsel=0 THEN 'true_in_cands_empty_pred' WHEN fp>0 AND tp=0 THEN 'wrong_pick' WHEN fp>0 THEN 'tp_plus_fp' ELSE '?' END,
   count(*), round(sum(1-f)/{n}*100,4) FROM ent WHERE nt=1 GROUP BY ALL ORDER BY 3 DESC""")
out["loss_by_nt_country"] = q(f"SELECT country, least(nt,5), count(*), round(avg(f),4), round(sum(1-f)/{n}*100,4) FROM ent GROUP BY ALL ORDER BY ALL")
for k, v in out.items():
    print(k, v)
note("## T2. Error analysis (v4 holdout) — raw query output\n\n```\n" + "\n".join(f"{k}: {v}" for k, v in out.items()) + "\n```")
