"""Task 3b: extra entity-level stage-2 features from stage-1 p (pred1_*), joined to the top-K stage-2 features.

Writes artifacts/v5_iu/s2x_{fit_in,tune_in,tune,holdout}.parquet with lid, rid, label, in_group, TOP features, E features.
"""
import json, time
from common import RUN, OUT, con
import pyarrow.parquet as pq

K = 60
imp = json.loads((RUN / "manifest.json").read_text())["importance_gain_stage2"]
TOP = [k for k, _ in sorted(imp.items(), key=lambda x: -x[1])[:K]]
E = ["e_l_n_best", "e_l_sum_best", "e_l_rank_best", "e_l_next_gap", "e_l_prev_gap", "e_l_n10", "e_l_n30",
     "e_ls_rank", "e_ls_n50", "e_ls_max_other", "e_r_n", "e_r_second", "e_r_other_lgap", "e_r_other_ln50", "e_p_over_lmax"]


def esql(pred):
    """DuckDB SELECT of (lid, rid, E...) over a stage-1 prediction parquet (all rows incl. rival context)."""
    return f"""
    WITH b AS (SELECT lid, rid, p::DOUBLE p, starts_with(rid,'S3-') s3,
        row_number() OVER (PARTITION BY rid ORDER BY p DESC, lid) r_rn, count(*) OVER (PARTITION BY rid) r_n
        FROM read_parquet('{pred}')),
    rt AS (SELECT rid, max(CASE WHEN r_rn=1 THEN lid END) l1, max(CASE WHEN r_rn=1 THEN p END) p1,
                       max(CASE WHEN r_rn=2 THEN lid END) l2, max(CASE WHEN r_rn=2 THEN p END) p2 FROM b WHERE r_rn<=2 GROUP BY rid),
    lt AS (SELECT lid, max(p) lmax, sum((p>=0.5)::INT) ln50 FROM b GROUP BY lid),
    c AS (SELECT b.*, r_rn=1 best,
        sum(CASE WHEN r_rn=1 THEN 1 ELSE 0 END) OVER wl n_best,
        sum(CASE WHEN r_rn=1 THEN p ELSE 0 END) OVER wl sum_best,
        sum((p>=0.1)::INT) OVER wl n10, sum((p>=0.3)::INT) OVER wl n30,
        lead(p) OVER (PARTITION BY lid ORDER BY p DESC, rid) nxt, lag(p) OVER (PARTITION BY lid ORDER BY p DESC, rid) prv,
        rank() OVER (PARTITION BY lid, s3 ORDER BY p DESC) ls_rank,
        sum((p>=0.5)::INT) OVER (PARTITION BY lid, s3) ls_n50,
        max(p) OVER (PARTITION BY lid) lmax_all
        FROM b WINDOW wl AS (PARTITION BY lid)),
    bestp AS (SELECT lid, list(p) bp FROM c WHERE best GROUP BY lid),
    ls2 AS (SELECT lid, s3, max(p) m1, arg_max(rid, p) a1 FROM c GROUP BY lid, s3)
    SELECT c.lid, c.rid,
      n_best::FLOAT e_l_n_best, sum_best::FLOAT e_l_sum_best,
      (1 + len(list_filter(coalesce(bp,[]), x -> x > c.p)))::FLOAT e_l_rank_best,
      (c.p - coalesce(nxt, 0))::FLOAT e_l_next_gap, (coalesce(prv, c.p) - c.p)::FLOAT e_l_prev_gap,
      n10::FLOAT e_l_n10, n30::FLOAT e_l_n30, ls_rank::FLOAT e_ls_rank, ls_n50::FLOAT e_ls_n50,
      (CASE WHEN ls2.a1 = c.rid THEN -1 ELSE ls2.m1 END)::FLOAT e_ls_max_other,
      r_n::FLOAT e_r_n,
      (CASE WHEN best THEN coalesce(rt.p2, 0) ELSE rt.p1 END)::FLOAT e_r_second,
      (CASE WHEN best THEN coalesce(l2t.lmax - rt.p2, -1) ELSE l1t.lmax - rt.p1 END)::FLOAT e_r_other_lgap,
      (CASE WHEN best THEN coalesce(l2t.ln50, -1) ELSE l1t.ln50 END)::FLOAT e_r_other_ln50,
      (c.p / greatest(lmax_all, 1e-9))::FLOAT e_p_over_lmax
    FROM c JOIN rt ON rt.rid = c.rid LEFT JOIN lt l1t ON l1t.lid = rt.l1 LEFT JOIN lt l2t ON l2t.lid = rt.l2
    LEFT JOIN bestp ON bestp.lid = c.lid JOIN ls2 ON ls2.lid = c.lid AND ls2.s3 = c.s3"""


if __name__ == "__main__":
    c = con()
    for g, src, pred in (("fit_in", "s2_fit_in", "pred1_fit"), ("tune_in", "s2_tune_in", "pred1_tune"),
                         ("tune", "s2_tune", "pred1_tune"), ("holdout", "s2_holdout", "pred1_holdout")):
        t0 = time.time()
        out = OUT / f"s2x_{g}.parquet"
        if out.exists():
            continue
        ef = OUT / f"e_{pred}.parquet"
        if not ef.exists():
            c.execute(f"COPY ({esql(RUN / (pred + '.parquet'))}) TO '{ef}' (FORMAT parquet)")
        cols = ", ".join(f"f.{x}" for x in TOP)
        ecols = ", ".join(f"e.{x}" for x in E)
        c.execute(f"""COPY (SELECT f.lid, f.rid, f.label, f.in_group, {cols}, {ecols} FROM read_parquet('{RUN / (src + '.parquet')}') f
                  JOIN read_parquet('{ef}') e ON e.lid = f.lid AND e.rid = f.rid) TO '{out}' (FORMAT parquet)""")
        n1 = pq.ParquetFile(out).metadata.num_rows; n0 = pq.ParquetFile(RUN / (src + '.parquet')).metadata.num_rows
        print(g, n0, n1, round(time.time() - t0, 1), flush=True)
        assert n0 == n1
