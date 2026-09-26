"""Task 2b: cross-group one-to-one losses: true holdout pairs whose right record's best candidate is an out-of-group S1."""
from common import RUN, con, note
from ber.split import group_where
c = con()
pr = RUN / "pred2_holdout.parquet"
q = f"""WITH pr AS (SELECT lid, rid, p, label FROM read_parquet('{pr}')),
  w AS (SELECT rid, arg_max(lid, p) wl, max(p) wp, arg_max(label, p) wlab FROM pr GROUP BY rid),
  lost AS (SELECT pr.*, w.wl, w.wp, w.wlab FROM pr JOIN w USING (rid) WHERE pr.label = 1 AND {group_where('pr.lid','holdout')}
           AND NOT {group_where('w.wl','holdout')})
  SELECT count(*) n, sum((wlab=1)::INT) winner_true, sum((wp < 0.5)::INT) rival_lt05, sum((wp<0.5 AND p>=0.5)::INT) rival_lt05_true_ge05,
   sum((wp<0.3 AND p>=0.3)::INT) r03, round(avg(wp),3) avg_wp, round(avg(p),3) avg_p, sum((wp-p<0.1)::INT) close_01,
   sum(starts_with(rid,'S3-')::INT) s3 FROM lost"""
r = c.execute(q).fetchall()
cols = [d[0] for d in c.description]
d = dict(zip(cols, r[0]))
print(d)
note("## T2b. Cross-group one-to-one losses (true holdout pair, right's best candidate is an out-of-group S1)\n\n"
     f"`{d}`\n\n(winner_true: the rival is ALSO labelled true — impossible under one-to-one, i.e. GT duplicates; "
     "rival_lt05: rival pair p<0.5 so the rival entity would likely reject R -> second-choice could recover it)")
