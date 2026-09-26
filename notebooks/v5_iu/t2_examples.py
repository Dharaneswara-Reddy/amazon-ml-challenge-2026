"""Task 2c: examples of decision FNs, FPs and n=1 misses with raw text (holdout, v4)."""
import duckdb
from common import OUT, ROOT, note
C = ROOT / "artifacts/cache"
db = duckdb.connect(); db.execute("SET threads=3; SET memory_limit='1GB'")
db.execute(f"""CREATE VIEW raw AS SELECT * FROM read_parquet(['{C}/raw_train_s1.parquet','{C}/raw_train_s2.parquet','{C}/raw_train_s3.parquet'])""")
db.execute(f"CREATE TABLE pj AS SELECT p.*, e.s1_id, e.nt, e.k, e.country FROM read_parquet('{OUT}/holdout_pairs_v4.parquet') p JOIN read_parquet('{OUT}/holdout_ent_v4.parquet') e USING (li)")
def show(title, where, n=12):
    rows = db.execute(f"""WITH x AS (SELECT * FROM pj WHERE {where} ORDER BY hash(rid_s) LIMIT {n})
      SELECT x.s1_id, x.nt, x.k, round(x.p,3) p, a.business_name||' | '||a.business_address L, x.rid_s, b.business_name||' | '||b.business_address R,
        (SELECT string_agg(r2.business_name||' |'||round(y.p,2)||'|'||(CASE WHEN y.lab THEN 'T' ELSE 'F' END)||(CASE WHEN y.sel THEN '*' ELSE '' END), ' ;; ')
          FROM pj y JOIN raw r2 ON r2.entity_id = y.rid_s WHERE y.li = x.li AND y.best) sib
      FROM x JOIN raw a ON a.entity_id = x.s1_id JOIN raw b ON b.entity_id = x.rid_s""").fetchall()
    txt = f"### {title}\n" + "\n".join(f"- `{r[0]}` n_true={int(r[1])} k={r[2]} p={r[3]}\n  - L: {r[4]}\n  - R `{r[5]}`: {r[6]}\n  - entity's R-best cands (name|p|label, *=selected): {r[7]}" for r in rows)
    print(txt)
    return txt
t = [show("Decision FN (R-best, entity has picks, p 0.3-0.7)", "lab AND NOT sel AND best AND k>0 AND p BETWEEN 0.3 AND 0.7", 8),
     show("FP with p > 0.99 (label noise?)", "sel AND NOT lab AND p > 0.99", 6),
     show("FP on matched entities, p 0.7-0.95", "sel AND NOT lab AND nt>0 AND p BETWEEN 0.7 AND 0.95", 6),
     show("n=1 entity, true pair in cands, empty prediction", "nt=1 AND lab AND NOT sel AND k=0", 6)]
note("## T2c. Examples (random samples, holdout v4)\n\n" + "\n\n".join(t))
