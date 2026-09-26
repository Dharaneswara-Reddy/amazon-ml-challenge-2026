"""Is a distractor derived from its parent S1 or from one of the parent's noisy true copies (siblings)?

Compares raw (case-sensitive) name/address similarity of the distractor to the parent S1 vs to its closest sibling.
Control: the same statistic for a true copy vs the other true copies of its S1.
"""
import pyarrow.parquet as pq
from rapidfuzz.fuzz import ratio
from f_common import OUT, connect

con = connect()
rows = con.execute(f"""SELECT d.rid, d.ln, d.la, d.rn, d.ra, list(s.sn), list(s.sa)
   FROM '{OUT}/f01_dist.parquet' d JOIN '{OUT}/f01_sib.parquet' s USING (rid) WHERE d.best_s >= 1.7 GROUP BY ALL""").fetchall()
tr = con.execute(f"""WITH t AS (SELECT * FROM '{OUT}/f01_true.parquet' WHERE md5_number(rid)%100 < 2)
   SELECT t.rid, t.ln, t.la, t.rn, t.ra, list(r.business_name), list(r.business_address)
   FROM t JOIN read_parquet('{OUT}/../../../artifacts/cache/train_gt_pairs.parquet') g ON g.s1_id = t.lid AND g.match_id <> t.rid
   JOIN (SELECT * FROM read_parquet('/home/toji339/Documents/Amazon_ML_Hackathon/artifacts/cache/raw_train_s2.parquet') UNION ALL
         SELECT * FROM read_parquet('/home/toji339/Documents/Amazon_ML_Hackathon/artifacts/cache/raw_train_s3.parquet')) r ON r.entity_id = g.match_id
   GROUP BY ALL""").fetchall()
for tag, rs in (("distractor", rows), ("true copy", tr)):
    closer_sib = closer_s1 = 0
    dn = []
    for _, ln, la, rn, ra, sn, sa in rs:
        s1 = ratio(rn or "", ln or "") + ratio(ra or "", la or "")
        sb = max(ratio(rn or "", x or "") + ratio(ra or "", y or "") for x, y in zip(sn, sa))
        closer_sib += sb > s1
        closer_s1 += s1 >= sb
        dn.append(s1 - sb)
    import numpy as np
    print(tag, len(rs), "closer to S1:", closer_s1, "closer to a sibling:", closer_sib, "mean(s1-sib):", round(float(np.mean(dn)), 1))
