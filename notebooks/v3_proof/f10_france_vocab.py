"""France (test): self-labeled distractor vocabulary. Near-duplicate best-candidate pairs are split by the NUMBER
signature (sole offset in S>=3 = likely distractor; all numbers equal = likely true copy) and the inserted /
replaced name words are counted in each group. Shows that the name vocab can be learned from unlabeled test data."""
from collections import Counter
from multiprocessing import Pool

from f07_france import ops
from f_common import C, connect, raw_right

S3 = {3, 4, 5, 7, 9, 11, 13, 21}
con = connect()
rows = con.execute(f"""WITH c AS (SELECT rid, arg_max(lid, coalesce(jw_core,0)+coalesce(jw_addr,0)) lid,
                             max(coalesce(jw_core,0)+coalesce(jw_addr,0)) s
                      FROM read_parquet('{C}/cand_test.parquet') WHERE md5_number(rid) % 100 < 12 GROUP BY rid HAVING s >= 1.7)
    SELECT l.country, NULL, l.business_name, l.business_address, r.business_name, r.business_address
    FROM c JOIN read_parquet('{C}/raw_test_s1.parquet') l ON l.entity_id = c.lid JOIN {raw_right('test')} r ON r.entity_id = c.rid
    WHERE l.country = 'France'""").fetchall()
with Pool(2) as pool:
    res = pool.map(ops, rows, chunksize=2000)
grp = {"dist-like": Counter(), "true-like": Counter()}
rep = {"dist-like": Counter(), "true-like": Counter()}
n = Counter()
for _, _, d in res:
    m, nm = d["num"], d["name"]
    if m is None or nm is None:
        continue
    offs = [v for (k, v), _, _ in m["pairs"] if k == "offset"]
    if len(m["pairs"]) == 1 and len(offs) == 1 and offs[0] in S3:
        g = "dist-like"
    elif len(m["pairs"]) == 0 and m["nl"] > 0:
        g = "true-like"
    else:
        continue
    n[g] += 1
    grp[g].update(w for w, _, _ in nm["inserted"])
    rep[g].update(f"{x}>{y}" for x, y in nm["replaced"])
print("groups:", dict(n))
for g in grp:
    print(g, "inserted:", grp[g].most_common(25))
    print(g, "replaced:", rep[g].most_common(20))
