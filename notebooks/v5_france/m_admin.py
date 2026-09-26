"""M: admin (region/department) disagreement in France: admin_diff = 1 pairs among near-certain true pairs (same core
name, same number, same street; C4 cells) and their #6 acceptance; train reference (true rate of admin_diff = 1)."""
from common import *
c = connect()
c.execute(f"""CREATE TABLE a6 AS SELECT source1_entity_id lid, unnest(string_split(nullif(coalesce(matched_entity_ids,''),''), ',')) rid
    FROM read_csv('{SUB6}', delim='\t', header=true, all_varchar=true)""")
print("France sure-true pairs by admin_eq/admin_diff: n, accepted #6:", c.execute(f"""SELECT f.admin_eq, f.admin_diff, count(*), count(a.rid)
    FROM '{TMP}/fr_cells.parquet' x JOIN '{FEAT['test']}' f USING (lid, rid) LEFT JOIN a6 a USING (lid, rid)
    WHERE x.d = 0 AND x.sb = 'same' GROUP BY 1, 2 ORDER BY 1, 2""").fetchall())
print("France all pairs admin_diff share / accepted with admin_diff:", c.execute(f"""SELECT avg(f.admin_diff), count(*) FILTER (WHERE f.admin_diff = 1 AND a.rid IS NOT NULL)
    FROM '{FEAT['test']}' f JOIN '{CACHE}/raw_test_s1.parquet' s ON s.entity_id = f.lid AND s.country = 'France' LEFT JOIN a6 a USING (lid, rid)""").fetchall())
print("examples:", c.execute(f"""SELECT l.business_address, r.business_address, (a.rid IS NOT NULL) FROM '{TMP}/fr_cells.parquet' x JOIN '{FEAT['test']}' f USING (lid, rid)
    JOIN '{CACHE}/raw_test_s1.parquet' l ON l.entity_id = x.lid JOIN read_parquet(['{CACHE}/raw_test_s2.parquet','{CACHE}/raw_test_s3.parquet']) r ON r.entity_id = x.rid
    LEFT JOIN a6 a USING (lid, rid) WHERE x.d = 0 AND x.sb = 'same' AND f.admin_diff = 1 ORDER BY hash(x.rid) LIMIT 6""").fetchall())
print("train holdout clean-number same-core pairs by admin_diff: n, true rate:", c.execute(f"""SELECT admin_diff, count(*), avg(label) FROM '{FEAT['holdout']}' f
    WHERE in_group = 1 AND {CLEAN} GROUP BY 1""").fetchall())
