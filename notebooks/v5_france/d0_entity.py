"""D0: entity-level view of #6 per test country (empty share, matches per S1) vs train truth, and the entity-level
weight of the rejected clean slice (how many of those S1 have an EMPTY prediction in #6)."""
from common import *
c = connect()
c.execute(f"""CREATE TABLE sub AS SELECT source1_entity_id lid, coalesce(matched_entity_ids,'') m,
    CASE WHEN coalesce(matched_entity_ids,'') = '' THEN 0 ELSE len(string_split(matched_entity_ids, ',')) END k
    FROM read_csv('{SUB6}', delim='\t', header=true, all_varchar=true)""")
print("test #6:", c.execute(f"""SELECT s.country, count(*), avg((k=0)::INT), avg(k), avg((k=1)::INT), avg((k>=5)::INT)
    FROM sub JOIN '{CACHE}/raw_test_s1.parquet' s ON s.entity_id = sub.lid GROUP BY 1 ORDER BY 1""").fetchall())
print("train truth:", c.execute(f"""SELECT n.country_norm, count(*), avg((coalesce(g.n_matches,0)=0)::INT), avg(coalesce(g.n_matches,0)),
    avg((coalesce(g.n_matches,0)=1)::INT), avg((coalesce(g.n_matches,0)>=5)::INT)
    FROM '{CACHE}/norm_train_s1.parquet' n LEFT JOIN '{CACHE}/train_gt_s1.parquet' g ON g.s1_id = n.entity_id GROUP BY 1""").fetchall())
c.execute(f"""CREATE TABLE acc AS SELECT lid, unnest(string_split(nullif(m,''), ',')) rid FROM sub""")
c.execute(f"""CREATE TABLE rej AS SELECT f.lid, f.rid, p.p FROM '{FEAT['test']}' f
    JOIN '{CACHE}/raw_test_s1.parquet' s ON s.entity_id = f.lid AND s.country = 'France'
    JOIN '{RUN}/pred2_test.parquet' p USING (lid, rid)
    WHERE {CLEAN} AND NOT EXISTS (SELECT 1 FROM acc a WHERE a.lid = f.lid AND a.rid = f.rid)""")
print("rejected clean France pairs / distinct S1 / p<0.1:", c.execute("SELECT count(*), count(DISTINCT lid), count(*) FILTER (WHERE p<0.1) FROM rej").fetchall())
print("their S1 by #6 k (0 = empty prediction):", c.execute("""SELECT least(k,5), count(*) FROM (SELECT DISTINCT lid FROM rej) JOIN sub USING (lid) GROUP BY 1 ORDER BY 1""").fetchall())
print("rejected clean pairs per S1:", c.execute("SELECT n, count(*) FROM (SELECT lid, least(count(*),5) n FROM rej GROUP BY 1) GROUP BY 1 ORDER BY 1").fetchall())
