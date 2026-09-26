"""G: coordinator check — very short (len <= 2, non-empty) S2/S3 business names in test: by country, examples,
and whether they sit in the France rejected clean slice / #6 accepted pairs."""
from common import *
c = connect()
for s in (2, 3):
    print(f"S{s} test very-short names by country:", c.execute(f"""SELECT n.country_norm, count(*) FILTER (WHERE length(r.business_name) BETWEEN 1 AND 2), count(*),
        round(100.0 * count(*) FILTER (WHERE length(r.business_name) BETWEEN 1 AND 2) / count(*), 3)
        FROM '{CACHE}/raw_test_s{s}.parquet' r JOIN '{CACHE}/norm_test_s{s}.parquet' n USING (entity_id) GROUP BY 1 ORDER BY 1""").fetchall())
    print(f"S{s} train very-short by country:", c.execute(f"""SELECT n.country_norm, round(100.0 * count(*) FILTER (WHERE length(r.business_name) BETWEEN 1 AND 2) / count(*), 3)
        FROM '{CACHE}/raw_train_s{s}.parquet' r JOIN '{CACHE}/norm_train_s{s}.parquet' n USING (entity_id) GROUP BY 1 ORDER BY 1""").fetchall())
print("France examples:", c.execute(f"""SELECT r.business_name, r.business_address FROM '{CACHE}/raw_test_s3.parquet' r JOIN '{CACHE}/norm_test_s3.parquet' n USING (entity_id)
    WHERE n.country_norm='france' AND length(r.business_name) BETWEEN 1 AND 2 USING SAMPLE 8 ROWS""").fetchall())
c.execute(f"""CREATE TABLE vs AS SELECT r.entity_id rid FROM read_parquet(['{CACHE}/raw_test_s2.parquet','{CACHE}/raw_test_s3.parquet']) r
    WHERE length(r.business_name) BETWEEN 1 AND 2""")
c.execute(f"""CREATE TABLE acc AS SELECT source1_entity_id lid, unnest(string_split(nullif(coalesce(matched_entity_ids,''),''), ',')) rid
    FROM read_csv('{SUB6}', delim='\t', header=true, all_varchar=true)""")
print("very-short rights: in candidates (by S1 country), accepted in #6:", c.execute(f"""SELECT s.country, count(DISTINCT f.rid), count(DISTINCT a.rid)
    FROM '{FEAT['test']}' f JOIN vs USING (rid) JOIN '{CACHE}/raw_test_s1.parquet' s ON s.entity_id = f.lid
    LEFT JOIN acc a USING (lid, rid) GROUP BY 1 ORDER BY 1""").fetchall())
print("very-short rights in France rejected clean slice (v4 p2<.1):", c.execute(f"""SELECT count(*) FROM '{FEAT['test']}' f JOIN vs USING (rid)
    JOIN '{CACHE}/raw_test_s1.parquet' s ON s.entity_id = f.lid AND s.country='France' JOIN '{RUN}/pred2_test.parquet' p USING (lid, rid)
    WHERE {CLEAN} AND p.p < 0.1""").fetchall())
c.execute(f"CREATE TABLE ex AS SELECT * FROM read_parquet(['{CACHE}/raw_test_s2.parquet','{CACHE}/raw_test_s3.parquet']) r WHERE length(r.business_name) BETWEEN 1 AND 2 AND r.entity_id IN (SELECT rid FROM vs)")
for r in c.execute(f"""SELECT x.business_name, x.business_address, l.business_name, l.business_address, round(p.p, 3), (a.rid IS NOT NULL) acc
    FROM (SELECT * FROM ex WHERE business_address ILIKE '%france%' OR true) x
    JOIN '{FEAT['test']}' f ON f.rid = x.entity_id AND f.r_rank = 1 JOIN '{CACHE}/raw_test_s1.parquet' l ON l.entity_id = f.lid AND l.country = 'France'
    JOIN '{RUN}/pred2_test.parquet' p ON p.lid = f.lid AND p.rid = f.rid LEFT JOIN acc a ON a.lid = f.lid AND a.rid = f.rid
    ORDER BY hash(x.entity_id) LIMIT 14""").fetchall():
    print(r)
print("France very-short rights: accepted best pair has initials of S1 name? share where name == initials of S1 words:", c.execute(f"""
    SELECT count(*), avg((lower(x.business_name) = lower(list_aggr(list_transform(string_split(n.name_norm, ' '), t -> t[1]), 'string_agg', '')))::INT)
    FROM ex x JOIN acc a ON a.rid = x.entity_id JOIN '{CACHE}/norm_test_s1.parquet' n ON n.entity_id = a.lid AND n.country_norm = 'france'""").fetchall())
