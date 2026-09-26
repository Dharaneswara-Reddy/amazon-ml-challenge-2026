"""Check 4: ID-existence + format edge cases for submission #6, via DuckDB
(equivalent to validate_submission.py --check-ids but memory-bounded).

- every matched/candidate id actually exists in test_source2/3
- no S1 ids referenced that aren't self-matches (shouldn't happen)
- every match also appears in that S1's candidate list
- exact row count / header / duplicate-row sanity (cheap, sanity re-check)
"""
import duckdb, os
os.makedirs("artifacts/v5_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("set memory_limit='2GB'")
con.execute("set threads=3")
con.execute("set temp_directory='artifacts/v5_tmp'")
CACHE = "artifacts/cache"
SUB = "submissions/06_20260926-1234"

def q(sql):
    return con.execute(sql).fetchall()

con.execute(f"""
  create or replace view mres as
  select * from read_csv('{SUB}/matching_results.tsv', delim='\t', header=true,
    columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}})
""")
con.execute(f"""
  create or replace view cpairs as
  select * from read_csv('{SUB}/candidate_pairs.tsv', delim='\t', header=true,
    columns={{'source1_entity_id':'VARCHAR','candidate_entity_ids':'VARCHAR'}})
""")

print("=== duplicate s1 rows in mres / cpairs ===")
print(q("select count(*) from (select source1_entity_id, count(*) c from mres group by 1 having count(*)>1)"))
print(q("select count(*) from (select source1_entity_id, count(*) c from cpairs group by 1 having count(*)>1)"))

print("=== every test S1 has exactly one row in mres ===")
print(q(f"""
  select
    (select count(*) from mres) as mres_rows,
    (select count(distinct entity_id) from '{CACHE}/raw_test_s1.parquet') as test_s1_n,
    (select count(*) from mres m left join '{CACHE}/raw_test_s1.parquet' t on t.entity_id = m.source1_entity_id where t.entity_id is null) as mres_rows_not_in_test,
    (select count(*) from '{CACHE}/raw_test_s1.parquet' t left join mres m on t.entity_id = m.source1_entity_id where m.source1_entity_id is null) as test_s1_missing_from_mres
"""))

print("=== matched ids: existence, self-matches, wrong prefix, intra-list dup ===")
print(q(f"""
  with exploded as (
    select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) as mid
    from mres where matched_entity_ids <> ''
  )
  select
    count(*) as total_matched_ids,
    count(*) filter (where mid like 'S1-%') as self_matches,
    count(*) filter (where mid not like 'S2-%' and mid not like 'S3-%') as wrong_prefix,
    count(*) filter (where mid like 'S2-%' and mid not in (select entity_id from '{CACHE}/raw_test_s2.parquet')) as s2_not_exist,
    count(*) filter (where mid like 'S3-%' and mid not in (select entity_id from '{CACHE}/raw_test_s3.parquet')) as s3_not_exist
  from exploded
"""))

print("=== intra-list duplicate ids within a single matched_entity_ids cell ===")
print(q("""
  with exploded as (
    select source1_entity_id, matched_entity_ids,
      len(string_split(matched_entity_ids, ',')) as n_raw,
      len(list_distinct(string_split(matched_entity_ids, ','))) as n_distinct
    from mres where matched_entity_ids <> ''
  )
  select count(*) from exploded where n_raw <> n_distinct
"""))

print("=== every matched id also present in that S1's candidate list ===")
print(q("""
  with m as (
    select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) as mid
    from mres where matched_entity_ids <> ''
  ),
  j as (
    select m.source1_entity_id, m.mid, c.candidate_entity_ids
    from m left join cpairs c on c.source1_entity_id = m.source1_entity_id
  )
  select count(*) as matches_missing_from_candidates
  from j
  where candidate_entity_ids is null or not contains(',' || candidate_entity_ids || ',', ',' || mid || ',')
"""))

con.close()
