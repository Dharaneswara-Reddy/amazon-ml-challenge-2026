"""Check 3 (retry, materialized tables to avoid re-parsing large TSVs per query):
submission #6 output vs train-truth expectations, per country.
"""
import duckdb, os, time
os.makedirs("artifacts/v5_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("set memory_limit='2GB'")
con.execute("set threads=3")
con.execute("set temp_directory='artifacts/v5_tmp'")
CACHE = "artifacts/cache"
SUB = "submissions/06_20260926-1234"

def q(label, sql):
    t0 = time.time()
    r = con.execute(sql).fetchall()
    print(f"{label} ({time.time()-t0:.1f}s):", r)
    return r

t0 = time.time()
con.execute(f"""
  create or replace table mres as
  select * from read_csv('{SUB}/matching_results.tsv', delim='\t', header=true,
    columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}})
""")
con.execute(f"""
  create or replace table cpairs as
  select * from read_csv('{SUB}/candidate_pairs.tsv', delim='\t', header=true,
    columns={{'source1_entity_id':'VARCHAR','candidate_entity_ids':'VARCHAR'}})
""")
print(f"load tables: {time.time()-t0:.1f}s")

con.execute(f"create or replace table t_s1 as select entity_id, country from '{CACHE}/raw_test_s1.parquet'")

q("row counts", "select (select count(*) from mres), (select count(*) from cpairs)")

q("n_matches per s1 (predicted), per country", f"""
  with m as (
    select r.source1_entity_id, s1.country,
      case when r.matched_entity_ids = '' then 0 else len(string_split(r.matched_entity_ids, ',')) end as n_pred
    from mres r join t_s1 s1 on s1.entity_id = r.source1_entity_id
  )
  select country, n_pred, count(*) from m group by 1,2 order by 1,2
""")

q("empty-prediction share per country (sub #6)", f"""
  with m as (
    select r.source1_entity_id, s1.country,
      case when r.matched_entity_ids = '' then 1 else 0 end as is_empty
    from mres r join t_s1 s1 on s1.entity_id = r.source1_entity_id
  )
  select country, avg(is_empty), count(*) from m group by 1 order by 1
""")

q("train singleton rate per country", f"""
  select s1.country, avg(case when g.n_matches=0 then 1.0 else 0 end), count(*)
  from '{CACHE}/train_gt_s1.parquet' g
  join '{CACHE}/raw_train_s1.parquet' s1 on s1.entity_id = g.s1_id
  group by 1 order by 1
""")

q("S2 vs S3 share among PREDICTED matches (sub#6) per country", """
  with m as (
    select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) as mid
    from mres where matched_entity_ids <> ''
  )
  select s1.country,
    sum(case when m.mid like 'S2-%' then 1 else 0 end) as n_s2,
    sum(case when m.mid like 'S3-%' then 1 else 0 end) as n_s3
  from m join t_s1 s1 on s1.entity_id = m.source1_entity_id
  group by 1 order by 1
""")

q("train truth S2 vs S3 share among TRUE matches per country", f"""
  select s1.country,
    sum(case when p.match_id like 'S2-%' then 1 else 0 end) as n_s2,
    sum(case when p.match_id like 'S3-%' then 1 else 0 end) as n_s3
  from '{CACHE}/train_gt_pairs.parquet' p
  join '{CACHE}/raw_train_s1.parquet' s1 on s1.entity_id = p.s1_id
  group by 1 order by 1
""")

q("S1 with zero candidates, per country", """
  select s1.country, count(*)
  from cpairs c join t_s1 s1 on s1.entity_id = c.source1_entity_id
  where c.candidate_entity_ids = ''
  group by 1 order by 1
""")

q(">10 predicted matches: count", """
  select count(*) from (
    select source1_entity_id, len(string_split(matched_entity_ids, ',')) as n
    from mres where matched_entity_ids <> ''
  ) where n > 10
""")
q(">10 predicted matches: examples", """
  select source1_entity_id, matched_entity_ids from mres
  where matched_entity_ids <> '' and len(string_split(matched_entity_ids, ',')) > 10
  limit 5
""")

q("cross-country leakage (matched S2/S3 country != S1 country)", f"""
  with m as (
    select r.source1_entity_id, s1.country as s1_country, unnest(string_split(r.matched_entity_ids, ',')) as mid
    from mres r join t_s1 s1 on s1.entity_id = r.source1_entity_id
    where r.matched_entity_ids <> ''
  ),
  m2 as (
    select m.*, s2.country as match_country
    from m left join (select entity_id, country from read_parquet('{CACHE}/raw_test_s2.parquet')) s2
      on s2.entity_id = m.mid
  ),
  m3 as (
    select m2.*, coalesce(m2.match_country, s3.country) as match_country2
    from m2 left join (select entity_id, country from read_parquet('{CACHE}/raw_test_s3.parquet')) s3
      on s3.entity_id = m2.mid and m2.match_country is null
  )
  select count(*) filter (where match_country2 is distinct from s1_country) as n_mismatch, count(*) as n_total
  from m3
""")

con.close()
