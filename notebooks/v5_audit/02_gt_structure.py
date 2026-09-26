"""Check 2: ground truth structure.

- n_matches distribution, per country and per source (S2-only / S3-only / both)
- any S2/S3 matched to >1 S1 (should be none per problem statement)
- duplicate-group ambiguity: do exact-duplicate-content S2/S3 records ever get
  assigned to DIFFERENT S1 entities in train_gt_pairs?
- singleton (n_matches=0) characteristics
"""
import duckdb, os
os.makedirs("artifacts/v5_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("set memory_limit='2GB'")
con.execute("set threads=3")
con.execute("set temp_directory='artifacts/v5_tmp'")
CACHE = "artifacts/cache"

def q(sql):
    return con.execute(sql).fetchall()

print("=== n_matches distribution ===")
print(q(f"""
  select n_matches, count(*) from '{CACHE}/train_gt_s1.parquet'
  group by 1 order by 1 limit 20
"""))

print("=== total S1, total with n_matches=0 ===")
print(q(f"select count(*), sum(case when n_matches=0 then 1 else 0 end) from '{CACHE}/train_gt_s1.parquet'"))

print("=== does any match_id (S2/S3) map to >1 distinct s1_id? ===")
print(q(f"""
  select count(*) from (
    select match_id, count(distinct s1_id) c from '{CACHE}/train_gt_pairs.parquet'
    group by 1 having count(distinct s1_id) > 1
  )
"""))

print("=== source split per s1: S2-only vs S3-only vs both vs none ===")
print(q(f"""
  with p as (
    select s1_id,
      max(case when match_id like 'S2-%' then 1 else 0 end) as has_s2,
      max(case when match_id like 'S3-%' then 1 else 0 end) as has_s3
    from '{CACHE}/train_gt_pairs.parquet'
    group by 1
  )
  select has_s2, has_s3, count(*) from p group by 1,2 order by 1,2
"""))
print("(s1 with zero matches not in this table; join to gt_s1 to get 'none' count)")
print(q(f"""
  select count(*) from '{CACHE}/train_gt_s1.parquet' g
  where g.n_matches = 0
"""))

print("=== singleton S1 vs matched S1: name/address length, country ===")
print(q(f"""
  select g.n_matches = 0 as is_singleton, s1.country,
    avg(length(s1.business_name)) as avg_name_len,
    avg(length(s1.business_address)) as avg_addr_len,
    count(*)
  from '{CACHE}/train_gt_s1.parquet' g
  join '{CACHE}/raw_train_s1.parquet' s1 on s1.entity_id = g.s1_id
  group by 1,2 order by 1,2
"""))

print("=== exact-duplicate S2 groups: do members ever map to different S1? ===")
print(q(f"""
  with dupg as (
    select business_name, business_address, country, list(entity_id) as ids, count(*) c
    from '{CACHE}/raw_train_s2.parquet'
    group by 1,2,3 having count(*) > 1
  ),
  exploded as (
    select business_name, business_address, country, unnest(ids) as entity_id from dupg
  ),
  joined as (
    select e.business_name, e.business_address, e.country, e.entity_id, p.s1_id
    from exploded e
    left join '{CACHE}/train_gt_pairs.parquet' p on p.match_id = e.entity_id
  )
  select
    count(distinct (business_name,business_address,country)) as groups_with_any_match,
    count(*) filter (where s1_id is not null) as matched_rows
  from joined
"""))
print(q(f"""
  with dupg as (
    select business_name, business_address, country, list(entity_id) as ids, count(*) c
    from '{CACHE}/raw_train_s2.parquet'
    group by 1,2,3 having count(*) > 1
  ),
  exploded as (
    select business_name, business_address, country, unnest(ids) as entity_id from dupg
  ),
  joined as (
    select e.business_name, e.business_address, e.country, e.entity_id, p.s1_id
    from exploded e
    join '{CACHE}/train_gt_pairs.parquet' p on p.match_id = e.entity_id
  ),
  per_group as (
    select business_name, business_address, country, count(distinct s1_id) as n_distinct_s1, count(*) as n_matched
    from joined
    group by 1,2,3
  )
  select n_distinct_s1, count(*) as n_groups, sum(n_matched) as total_matched_rows
  from per_group group by 1 order by 1
"""))

print("=== same for S3 ===")
print(q(f"""
  with dupg as (
    select business_name, business_address, country, list(entity_id) as ids, count(*) c
    from '{CACHE}/raw_train_s3.parquet'
    group by 1,2,3 having count(*) > 1
  ),
  exploded as (
    select business_name, business_address, country, unnest(ids) as entity_id from dupg
  ),
  joined as (
    select e.business_name, e.business_address, e.country, e.entity_id, p.s1_id
    from exploded e
    join '{CACHE}/train_gt_pairs.parquet' p on p.match_id = e.entity_id
  ),
  per_group as (
    select business_name, business_address, country, count(distinct s1_id) as n_distinct_s1, count(*) as n_matched
    from joined
    group by 1,2,3
  )
  select n_distinct_s1, count(*) as n_groups, sum(n_matched) as total_matched_rows
  from per_group group by 1 order by 1
"""))

con.close()
