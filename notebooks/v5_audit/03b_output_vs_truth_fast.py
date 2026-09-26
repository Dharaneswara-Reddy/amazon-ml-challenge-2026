"""Check 3 (fast retry): avoid parsing the 266MB candidate_pairs.tsv text file at all —
use the columnar cand3_test.parquet (same final candidate set, per ber/pipeline.py
model_cand_path) instead. Only matching_results.tsv (97MB) is parsed as text.
"""
import duckdb, os, sys, time
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
    print(f"{label} ({time.time()-t0:.1f}s):", r, flush=True)
    return r

t0 = time.time()
con.execute(f"""
  create or replace table mres as
  select * from read_csv('{SUB}/matching_results.tsv', delim='\t', header=true,
    columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}})
""")
print(f"load mres: {time.time()-t0:.1f}s", flush=True)

con.execute(f"create or replace table t_s1 as select entity_id, country from '{CACHE}/raw_test_s1.parquet'")
con.execute("create or replace table mexp as select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) as mid from mres where matched_entity_ids <> ''")

q("n_matches per s1 (predicted), per country", """
  with m as (
    select r.source1_entity_id, s1.country,
      case when r.matched_entity_ids = '' then 0 else len(string_split(r.matched_entity_ids, ',')) end as n_pred
    from mres r join t_s1 s1 on s1.entity_id = r.source1_entity_id
  )
  select country, n_pred, count(*) from m group by 1,2 order by 1,2
""")

q("empty-prediction share per country (sub #6)", """
  select s1.country, avg(case when r.matched_entity_ids='' then 1 else 0 end), count(*)
  from mres r join t_s1 s1 on s1.entity_id = r.source1_entity_id
  group by 1 order by 1
""")

q("train singleton rate per country", f"""
  select s1.country, avg(case when g.n_matches=0 then 1.0 else 0 end), count(*)
  from '{CACHE}/train_gt_s1.parquet' g
  join '{CACHE}/raw_train_s1.parquet' s1 on s1.entity_id = g.s1_id
  group by 1 order by 1
""")

q("S2 vs S3 share among PREDICTED matches (sub#6) per country", """
  select s1.country,
    sum(case when m.mid like 'S2-%' then 1 else 0 end) as n_s2,
    sum(case when m.mid like 'S3-%' then 1 else 0 end) as n_s3
  from mexp m join t_s1 s1 on s1.entity_id = m.source1_entity_id
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

q(">10 predicted matches: count", """
  select count(*) from (
    select source1_entity_id, count(*) as n from mexp group by 1
  ) where n > 10
""")
q(">10 predicted matches: examples", """
  select source1_entity_id, matched_entity_ids from mres
  where matched_entity_ids <> '' and len(string_split(matched_entity_ids, ',')) > 10
  limit 5
""")

q("cross-country leakage (matched S2/S3 country != S1 country)", f"""
  with m2 as (
    select m.source1_entity_id, s1.country as s1_country, m.mid, s2.country as match_country
    from mexp m
    join t_s1 s1 on s1.entity_id = m.source1_entity_id
    left join (select entity_id, country from read_parquet('{CACHE}/raw_test_s2.parquet')) s2
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

# --- candidate-side checks via cand3_test.parquet (columnar, avoids the 266MB TSV) ---
q("cand3_test: distinct s1 with candidates vs total test s1, per country", f"""
  with c as (select distinct lid from '{CACHE}/cand3_test.parquet')
  select s1.country,
    count(*) filter (where c.lid is not null) as with_cands,
    count(*) filter (where c.lid is null) as zero_cands,
    count(*) as total
  from t_s1 s1 left join c on c.lid = s1.entity_id
  group by 1 order by 1
""")

q("matched ids not present in cand3_test candidates for that s1 (should be 0)", f"""
  with c as (select distinct lid, rid from '{CACHE}/cand3_test.parquet')
  select count(*) as matches_missing_from_candidates
  from mexp m
  left join c on c.lid = m.source1_entity_id and c.rid = m.mid
  where c.lid is null
""")

con.close()
