"""Check 3c (fast, simple joins only): cross-country leakage + candidate-side checks
for submission #6, using cand3_test.parquet (columnar) instead of the 266MB TSV.
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
    print(f"{label} ({time.time()-t0:.1f}s):", r, flush=True)
    return r

con.execute(f"""
  create or replace table mres as
  select * from read_csv('{SUB}/matching_results.tsv', delim='\t', header=true,
    columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}})
""")
con.execute(f"create or replace table t_s1 as select entity_id, country from '{CACHE}/raw_test_s1.parquet'")
con.execute("create or replace table mexp as select source1_entity_id, unnest(string_split(matched_entity_ids, ',')) as mid from mres where matched_entity_ids is not null")
con.execute(f"""
  create or replace table right_country as
  select entity_id, country from '{CACHE}/raw_test_s2.parquet'
  union all
  select entity_id, country from '{CACHE}/raw_test_s3.parquet'
""")
print("tables loaded", flush=True)

q("cross-country leakage (simple union-based lookup)", """
  select count(*) filter (where rc.country is distinct from s1.country) as n_mismatch, count(*) as n_total
  from mexp m
  join t_s1 s1 on s1.entity_id = m.source1_entity_id
  left join right_country rc on rc.entity_id = m.mid
""")

q("matched ids with no country lookup at all (shouldn't happen if ids exist)", """
  select count(*) from mexp m left join right_country rc on rc.entity_id = m.mid where rc.entity_id is null
""")

q("cand3_test: S1 with candidates vs zero candidates, per country", f"""
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
print("ALL DONE", flush=True)
