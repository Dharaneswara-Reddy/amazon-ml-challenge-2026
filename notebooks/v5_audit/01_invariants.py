"""Check 1: data invariants across train/test sources.

Row counts vs raw TSV line counts, country spelling variants, ID formats/gaps,
exact-duplicate records within S1/S2/S3, stray whitespace/newlines/control chars
in name/address fields, empty/very short fields.
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

print("=== row counts (parquet) ===")
for t in ["raw_train_s1","raw_train_s2","raw_train_s3","raw_test_s1","raw_test_s2","raw_test_s3"]:
    n = q(f"select count(*) from '{CACHE}/{t}.parquet'")[0][0]
    print(t, n)

print("=== country value spellings ===")
for t in ["raw_train_s1","raw_test_s1"]:
    print(t, q(f"select country, count(*) from '{CACHE}/{t}.parquet' group by 1 order by 2 desc"))

print("=== id format sanity (prefix + numeric suffix) ===")
for t in ["raw_train_s1","raw_train_s2","raw_train_s3","raw_test_s1","raw_test_s2","raw_test_s3"]:
    r = q(f"""
      select
        count(*) filter (where entity_id !~ '^[A-Za-z]+[0-9]+-?[0-9]*$' and entity_id !~ '^[A-Za-z]+-[0-9]+$') as odd_format,
        min(entity_id), max(entity_id), count(distinct entity_id), count(*)
      from '{CACHE}/{t}.parquet'
    """)
    print(t, r)

print("=== whitespace / control-char anomalies in name+address ===")
for t in ["raw_train_s1","raw_train_s2","raw_train_s3","raw_test_s1","raw_test_s2","raw_test_s3"]:
    r = q(f"""
      select
        count(*) filter (where business_name ~ '[\\t\\n\\r]') as name_ctrl,
        count(*) filter (where business_address ~ '[\\t\\n\\r]') as addr_ctrl,
        count(*) filter (where business_name <> trim(business_name)) as name_untrimmed,
        count(*) filter (where business_address <> trim(business_address)) as addr_untrimmed,
        count(*) filter (where business_name ~ '  ') as name_doublespace,
        count(*) filter (where business_name = '') as name_empty,
        count(*) filter (where business_address = '') as addr_empty,
        count(*) filter (where length(business_name) <= 2 and business_name <> '') as name_vshort,
        count(*) filter (where contains(business_name, chr(65533)) or contains(business_address, chr(65533))) as mojibake_replacement_char
      from '{CACHE}/{t}.parquet'
    """)
    print(t, r)

print("=== exact duplicate records WITHIN each source (same name+address+country) ===")
for t in ["raw_train_s1","raw_train_s2","raw_train_s3","raw_test_s1","raw_test_s2","raw_test_s3"]:
    r = q(f"""
      with d as (
        select business_name, business_address, country, count(*) c
        from '{CACHE}/{t}.parquet'
        group by 1,2,3
        having count(*) > 1
      )
      select count(*) as dup_groups, sum(c) as dup_rows, max(c) as max_group from d
    """)
    print(t, r)

con.close()
