"""Label-free error estimate per country for a submission: among ACCEPTED pairs whose only change is one address
number, true copies make symmetric +-1/+-2 typos, so an excess of +1/+2 over -1/-2 (and any generator offset >= 3)
estimates accepted distractors. Also counts clean near-identical pairs (numbers equal, same core name) left unmatched."""
import sys
import duckdb
subs = sys.argv[1:]
c = duckdb.connect(); c.execute("SET memory_limit='5GB'; SET threads=8")
F = "artifacts/cache/feat_v4_test.parquet"
c.execute(f"""CREATE TABLE f AS SELECT f.lid, f.rid, f.r_rank, f.sg_num_changed n_ch, f.sg_first_off AS d_off, f.sg_num_l_only lo,
    f.sg_num_r_only ro, f.sg_num_eq neq, f.sg_off_in_s AS in_s, f.core_ratio cr, s.country
    FROM '{F}' f JOIN 'artifacts/cache/raw_test_s1.parquet' s ON s.entity_id = f.lid""")
for sub in subs:
    c.execute(f"""CREATE OR REPLACE TABLE acc AS SELECT source1_entity_id lid, unnest(string_split(nullif(matched_entity_ids,''), ',')) rid
        FROM read_csv('submissions/{sub}/matching_results.tsv', delim='\t', header=true, all_varchar=true)""")
    print("==", sub)
    rows = c.execute("""
      SELECT f.country,
        count(*) FILTER (WHERE a.rid IS NOT NULL AND n_ch = 1 AND d_off IN (1, 2)) acc_p12,
        count(*) FILTER (WHERE a.rid IS NOT NULL AND n_ch = 1 AND d_off IN (-1, -2)) acc_m12,
        count(*) FILTER (WHERE a.rid IS NOT NULL AND n_ch = 1 AND d_off IN (3,4,5,7,9,11,13,21)) acc_genoff,
        count(*) FILTER (WHERE a.rid IS NOT NULL) accepted,
        count(*) FILTER (WHERE a.rid IS NULL AND n_ch = 0 AND lo = 0 AND ro = 0 AND neq >= 1 AND cr >= 0.95 AND r_rank = 1) rej_clean_best
      FROM f LEFT JOIN acc a USING (lid, rid) GROUP BY 1 ORDER BY 1""").fetchall()
    for r in rows:
        print(f"  {r[0]:7s} accepted={r[4]:>9,}  +1/+2={r[1]:>6,}  -1/-2={r[2]:>6,}  excess(+1/+2 distractors)={r[1]-r[2]:>6,}"
              f"  gen-offset>=3={r[3]:>6,}  rejected clean best pairs={r[5]:>6,}")
