"""H: French name-edit distractors among #6 accepted pairs: right name contains a French generator word
(groupe, holding, developpement, participations, france, distribution, international, snc) that the S1 name lacks,
split by number relation (numbers identical vs other). Reference: same check with the train generator words on
India/US test accepted pairs."""
from common import *
c = connect()
c.execute(f"""CREATE TABLE acc AS SELECT source1_entity_id lid, unnest(string_split(nullif(coalesce(matched_entity_ids,''),''), ',')) rid
    FROM read_csv('{SUB6}', delim='\t', header=true, all_varchar=true)""")
FRW = "['groupe','holding','developpement','participations','france','distribution','international']"
TRW = "['holdings','industries','enterprises','infratech','group','partners']"
c.execute(f"""CREATE TABLE pr AS SELECT f.lid, f.rid, s.country, (a.rid IS NOT NULL) acc,
      (f.sg_num_changed = 0 AND f.sg_num_l_only = 0 AND f.sg_num_r_only = 0 AND f.sg_num_eq >= 1) numeq, f.r_rank,
      string_split(ln.name_norm, ' ') lt, string_split(rn.name_norm, ' ') rt
    FROM '{FEAT['test']}' f JOIN '{CACHE}/raw_test_s1.parquet' s ON s.entity_id = f.lid
    LEFT JOIN acc a USING (lid, rid)
    JOIN '{CACHE}/norm_test_s1.parquet' ln ON ln.entity_id = f.lid
    JOIN read_parquet(['{CACHE}/norm_test_s2.parquet', '{CACHE}/norm_test_s3.parquet']) rn ON rn.entity_id = f.rid
    WHERE f.r_rank <= 2""")
c.execute("SELECT 1"); (lambda: None)() or print(c.execute(f"""SELECT country, acc, numeq, count(*),
      count(*) FILTER (WHERE len(list_filter(CASE WHEN country='France' THEN {FRW} ELSE {TRW} END, w -> list_contains(rt, w) AND NOT list_contains(lt, w))) > 0) genword
    FROM pr GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""").fetchall())
print("France accepted, per inserted word (numeq / not numeq):")
for w in ['groupe','holding','developpement','participations','france','distribution','international','snc','sarl','sas','sci']:
    print(" ", w, c.execute(f"""SELECT count(*) FILTER (WHERE numeq), count(*) FILTER (WHERE NOT numeq),
        count(*) FILTER (WHERE numeq AND NOT acc), count(*) FILTER (WHERE NOT numeq AND NOT acc)
        FROM pr WHERE country='France' AND list_contains(rt, '{w}') AND NOT list_contains(lt, '{w}') AND (acc OR true)""").fetchone(),
        c.execute(f"""SELECT count(*) FROM pr WHERE country='France' AND acc AND list_contains(rt, '{w}') AND NOT list_contains(lt, '{w}') AND numeq""").fetchone())
