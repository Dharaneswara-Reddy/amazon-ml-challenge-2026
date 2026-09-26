"""K: label-free comparison of an arm's French accepted set vs the BASE proxy (and vs #6): pairs added / removed,
characterized as in f_decision (near-certain true / false, collision cells, +1/+2 vs -1/-2, generator offsets) plus
pure-generator-word insertions (holding, participations, distribution, international, snc)."""
import sys
from common import *
c = connect()
arms = sys.argv[1:]
c.execute(f"""CREATE TABLE ann AS SELECT lid, rid, max((d = 0 AND sb = 'same')::INT) sure_t, max((d <> 0)::INT) sure_f,
    max((d = 0 AND sb <> 'same')::INT) coll FROM '{TMP}/fr_cells.parquet' GROUP BY 1, 2""")
c.execute(f"""CREATE TABLE a6 AS SELECT source1_entity_id lid, unnest(string_split(nullif(coalesce(matched_entity_ids,''),''), ',')) rid
    FROM read_csv('{SUB6}', delim='\t', header=true, all_varchar=true) s JOIN '{CACHE}/raw_test_s1.parquet' r ON r.entity_id = s.source1_entity_id AND r.country = 'France'""")
def prof(sql):
    """Characterize a set of (lid, rid) pairs."""
    return c.execute(f"""WITH x AS ({sql}) SELECT count(*), coalesce(sum(a.sure_t),0), coalesce(sum(a.sure_f),0), coalesce(sum(a.coll),0),
        count(*) FILTER (WHERE f.sg_num_changed = 1 AND f.sg_first_off IN (1,2)), count(*) FILTER (WHERE f.sg_num_changed = 1 AND f.sg_first_off IN (-1,-2)),
        count(*) FILTER (WHERE f.sg_num_changed = 1 AND f.sg_first_off IN (3,4,5,7,9,11,13,21)),
        count(*) FILTER (WHERE len(list_filter(['holding','participations','distribution','international','snc'],
              w -> list_contains(string_split(rn.name_norm, ' '), w) AND NOT list_contains(string_split(ln.name_norm, ' '), w))) > 0)
        FROM x LEFT JOIN ann a USING (lid, rid) JOIN '{FEAT['test']}' f USING (lid, rid)
        JOIN '{CACHE}/norm_test_s1.parquet' ln ON ln.entity_id = x.lid
        JOIN read_parquet(['{CACHE}/norm_test_s2.parquet', '{CACHE}/norm_test_s3.parquet']) rn ON rn.entity_id = x.rid""").fetchone()
print("columns: n, sureTrue, sureFalse(off-offset), collision, +1/+2, -1/-2, genoff>=3, genword")
for arm, ref in [(a, "BASE") for a in arms] + [("BASE", "6")]:
    A = f"SELECT lid, rid FROM '{TMP}/arm_{arm}/fr_accepted.parquet'"
    R = "SELECT lid, rid FROM a6" if ref == "6" else f"SELECT lid, rid FROM '{TMP}/arm_{ref}/fr_accepted.parquet'"
    print(f"{arm} vs {ref}: ADDED  ", prof(f"({A}) EXCEPT ({R})"), flush=True)
    print(f"{arm} vs {ref}: REMOVED", prof(f"({R}) EXCEPT ({A})"), flush=True)
