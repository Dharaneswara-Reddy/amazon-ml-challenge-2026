"""Which of A's holdout FP / decision-FN categories are fixed by the full signature model, and how many carry the signature."""
from f_common import OUT, connect

A = "/home/toji339/Documents/Amazon_ML_Hackathon/notebooks/v3_research/out"
con = connect()
con.execute(f"CREATE VIEW sf AS SELECT lid, rid FROM '{OUT}/f06_sel_full.parquet'")
con.execute(f"CREATE VIEW hf AS SELECT * FROM '{OUT}/f05_feat_holdout.parquet'")
for name, f, fixed in (("FP", "ex_FP", "NOT EXISTS (SELECT 1 FROM sf WHERE sf.lid=e.lid AND sf.rid=e.rid)"),
                       ("decision FN", "ex_DEC_FN", "EXISTS (SELECT 1 FROM sf WHERE sf.lid=e.lid AND sf.rid=e.rid)")):
    print(f"== {name}: tag | n | fixed by full model | carries distractor signature (offset in S or name-edit lo>=3) | fixed among those")
    for r in con.execute(f"""SELECT e.tag, count(*), count(*) FILTER ({fixed}),
               count(*) FILTER (h.sole_off_in_S = 1 OR h.edit_lo_max >= 3),
               count(*) FILTER ((h.sole_off_in_S = 1 OR h.edit_lo_max >= 3) AND {fixed})
            FROM '{A}/{f}.parquet' e LEFT JOIN hf h ON h.lid = e.lid AND h.rid = e.rid GROUP BY 1 ORDER BY 2 DESC""").fetchall():
        print(" | ".join(map(str, r)))
    print("total", con.execute(f"SELECT count(*), count(*) FILTER ({fixed}) FROM '{A}/{f}.parquet' e").fetchone())
print("new FPs of the full model not in v2's FP list:")
print(con.execute(f"""SELECT count(*) FROM sf JOIN hf USING (lid, rid) WHERE hf.label = 0
                      AND NOT EXISTS (SELECT 1 FROM '{A}/ex_FP.parquet' e WHERE e.lid = sf.lid AND e.rid = sf.rid)""").fetchone())
