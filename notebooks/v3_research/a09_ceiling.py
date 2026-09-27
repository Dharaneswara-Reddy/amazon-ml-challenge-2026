"""Ceiling estimate: perfect candidates, current model quality per pattern category; loss by n_true x cause."""
from pathlib import Path
import duckdb
import numpy as np
OUT = Path("/home/toji339/Documents/Amazon_ML_Hackathon/notebooks/v3_research/out")
con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=4")
con.execute(f"CREATE VIEW tc AS SELECT * FROM read_parquet('{OUT}/tp_cat.parquet')")
rates = dict(con.execute("""SELECT cat, sum((status='tp')::INT)::DOUBLE / sum((status<>'block_miss')::INT) FROM tc GROUP BY 1""").fetchall())
print("TP rate among in-candidate true pairs by category:", {k: round(v, 4) for k, v in rates.items()})
con.execute(f"CREATE TEMP TABLE e AS SELECT * FROM read_parquet('{OUT}/hold_ent_counts.parquet')")
con.execute(f"CREATE TEMP TABLE he AS SELECT idx, s1_id FROM read_parquet('{OUT}/hold_ent.parquet')")
case = " ".join(f"WHEN '{k}' THEN {v}" for k, v in rates.items())
rec = con.execute(f"""SELECT he.idx, coalesce(sum(CASE tc.cat {case} END) FILTER (WHERE tc.status='block_miss'), 0) rec
    FROM he LEFT JOIN tc ON tc.s1_id = he.s1_id GROUP BY 1 ORDER BY 1""").fetchnumpy()["rec"]
d = con.execute("SELECT * FROM e ORDER BY idx").fetchnumpy()
tp, fp, fb, fo, fd, n = d["tp"], d["fp"], d["fn_block"], d["fn_o2o"], d["fn_dec"], d["n_true"]
def F(tp, fp, fn):
    den = 1.25 * tp + 0.25 * fn + fp
    f = np.where(den > 0, 1.25 * tp / np.maximum(den, 1e-12), 0.0)
    return np.where(n == 0, (fp == 0).astype(float), f)
print("base", F(tp, fp, fb + fo + fd).mean())
print("perfect candidates, current per-category TP rate (expected):", F(tp + rec, fp, fb - rec + fo + fd).mean())
print("recover 49% of misses at the same rates:", F(tp + 0.49 * rec, fp, fb - 0.49 * rec + fo + fd).mean())
base = F(tp, fp, fb + fo + fd)
for name, m in [("n=1", n == 1), ("n=2", n == 2), ("n=3-4", (n >= 3) & (n <= 4)), ("n>=5", n >= 5)]:
    only_block = F(tp + fb, fp, fo + fd)
    print(name, "loss pts", round((1 - base[m]).sum() / len(n), 5), " of which fixed by blocking oracle",
          round((only_block[m] - base[m]).sum() / len(n), 5), " zero-F entities", int((base[m] == 0).sum()))
