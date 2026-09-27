"""Characterize blocking misses on holdout: rid candidate counts, text similarity, examples."""
from pathlib import Path

import duckdb
import numpy as np
from rapidfuzz import fuzz

ROOT = Path("/home/toji339/Documents/Amazon_ML_Hackathon")
OUT = ROOT / "notebooks/v3_research/out"
C = ROOT / "artifacts/cache"
con = duckdb.connect()
con.execute("SET memory_limit='3GB'; SET threads=4")
con.execute(f"CREATE VIEW ts AS SELECT * FROM read_parquet('{OUT}/tp_status.parquet')")
con.execute(f"""CREATE TEMP TABLE r AS SELECT * FROM read_parquet('{C}/raw_train_s2.parquet')
                UNION ALL SELECT * FROM read_parquet('{C}/raw_train_s3.parquet')""")
con.execute(f"""CREATE TEMP TABLE rn AS SELECT * FROM read_parquet('{C}/norm_train_s2.parquet')
                UNION ALL SELECT * FROM read_parquet('{C}/norm_train_s3.parquet')""")
con.execute(f"CREATE VIEW cand AS SELECT lid, rid, block_score FROM read_parquet('{C}/cand_train.parquet')")
con.execute("""CREATE TEMP TABLE bm AS
  SELECT ts.s1_id, ts.match_id, l.business_name ln, l.business_address la, l.country lc,
         r.business_name rn_, r.business_address ra, r.source rs,
         ln_.name_norm lnn, rnn.name_norm rnn, ln_.addr_norm lan, rnn.addr_norm ran, ln_.script lsc, rnn.script rsc,
         ln_.postal lpo, rnn.postal rpo, ln_.addr_nums lnum, rnn.addr_nums rnum
  FROM ts JOIN read_parquet('%s/raw_train_s1.parquet') l ON l.entity_id = ts.s1_id
  JOIN r ON r.entity_id = ts.match_id
  JOIN read_parquet('%s/norm_train_s1.parquet') ln_ ON ln_.entity_id = ts.s1_id
  JOIN rn rnn ON rnn.entity_id = ts.match_id
  WHERE ts.status = 'block_miss'""" % (C, C))
con.execute("""CREATE TEMP TABLE bmc AS SELECT bm.s1_id, bm.match_id, count(c.lid) AS r_ncand,
     max(c.block_score) AS r_top_bs FROM bm LEFT JOIN cand c ON c.rid = bm.match_id GROUP BY 1, 2""")
print("rid candidate count distribution for missed true pairs:",
      con.execute("SELECT r_ncand, count(*) FROM bmc GROUP BY 1 ORDER BY 1").fetchall())
# does the rid's true S1 have other matches that were found? (entity partially found)
print("by source / country / script:")
print(con.execute("SELECT rs, count(*) FROM bm GROUP BY 1").fetchall())
print(con.execute("SELECT lc, count(*) FROM bm GROUP BY 1").fetchall())
print(con.execute("SELECT lsc, rsc, count(*) FROM bm GROUP BY 1,2 ORDER BY 3 DESC").fetchall())
import pyarrow as pa
import pyarrow.parquet as pq
t = con.execute("SELECT * FROM bm ORDER BY s1_id, match_id").arrow()
if hasattr(t, "read_all"):
    t = t.read_all()
d = {k: t.column(k).to_pylist() for k in t.column_names}
ns = np.array([fuzz.token_set_ratio(a or "", b or "") for a, b in zip(d["lnn"], d["rnn"])])
ad = np.array([fuzz.token_set_ratio(a or "", b or "") for a, b in zip(d["lan"], d["ran"])])
rae = np.array([not (x or "") for x in d["ran"]])
lae = np.array([not (x or "") for x in d["lan"]])
print("name token_set sim quantiles", np.percentile(ns, [10, 25, 50, 75, 90]))
print("addr token_set sim quantiles", np.percentile(ad, [10, 25, 50, 75, 90]))
print("r addr empty share", rae.mean(), "l addr empty", lae.mean())
print("name>=90&addr>=90:", ((ns >= 90) & (ad >= 90)).mean(), "name<60:", (ns < 60).mean(),
      "addr<60:", (ad < 60).mean(), "both<60", ((ns < 60) & (ad < 60)).mean())
t = t.append_column("name_sim", pa.array(ns)).append_column("addr_sim", pa.array(ad))
pq.write_table(t, OUT / "block_miss.parquet")
rng = np.random.default_rng(1)
for i in rng.choice(len(ns), 40, replace=False):
    print(f"--- {d['lc'][i]} S{d['rs'][i]} ns={ns[i]:.0f} as={ad[i]:.0f}\n  L: {d['ln'][i]} | {d['la'][i]}\n  R: {d['rn_'][i]} | {d['ra'][i]}\n  Ln: {d['lnn'][i]} | {d['lan'][i]}\n  Rn: {d['rnn'][i]} | {d['ran'][i]}")
