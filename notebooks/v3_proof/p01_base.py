"""P01: base tables — integer ids for L (all train S1) and R (all train S2 ∪ S3), phonetic name skeleton `phn`,
v2 candidates (cand_train) and GT mapped to integer ids."""
import re
import pyarrow as pa
from common import *

con = connect()
t0 = time.time()
cols = "entity_id, country_norm AS c, name_core, addr_norm, addr_nums"
con.execute(f"""CREATE OR REPLACE TABLE L AS SELECT (row_number() OVER (ORDER BY entity_id) - 1)::INT AS iid, {cols},
                 (md5_number(entity_id) % 100)::TINYINT AS bkt FROM '{C}/norm_train_s1.parquet'""")
con.execute(f"""CREATE OR REPLACE TABLE R AS SELECT (row_number() OVER (ORDER BY entity_id) - 1)::INT AS iid, {cols}
                 FROM read_parquet(['{C}/norm_train_s2.parquet', '{C}/norm_train_s3.parquet'])""")
log("L,R", con.execute("SELECT (SELECT count(*) FROM L), (SELECT count(*) FROM R)").fetchone(), time.time() - t0)

_PH = [("ph", "f"), ("sh", "s"), ("ch", "k"), ("th", "t"), ("kh", "k"), ("gh", "k"), ("bh", "b"), ("dh", "d"), ("jh", "j"),
       ("x", "ks"), ("[gqc]", "k"), ("b", "p"), ("d", "t"), ("[vw]", "f"), ("z", "s"), ("j", "s")]
_PHC = [(re.compile(a), b) for a, b in _PH]
_V = re.compile("[aeiouhy]"); _RR = re.compile(r"(.)\1+")
cache = {}


def phw(w):
    """Phonetic consonant skeleton of one word (B §implementation rules), cached."""
    r = cache.get(w)
    if r is None:
        x = w
        for a, b in _PHC:
            x = a.sub(b, x)
        x = x[:1] + _V.sub("", x[1:])
        r = cache[w] = _RR.sub(r"\1", x)
    return r


for side in ("L", "R"):
    con.execute(f"DROP TABLE IF EXISTS {side}_ph")
    full = con.execute(f"SELECT iid, name_core FROM {side}").fetch_arrow_table()
    first = True
    for b in full.to_batches(max_chunksize=1_000_000):
        names = b.column(1).to_pylist()
        ph = [" ".join(sorted({phw(w) for w in (n or "").split() if len(w) >= 2})) for n in names]
        tab = pa.table({"iid": b.column(0), "phn": pa.array(ph)})
        con.register("tab", tab)
        con.execute(f"CREATE TABLE {side}_ph AS SELECT * FROM tab" if first else f"INSERT INTO {side}_ph SELECT * FROM tab")
        con.unregister("tab"); first = False
    log(side, "phn done", time.time() - t0)
for side in ("L", "R"):
    con.execute(f"CREATE OR REPLACE TABLE {side}2 AS SELECT x.*, p.phn FROM {side} x JOIN {side}_ph p USING (iid)")
    con.execute(f"DROP TABLE {side}; DROP TABLE {side}_ph; ALTER TABLE {side}2 RENAME TO {side}")

con.execute(f"""CREATE OR REPLACE TABLE gt AS SELECT l.iid AS lid, r.iid AS rid FROM '{C}/train_gt_pairs.parquet' g
               JOIN L l ON l.entity_id = g.s1_id JOIN R r ON r.entity_id = g.match_id""")
con.execute(f"""CREATE OR REPLACE TABLE v2 AS SELECT l.iid AS lid, r.iid AS rid, c.* EXCLUDE (lid, rid)
               FROM '{C}/cand_train.parquet' c JOIN L l ON l.entity_id = c.lid JOIN R r ON r.entity_id = c.rid""")
log("gt, v2", con.execute("SELECT (SELECT count(*) FROM gt), (SELECT count(*) FROM v2)").fetchone(), time.time() - t0)
print(con.execute("SELECT name_core, phn FROM L USING SAMPLE 5 ROWS").fetchall())
