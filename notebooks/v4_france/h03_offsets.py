"""Offset histogram of pairs with exactly one changed number (offset class), by country and p band, rid-best pairs.

Checks that French high-p pairs with an offset in S look like distractors (uniform over S, off-S offsets rare).
"""
from h_common import TMP, connect

con = connect()
q = """SELECT country, CASE WHEN p >= 0.69 THEN 'p>=.69' WHEN p >= 0.2 THEN '.2-.69' ELSE '<.2' END band,
  sg_first_off::INT AS o, count(*) n{lab}
  FROM read_parquet('{path}') WHERE r_rank = 1 AND sg_num_changed = 1 AND sg_first_off BETWEEN -3 AND 22
  AND sg_num_trunc = 0 AND sg_num_add_digit = 0 GROUP BY ALL"""
rows = con.execute(q.format(path=f"{TMP}/pairs_train.parquet", lab=", count(*) FILTER (WHERE label=1) npos")).fetchall()
rows += [(*r, None) for r in con.execute(q.format(path=f"{TMP}/pairs_fr.parquet", lab="")).fetchall()]
offs = list(range(-3, 23))
from collections import defaultdict
tab = defaultdict(dict)
for c, b, o, n, npos in rows:
    tab[(c, b)][o] = (n, npos)
print("| country | p band | " + " | ".join(f"{o:+d}" for o in offs) + " |")
for key in sorted(tab):
    cells = []
    for o in offs:
        n, npos = tab[key].get(o, (0, None))
        cells.append(f"{n}" + (f"/{npos}" if npos is not None else ""))
    print(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
n = con.execute(f"""SELECT count(*) FILTER (WHERE sg_sole_off_in_s=1 AND r_rank=1 AND p>=0.69),
   count(*) FILTER (WHERE r_rank=1 AND p>=0.69), count(DISTINCT lid) FROM read_parquet('{TMP}/pairs_fr.parquet')""").fetchone()
print("France rid-best p>=0.69 with sole offset in S:", n)
for c in ("India", "US"):
    print(c, con.execute(f"""SELECT count(*) FILTER (WHERE sg_sole_off_in_s=1 AND r_rank=1 AND p>=0.69),
       count(*) FILTER (WHERE sg_sole_off_in_s=1 AND r_rank=1 AND p>=0.69 AND label=0),
       count(*) FILTER (WHERE r_rank=1 AND p>=0.69), count(DISTINCT lid) FROM read_parquet('{TMP}/pairs_train.parquet') WHERE country='{c}'""").fetchone())
