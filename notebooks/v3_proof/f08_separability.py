"""Separability of each generator signature on real candidate pairs (TUNE group, p2 >= 1e-3, labels known).

For each op / signature: pairs with it, share that are non-matches = P(non-match | op), and the same restricted to
the hard band 0.05 <= p2 <= 0.95 and to pairs the v2 model is confident about (p2 > 0.95).
"""
from f_common import OUT, connect

con = connect()
con.execute(f"CREATE VIEW f AS SELECT * FROM '{OUT}/f05_feat_tune.parquet'")
SIG = {
    "sole offset in S (+1..+21 set)": "sole_off_in_S = 1",
    "sole offset in S, >= +3": "sole_off_in_S = 1 AND first_off >= 3",
    "sole offset +1/+2": "sole_off_in_S = 1 AND first_off IN (1, 2)",
    "offset -1/-2": "off_m12 > 0",
    "number truncated / digit dropped": "num_trunc > 0",
    "name edit log-odds >= 3 (legal swap / vocab insert)": "edit_lo_max >= 3",
    "legal swap (repl_lo_max >= 5)": "repl_lo_max >= 5",
    "vocab insert (ins_lo_max >= 5)": "ins_lo_max >= 5",
    "name edit lo >= 3 AND offset in S": "edit_lo_max >= 3 AND sole_off_in_S = 1",
    "name edit lo >= 3 OR offset in S>=3": "edit_lo_max >= 3 OR (sole_off_in_S = 1 AND first_off >= 3)",
    "no name token op, numbers equal": "n_repl = 0 AND n_ins = 0 AND num_changed = 0",
    "R address empty": "r_addr_empty = 1",
    "name deleted-word lo <= -2 (legal drop)": "del_lo_min <= -2",
}
print("| signature | pairs | P(non-match) | pairs 0.05<=p2<=0.95 | P(nm) there | pairs p2>0.95 | P(nm) there | FP-able: nm with p2>0.5 |")
print("|---|---|---|---|---|---|---|---|")
for name, cond in SIG.items():
    r = con.execute(f"""SELECT count(*), avg(1-label), count(*) FILTER (p BETWEEN 0.05 AND 0.95), avg(1-label) FILTER (p BETWEEN 0.05 AND 0.95),
                           count(*) FILTER (p > 0.95), avg(1-label) FILTER (p > 0.95), count(*) FILTER (p > 0.5 AND label = 0)
                        FROM f WHERE {cond}""").fetchone()
    fmt = lambda x: "-" if x is None else (f"{x:.4f}" if isinstance(x, float) else f"{x:,}")
    print(f"| {name} | " + " | ".join(fmt(x) for x in r) + " |")
print("all band pairs:", con.execute("SELECT count(*), avg(1-label), count(*) FILTER (p>0.5 AND label=0) FROM f").fetchone())
print("first_off histogram, band pairs p2 > 0.3 (label 1 / 0):")
print(con.execute("""SELECT first_off::INT o, count(*) FILTER (label=1), count(*) FILTER (label=0) FROM f WHERE p > 0.3 AND num_changed = 1
                     AND first_off BETWEEN -5 AND 25 GROUP BY 1 ORDER BY 1""").fetchall())
