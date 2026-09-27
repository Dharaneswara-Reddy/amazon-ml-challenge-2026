"""B07: recall of base (v2 K=3 candidates) + combinations of prototype passes on the 5% right subset; union size."""
import duckdb, sys, itertools
S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb", read_only=True); con.execute("SET memory_limit='3GB'; SET threads=4")
passes = sys.argv[1].split(",")
ngt, nr = con.execute("SELECT count(*), (SELECT count(*) FROM R) FROM gt").fetchone()
base = con.execute("SELECT avg(found::INT) FROM gt").fetchone()[0]
print(f"base recall {base:.4f}")
def cov(ps):
    """Recall and union pairs per right of base ∪ passes ps."""
    u = " UNION ".join(["SELECT lid, rid FROM base"] + [f"SELECT lid, rid FROM cand_{p}" for p in ps])
    rec, n = con.execute(f"""WITH u AS ({u}) SELECT (SELECT count(*) FROM gt SEMI JOIN u USING (lid, rid)), (SELECT count(*) FROM u)""").fetchone()
    return rec / ngt, n / nr
for r in range(1, len(passes) + 1):
    for ps in itertools.combinations(passes, r):
        if r in (1, len(passes)) or r == len(passes) - 1 or len(passes) <= 4:
            rec, ppr = cov(ps); print(f"{'+'.join(ps):40s} recall {rec:.4f} (+{100*(rec-base):.2f}pp) union pairs/right {ppr:.2f}")
