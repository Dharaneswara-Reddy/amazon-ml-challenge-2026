"""B09: greedy forward selection of prototype passes by union recall gain on the right subset."""
import duckdb, sys
S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb", read_only=True); con.execute("SET memory_limit='3GB'; SET threads=4")
pool = sys.argv[1].split(",")
ngt, nr = con.execute("SELECT count(*), (SELECT count(*) FROM R) FROM gt").fetchone()
def cov(ps):
    u = " UNION ".join(["SELECT lid, rid FROM base"] + [f"SELECT lid, rid FROM cand_{p}" for p in ps])
    rec, n = con.execute(f"WITH u AS ({u}) SELECT (SELECT count(*) FROM gt SEMI JOIN u USING (lid, rid)), (SELECT count(*) FROM u)").fetchone()
    return rec / ngt, n / nr
chosen, cur = [], cov([])
print(f"base recall {cur[0]:.4f} pairs/right {cur[1]:.2f}")
while pool:
    best = max(((p, cov(chosen + [p])) for p in pool), key=lambda x: x[1][0])
    p, (rec, ppr) = best
    print(f"+ {p:4s} recall {rec:.4f} (+{100*(rec-cur[0]):.2f}pp) union pairs/right {ppr:.2f}", flush=True)
    chosen.append(p); pool.remove(p); cur = (rec, ppr)
