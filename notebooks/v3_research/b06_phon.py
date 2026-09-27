"""B06: add a phonetic name column `phn` (sorted consonant skeletons of name words) to scratch tables L and R."""
import duckdb, re, time, pyarrow as pa
S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb"); con.execute(f"SET memory_limit='3GB'; SET threads=4; SET temp_directory='{S}/tmp'")
_PH = [("ph", "f"), ("sh", "s"), ("ch", "k"), ("th", "t"), ("kh", "k"), ("gh", "k"), ("bh", "b"), ("dh", "d"), ("jh", "j"),
       ("x", "ks"), ("[gqc]", "k"), ("b", "p"), ("d", "t"), ("[vw]", "f"), ("z", "s"), ("j", "s")]
_PHC = [(re.compile(a), b) for a, b in _PH]
cache = {}
def phw(w):
    """Phonetic consonant skeleton of one word (cached)."""
    r = cache.get(w)
    if r is None:
        x = w
        for a, b in _PHC: x = a.sub(b, x)
        x = x[:1] + re.sub("[aeiouhy]", "", x[1:])
        r = cache[w] = re.sub(r"(.)\1+", r"\1", x)
    return r
t = time.time()
for T in ("L", "R"):
    ids, names = zip(*con.execute(f"SELECT id, name_core FROM {T}").fetchall())
    ph = [" ".join(sorted({phw(w) for w in n.split() if len(w) >= 2})) for n in names]
    tab = pa.table({"id": list(ids), "phn": ph}); con.register("tab", tab)
    con.execute(f"CREATE OR REPLACE TABLE {T}_ph AS SELECT * FROM tab"); con.unregister("tab")
    print(T, time.time() - t, ph[:5])
