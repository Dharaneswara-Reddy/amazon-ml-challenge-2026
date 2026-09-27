"""B04: similarity stats and a miss taxonomy for the 20k missed vs 20k found sample pairs."""
import duckdb, re, collections, numpy as np
from rapidfuzz.distance import JaroWinkler
S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb", read_only=True); con.execute("SET memory_limit='2GB'; SET threads=4")
cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{S}/pairs_sample.parquet'").fetchall()]
rows = [dict(zip(cols, r)) for r in con.execute(f"SELECT * FROM '{S}/pairs_sample.parquet'").fetchall()]
toks = set()
for r in rows:
    for s in ("l", "r"):
        toks.update("n:" + t for t in r[f"{s}_name_core"].split()); toks.update("a:" + t for t in r[f"{s}_addr_norm"].split())
con.execute("CREATE TEMP TABLE tq(t VARCHAR)"); con.executemany("INSERT INTO tq VALUES (?)", [(t,) for t in toks])
df = {(c, t): d for c, t, d in con.execute("SELECT c, t, dl FROM ldf SEMI JOIN tq USING (t)").fetchall()}
def words(s): return [w for w in s.split() if w != ","]
def jac(a, b):
    a, b = set(a), set(b); return len(a & b) / max(len(a | b), 1)
out = []
for r in rows:
    c = r["l_country_norm"]
    ln, rn = words(r["l_name_core"]), words(r["r_name_core"]); la, ra = words(r["l_addr_norm"]), words(r["r_addr_norm"])
    shared = [("n:" + t) for t in set(ln) & set(rn)] + [("a:" + t) for t in set(la) & set(ra)]
    dls = [df.get((c, t), 0) for t in shared]
    nums_l, nums_r = set(r["l_addr_nums"].split()), set(r["r_addr_nums"].split())
    d = dict(miss=r["miss"], c=c,
        r_nonlatin=r["r_script"] != "latin", l_nonlatin=r["l_script"] != "latin",
        r_domain=bool(re.search(r"\.(com|in|net|org|co)\b|www", (r["r_rname"] or "").lower())) or (len(rn) == 1 and len(ln) >= 2 and rn[0] == "".join(ln)),
        r_addr_short=len(ra) <= 2, l_addr_short=len(la) <= 2,
        name_eq=r["l_name_core"] == r["r_name_core"], name_sorted_eq=sorted(ln) == sorted(rn) and ln != rn,
        jw_name=JaroWinkler.similarity(r["l_name_core"], r["r_name_core"]), jw_skel=JaroWinkler.similarity(r["l_name_skel"], r["r_name_skel"]),
        jac_name=jac(ln, rn), jac_addr=jac(la, ra), share_num=bool(nums_l & nums_r),
        house_eq=bool(set(r["l_addr_house"].split()) & set(r["r_addr_house"].split())),
        n_shared=len(shared), min_dl=min(dls) if dls else 10**9, n_rare100=sum(x <= 100 for x in dls), n_rare1000=sum(x <= 1000 for x in dls),
        name_shared=len(set(ln) & set(rn)))
    out.append((d, r))
M = [d for d, _ in out if d["miss"]]; F = [d for d, _ in out if not d["miss"]]
def mean(L, k): return np.mean([float(x[k]) for x in L])
def med(L, k): return np.median([float(x[k]) for x in L])
print("stat | missed | found")
for k in ["r_nonlatin", "l_nonlatin", "r_domain", "r_addr_short", "l_addr_short", "name_eq", "name_sorted_eq", "share_num", "house_eq"]:
    print(f"{k} | {mean(M,k):.3f} | {mean(F,k):.3f}")
for k in ["jw_name", "jw_skel", "jac_name", "jac_addr", "n_shared", "n_rare100", "n_rare1000", "min_dl"]:
    print(f"{k} mean/median | {mean(M,k):.3f} / {med(M,k):.3f} | {mean(F,k):.3f} / {med(F,k):.3f}")
print("share of pairs with >=1 shared token of left df<=100:", mean([dict(x, z=x['n_rare100']>0) for x in M],'z'), mean([dict(x, z=x['n_rare100']>0) for x in F],'z'))
# exclusive taxonomy of misses (priority order)
def cat(d):
    if d["r_nonlatin"] or d["l_nonlatin"]: return "A non-Latin script on one side (transliterated name)"
    if d["r_domain"]: return "B name is a web domain / concatenated"
    if d["r_addr_short"] or d["l_addr_short"]:
        return "C1 addr empty/short, name jw>=0.9" if d["jw_name"] >= 0.9 else "C2 addr empty/short, name jw<0.9"
    if d["name_eq"] or d["name_sorted_eq"]:
        return "D1 same name (or reordered), shares house no" if d["share_num"] else "D2 same name (or reordered), no shared number"
    if d["jw_name"] >= 0.85:
        return "E1 similar name (jw>=.85, typo/extra word), shares number" if d["share_num"] else "E2 similar name, no shared number"
    if d["jac_addr"] >= 0.4 or d["share_num"]: return "F name differs (jw<.85), address agrees"
    return "G both name and address differ"
cnt = collections.Counter(cat(d) for d in M); cntF = collections.Counter(cat(d) for d in F)
print("\ncategory | misses | found-sample")
for k in sorted(cnt): print(f"{k} | {cnt[k]} ({cnt[k]/len(M):.1%}) | {cntF[k]} ({cntF[k]/len(F):.1%})")
# miss rate per category = misses/(misses+found*scale); scale to population: found sample is from all GT
print("\nwithin misses: min_dl>100 (no shared token passes df cap 100):", mean([dict(x, z=x['n_rare100']==0) for x in M],'z'))
import json
json.dump([(d, {k: r[k] for k in ('l_rname','l_raddr','r_rname','r_raddr')}, cat(d)) for d, r in out if d['miss']], open(f"{S}/miss_cats.json", "w"))
