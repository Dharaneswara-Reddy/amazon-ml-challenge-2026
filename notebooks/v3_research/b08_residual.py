"""B08: categorize GT pairs still missed after base ∪ given passes (subset), with examples; also check S2<->S3 transitivity."""
import duckdb, sys, collections, random, re
from rapidfuzz.distance import JaroWinkler
C = "artifacts/cache"; S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb", read_only=True); con.execute("SET memory_limit='3GB'; SET threads=4")
passes = sys.argv[1].split(",")
u = " UNION ".join(["SELECT lid, rid FROM base"] + [f"SELECT lid, rid FROM cand_{p}" for p in passes])
f = "name_core, addr_norm, addr_nums, script"
rows = con.execute(f"""WITH u AS ({u}), m AS (SELECT * FROM gt ANTI JOIN u USING (lid, rid))
  SELECT m.lid, m.rid, l.c, l.name_core, l.addr_norm, l.addr_nums, r.name_core, r.addr_norm, r.addr_nums, r.script,
         rl.business_name, rl.business_address, rr.business_name, rr.business_address
  FROM m JOIN L l ON l.id = m.lid JOIN R r ON r.id = m.rid
  JOIN read_parquet('{C}/raw_train_s1.parquet') rl ON rl.entity_id = m.lid
  JOIN read_parquet(['{C}/raw_train_s2.parquet', '{C}/raw_train_s3.parquet']) rr ON rr.entity_id = m.rid""").fetchall()
print("residual misses", len(rows))
def W(s): return [w for w in s.split() if w != ","]
cats = collections.defaultdict(list)
for (lid, rid, c, ln, la, lnum, rn, ra, rnum, rs, lrn, lra, rrn, rra) in rows:
    jw = JaroWinkler.similarity(ln, rn); sn = set(lnum.split()) & set(rnum.split())
    ja = len(set(W(la)) & set(W(ra))) / max(len(set(W(la)) | set(W(ra))), 1)
    dom = bool(re.search(r"\.(com|in|net|org|co)\b|www", (rrn or "").lower())) or (len(W(rn)) == 1 and len(W(ln)) >= 2)
    if rs != "latin": k = "A non-Latin right name"
    elif dom: k = "B domain / concatenated name"
    elif len(W(ra)) <= 2: k = "C addr empty/short" + (" jw>=.9" if jw >= .9 else " jw<.9")
    elif jw >= .85: k = "E similar name" + (", shares number" if sn else ", no shared number")
    elif ja >= .4 or sn: k = "F name differs, address agrees"
    else: k = "G both differ"
    cats[k].append((lrn, lra, rrn, rra, ln, rn))
for k in sorted(cats): print(f"| {k} | {len(cats[k])} ({len(cats[k])/len(rows):.1%}) |")
random.seed(2)
for k in sorted(cats):
    print(f"\n**{k}**")
    for e in random.sample(cats[k], min(3, len(cats[k]))): print(f"- L: `{e[0]} | {e[1]}`  \n  R: `{e[2]} | {e[3]}`  (core: `{e[4]}` / `{e[5]}`)")
