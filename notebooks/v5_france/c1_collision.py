"""C1: label-free estimate of how many "same core name + same house number + different street" (S1, right) pairs are
true copies vs chance collisions of two different same-named businesses. Join S1 x right on name_core within a
country, count pairs by house-number difference d = h_R - h_L and street similarity; chance collisions are smooth in d,
so the count at non-generator, non-typo offsets (d <= -3, or d even >= 6 and != 21) estimates the chance level at d = 0.
Validated on a 10% train S1 sample with labels (true count at d = 0 vs observed - baseline); then applied to France."""
import json
from common import *
c = connect()
BASE = "(d BETWEEN -30 AND -3) OR d IN (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30)"
def tables(split, lwhere, rwhere):
    """Create L / R tables (id, name_core, int house number, street, city)."""
    c.execute(f"""CREATE OR REPLACE TABLE L AS SELECT entity_id id, name_core, try_cast(addr_house AS INT) h, addr_street st, addr_city ci
        FROM '{CACHE}/norm_{split}_s1.parquet' WHERE {lwhere} AND try_cast(addr_house AS INT) IS NOT NULL AND name_core <> ''""")
    c.execute(f"""CREATE OR REPLACE TABLE R AS SELECT entity_id id, name_core, try_cast(addr_house AS INT) h, addr_street st, addr_city ci
        FROM read_parquet(['{CACHE}/norm_{split}_s2.parquet', '{CACHE}/norm_{split}_s3.parquet'])
        WHERE {rwhere} AND try_cast(addr_house AS INT) IS NOT NULL AND name_core <> ''""")
def pairs(with_label):
    """Pairs joined on name_core with |d| <= 30, street-similarity bucket, city equality, optional label."""
    lab = f", (g.s1_id IS NOT NULL)::INT y" if with_label else ", NULL::INT y"
    gj = f"LEFT JOIN '{CACHE}/train_gt_pairs.parquet' g ON g.s1_id = l.id AND g.match_id = r.id" if with_label else ""
    c.execute(f"""CREATE OR REPLACE TABLE P AS SELECT r.h - l.h d,
          CASE WHEN jaro_winkler_similarity(l.st, r.st) >= 0.9 THEN 'same' WHEN jaro_winkler_similarity(l.st, r.st) >= 0.6 THEN 'mid' ELSE 'diff' END sb,
          (l.ci = r.ci)::INT ce {lab}
        FROM L l JOIN R r ON r.name_core = l.name_core AND abs(r.h - l.h) <= 30 {gj}""")
res = {}
for name, split, lw, rw, lab in (
        ("train10", "train", "hash(entity_id) % 10 = 0", "true", True),
        ("France", "test", "country_norm = 'france'", "country_norm = 'france'", False),
        ("US_test", "test", "country_norm = 'us' AND hash(entity_id) % 4 = 0", "country_norm = 'us'", False)):
    tables(split, lw, rw); pairs(lab)
    rows = c.execute(f"""SELECT sb, ce, count(*) FILTER (WHERE d = 0) n0, sum(y) FILTER (WHERE d = 0) t0,
            count(*) FILTER (WHERE {BASE}) / 41.0 base, count(*) FILTER (WHERE d IN (1, 2)) np12, count(*) FILTER (WHERE d IN (-1, -2)) nm12
        FROM P GROUP BY 1, 2 ORDER BY 1, 2""").fetchall()
    print(f"\n== {name} (pairs {c.execute('SELECT count(*) FROM P').fetchone()[0]:,})")
    print("street  city_eq   n(d=0)  true(d=0)  chance/offset  est_true=n0-base  est_true_share   n(+1,+2) n(-1,-2)")
    out = []
    for sb, ce, n0, t0, base, np12, nm12 in rows:
        est = n0 - base
        print(f"{sb:6s} {ce:7d} {n0:9,d} {t0 if t0 is not None else '-':>10} {base:13.1f} {est:16.1f} {est / max(n0, 1):14.3f} {np12:10,d} {nm12:8,d}")
        out.append(dict(sb=sb, ce=ce, n0=n0, t0=t0, base=base, est_share=est / max(n0, 1)))
    res[name] = out
json.dump(res, open(OUT / "c1_collision.json", "w"), indent=1, default=float)
