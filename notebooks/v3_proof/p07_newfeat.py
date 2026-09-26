"""P07: arm-3 features appended to a feature directory (fitsub/tune/holdout.parquet -> *_x.parquet).
(a) signed house-number features: d1 = R - L of the first (house) number, d1 in the distractor-offset set LEARNED on
    FIT-SUB labels (offsets in [-30, 30] with non-match:match ratio >= 8 and >= 30 occurrences), parity, relation class
    (equal / truncation / inner-digit deletion / extra digits / other), signed diff of the closest number pair,
    unshared number counts;
(b) crowded names: S1 count with the same (country, name_core) for L and for R's name, number of the right's
    candidates with an identical core name / core JW >= 0.95, number of the entity's candidates with identical name,
    r_addr_empty x l_name_freq;
(c) jw_phn, jw_strip, cut_p are already columns of the v3 feature files (from P03/P04).
Usage: python p07_newfeat.py FEAT_DIR"""
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from common import *

fd = Path(sys.argv[1])
con = connect(fd / "x.duckdb", mem="3GB", threads=8)
t = time.time()
con.execute(f"""CREATE OR REPLACE TABLE N AS SELECT entity_id, country_norm c, name_core, addr_house, addr_nums, addr_norm
               FROM read_parquet(['{C}/norm_train_s1.parquet', '{C}/norm_train_s2.parquet', '{C}/norm_train_s3.parquet'])""")
con.execute("CREATE OR REPLACE TABLE lcnt AS SELECT c, name_core, count(*) AS n FROM N WHERE entity_id LIKE 'S1-%' GROUP BY ALL")
log("N", time.time() - t)


def num(s):
    """Leading integer of a number string, or None."""
    d = ""
    for ch in s or "":
        if ch.isdigit():
            d += ch
        else:
            break
    return int(d[:12]) if d else None


def first(house, nums):
    """House number string, else the first address number string."""
    if house:
        return house.split()[0]
    return nums.split()[0] if nums else ""


def subseq(a, b):
    """True if string a is a subsequence of b."""
    it = iter(b)
    return all(ch in it for ch in a)


def number_feats(lh, ln, rh, rn):
    """Signed number features for one pair of (house, nums) strings."""
    ls, rs = first(lh, ln), first(rh, rn)
    li, ri = num(ls), num(rs)
    d1 = np.nan if li is None or ri is None else float(ri - li)
    if li is None or ri is None:
        rel = 0
    elif ls == rs:
        rel = 1
    elif len(rs) < len(ls) and ls.startswith(rs):
        rel = 2
    elif len(rs) == len(ls) - 1 and subseq(rs, ls):
        rel = 3
    elif len(rs) > len(ls) and subseq(ls, rs):
        rel = 4
    else:
        rel = 5
    la = [x for x in (num(v) for v in (ln or "").split()) if x is not None]
    ra = [x for x in (num(v) for v in (rn or "").split()) if x is not None]
    if la and ra:
        best = min(((r - l) for l in la for r in ra), key=lambda z: (abs(z), -z))
        db = float(best)
    else:
        db = np.nan
    sl, sr = set(la), set(ra)
    return d1, rel, db, len(sl - sr), len(sr - sl)


dset_path = T / "dset.json"
for g in ("fitsub", "tune", "holdout"):
    src, dst = fd / f"{g}.parquet", fd / f"{g}_x.parquet"
    if dst.exists():
        continue
    con.execute(f"""CREATE OR REPLACE TABLE P AS
        WITH f AS (SELECT lid, rid, core_jw, r_addr_empty, l_name_freq, label FROM '{src}'),
        j AS (SELECT f.*, l.c, l.name_core AS lnc, r.name_core AS rnc, l.addr_house lh, l.addr_nums ln, r.addr_house rh, r.addr_nums rn
              FROM f JOIN N l ON l.entity_id = f.lid JOIN N r ON r.entity_id = f.rid)
        SELECT j.*, coalesce(a.n, 0) AS l_core_cnt_c, coalesce(b.n, 0) AS r_core_cnt_c,
          sum((lnc = rnc)::INT) OVER (PARTITION BY rid) AS r_same_name_cands,
          sum((core_jw >= 0.95)::INT) OVER (PARTITION BY rid) AS r_hijw_cands,
          sum((lnc = rnc)::INT) OVER (PARTITION BY lid) AS l_same_name_cands,
          r_addr_empty * l_name_freq AS remp_x_lfreq
        FROM j LEFT JOIN lcnt a ON a.c = j.c AND a.name_core = j.lnc LEFT JOIN lcnt b ON b.c = j.c AND b.name_core = j.rnc""")
    d = con.execute("SELECT lid, rid, lh, ln, rh, rn FROM P").fetchnumpy()
    nf = [number_feats(*x) for x in zip(d["lh"], d["ln"], d["rh"], d["rn"])]
    arr = np.array(nf, dtype=np.float64)
    tab = pa.table({"lid": d["lid"], "rid": d["rid"], "d1": arr[:, 0], "d1_rel": arr[:, 1], "d_best": arr[:, 2],
                    "n_l_unshared": arr[:, 3], "n_r_unshared": arr[:, 4]})
    con.register("nt", tab)
    con.execute("CREATE OR REPLACE TABLE P3 AS SELECT P.*, nt.* EXCLUDE (lid, rid) FROM P JOIN nt USING (lid, rid)")
    con.unregister("nt")
    if g == "fitsub":
        rows = con.execute("""SELECT d1, sum(label) m, sum(1 - label) nm FROM P3 WHERE d1 BETWEEN -30 AND 30 AND d1 <> 0
                              GROUP BY 1 ORDER BY 1""").fetchall()
        dset = [int(a) for a, m, nm in rows if nm >= 30 and nm >= 8 * max(m, 1)]
        dset_path.write_text(json.dumps(dset))
        note(f"- arm-3 distractor offsets learned on FIT-SUB (non-match:match >= 8): {dset}")
    dset = json.loads(dset_path.read_text())
    con.execute(f"""COPY (SELECT f.*, x.d1, x.d1_rel, x.d_best, x.n_l_unshared, x.n_r_unshared,
          (x.d1 IN ({','.join(map(str, dset)) or 'NULL'}))::INT AS d1_in_set, (abs(x.d1) % 2) AS d1_odd,
          (x.d_best IN ({','.join(map(str, dset)) or 'NULL'}))::INT AS dbest_in_set,
          x.l_core_cnt_c, x.r_core_cnt_c, x.r_same_name_cands, x.r_hijw_cands, x.l_same_name_cands, x.remp_x_lfreq
        FROM '{src}' f JOIN P3 x USING (lid, rid)) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)""")
    log(g, "done", time.time() - t)
con.close(); (fd / "x.duckdb").unlink(missing_ok=True)
