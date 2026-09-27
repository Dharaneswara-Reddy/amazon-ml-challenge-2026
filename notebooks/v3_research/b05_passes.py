"""B05: prototype compound-key blocking passes on the 5% right subset vs all train S1; measure recall gain and cost.

Each pass: key set per record (hashed, country-scoped); keep keys with left df <= cap; score (lid, rid) by summed
idf = ln(1 + NL/dl) over shared keys; keep top-k lids per rid. Output table cand_<name>(lid, rid, s).
Usage: python b05_passes.py name [name ...]
"""
import duckdb, sys, time, json, re
S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/b.duckdb")
con.execute(f"SET memory_limit='3GB'; SET threads=4; SET max_temp_directory_size='20GiB'; SET preserve_insertion_order=false; SET temp_directory='{S}/tmp'")

NUMS = "list_slice(list_reverse(list_distinct(list_filter(string_split(addr_nums, ' '), x -> x <> ''))), 1, 5)"  # 5 largest (addr_nums is sorted ascending)
AW = "list_distinct(list_filter(string_split(addr_norm, ' '), x -> length(x) >= 3 AND NOT regexp_matches(x, '^[0-9]+$')))"
NW = "list_distinct(list_filter(string_split(name_core, ' '), x -> length(x) >= 2))"
STRIP = "replace(name_core, ' ', '')"
AWL = "list_filter(string_split(addr_norm, ' '), x -> length(x) >= 3 AND NOT regexp_matches(x, '^[0-9]+$'))"
PFX = "substr(replace(name_core, ' ', ''), 1, 6)"
PASSES = {
    # phonetic name word x address number / address word
    "pn": (f"SELECT id, c, w || '#' || n AS key FROM (SELECT id, c, unnest({NUMS}) n, string_split(phn, ' ') pw FROM {{T}} WHERE phn <> ''), unnest(pw) u(w)", 50, 5),
    "pa": (f"SELECT id, c, w || '#' || a AS key FROM (SELECT id, c, unnest(string_split(phn, ' ')) w, {AW} aw FROM {{T}} WHERE phn <> ''), unnest(aw) u(a)", 30, 5),
    # address word bigrams and skip-grams (order kept, numbers removed)
    "aa": (f"SELECT id, c, l[i] || '#' || l[j] AS key FROM (SELECT id, c, l, unnest(range(1, len(l))) i FROM (SELECT id, c, {AWL} l FROM {{T}})), unnest([i + 1, i + 2]) u(j) WHERE j <= len(l)", 30, 5),
    # 6-char prefix of the space-stripped name x address number / word (domains, concatenations)
    "px": (f"SELECT id, c, {PFX} || '#' || x AS key FROM (SELECT id, c, name_core, list_concat({NUMS}, {AW}) xs FROM {{T}} WHERE length(replace(name_core, ' ', '')) >= 6), unnest(xs) u(x)", 30, 5),
    # house/any address number x address word
    "hs": (f"SELECT id, c, n || '#' || w AS key FROM (SELECT id, c, unnest({NUMS}) n, {AW} aw FROM {{T}}), unnest(aw) u(w)", 50, 5),
    # name word x address number
    "nn": (f"SELECT id, c, w || '#' || n AS key FROM (SELECT id, c, unnest({NUMS}) n, {NW} nw FROM {{T}}), unnest(nw) u(w)", 50, 5),
    # name word x address word
    "na": (f"SELECT id, c, w || '#' || a AS key FROM (SELECT id, c, unnest({NW}) w, {AW} aw FROM {{T}}), unnest(aw) u(a)", 30, 5),
    # name word pairs (unordered)
    "n2": (f"SELECT id, c, least(a, b) || '#' || greatest(a, b) AS key FROM (SELECT id, c, {NW} nw FROM {{T}}), unnest(nw) u(a), unnest(nw) v(b) WHERE a < b", 30, 5),
    # char 4-grams of the space-stripped core name (domains, concatenations, transliteration drift)
    "s4": (f"SELECT DISTINCT id, c, substr(s, i, 4) AS key FROM (SELECT id, c, s, unnest(range(1, greatest(length(s) - 2, 1))) i FROM (SELECT id, c, {STRIP} s FROM {{T}} WHERE length({STRIP}) >= 4))", 1000, 5),
    # phonetic skeleton words of the name (transliteration)
    "ph": ("SELECT id, c, unnest(string_split(phn, ' ')) AS key FROM {T} WHERE phn <> ''", 300, 5),
    # phonetic name word pairs
    "ph2": ("SELECT id, c, least(a, b) || '#' || greatest(a, b) AS key FROM (SELECT id, c, string_split(phn, ' ') pw FROM {T}), unnest(pw) u(a), unnest(pw) v(b) WHERE a < b", 30, 5),
    # sorted phonetic name (whole-name key, exact join)
    "phx": ("SELECT id, c, phn AS key FROM {T} WHERE phn <> ''", 30, 30),
}

def run(name, cap=None, k=None):
    """Build keys, left df, join with cap, keep top-k; return stats dict."""
    sql, cap0, k0 = PASSES[name]; cap = cap or cap0; k = k or k0
    t = time.time()
    for side in ("L", "R"):
        con.execute(f"CREATE OR REPLACE TEMP TABLE {side}k AS SELECT DISTINCT id, hash(c || '|' || key) AS k FROM ({sql.format(T=f'(SELECT x.*, p.phn FROM {side} x JOIN {side}_ph p USING (id))')})")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE kdf AS SELECT k, count(*) dl, ln(1 + (SELECT count(*) FROM L)::DOUBLE / count(*)) idf
                    FROM Lk GROUP BY k HAVING count(*) <= {cap}""")
    nk = con.execute("SELECT (SELECT count(*) FROM Lk), (SELECT count(*) FROM Rk)").fetchone()
    raw = con.execute("SELECT coalesce(sum(dl),0) FROM Rk JOIN kdf USING (k)").fetchone()[0]
    con.execute(f"""CREATE OR REPLACE TABLE cand_{name} AS SELECT lid, rid, s FROM (
        SELECT l.id AS lid, r.id AS rid, sum(kdf.idf) AS s FROM Rk r JOIN kdf USING (k) JOIN Lk l USING (k) GROUP BY ALL)
        QUALIFY row_number() OVER (PARTITION BY rid ORDER BY s DESC, lid) <= {k}""")
    st = con.execute(f"""SELECT count(*) AS pairs,
          (SELECT count(*) FROM gt g SEMI JOIN cand_{name} c ON c.lid = g.lid AND c.rid = g.rid) AS hit,
          (SELECT count(*) FROM gt g SEMI JOIN cand_{name} c ON c.lid = g.lid AND c.rid = g.rid WHERE NOT g.found) AS gain
        FROM cand_{name}""").fetchone()
    ngt, nmiss, nr = con.execute("SELECT count(*), sum((NOT found)::INT), (SELECT count(*) FROM R) FROM gt").fetchone()
    out = dict(name=name, cap=cap, k=k, Lkeys=nk[0], Rkeys=nk[1], raw_pairs_per_r=round(raw / nr, 1), pairs_per_r=round(st[0] / nr, 2),
               recall_alone=round(st[1] / ngt, 4), gain_pp=round(100 * st[2] / ngt, 3), miss_recovered=round(st[2] / nmiss, 3), secs=round(time.time() - t))
    print(json.dumps(out), flush=True)
    with open(f"{S}/pass_results.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
    return out

if __name__ == "__main__":
    for a in sys.argv[1:]:
        p = a.split(":"); run(p[0], *(int(x) for x in p[1:]))
