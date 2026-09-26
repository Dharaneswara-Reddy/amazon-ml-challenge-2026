"""P02: the five new compound-key passes of B §Q5 (px, hs, n2, na, phx) on ALL train rights vs ALL train S1.

Key set per record (country-scoped, hashed); keys kept when left df <= cap; pair score = sum of ln(1 + NL/df) over
shared keys; top-k lids per right. Runs per right bucket (iid % 8). Output tables cand_<pass>(lid, rid, s).
Usage: python p02_passes.py px hs n2 na phx
"""
from common import *

NUMS = "list_slice(list_reverse(list_distinct(list_filter(string_split(addr_nums, ' '), x -> x <> ''))), 1, 5)"
AW = "list_distinct(list_filter(string_split(addr_norm, ' '), x -> length(x) >= 3 AND NOT regexp_matches(x, '^[0-9]+$')))"
NW = "list_distinct(list_filter(string_split(name_core, ' '), x -> length(x) >= 2))"
PFX = "substr(replace(name_core, ' ', ''), 1, 6)"
KEYS = {
    "px": (f"SELECT iid, c, {PFX} || '#' || x AS key FROM (SELECT iid, c, name_core, list_concat({NUMS}, {AW}) xs FROM {{T}} WHERE length(replace(name_core, ' ', '')) >= 6), unnest(xs) u(x)", 30, 5),
    "hs": (f"SELECT iid, c, n || '#' || w AS key FROM (SELECT iid, c, unnest({NUMS}) n, {AW} aw FROM {{T}}), unnest(aw) u(w)", 50, 5),
    "n2": (f"SELECT iid, c, least(a, b) || '#' || greatest(a, b) AS key FROM (SELECT iid, c, {NW} nw FROM {{T}}), unnest(nw) u(a), unnest(nw) v(b) WHERE a < b", 30, 5),
    "na": (f"SELECT iid, c, w || '#' || a AS key FROM (SELECT iid, c, unnest({NW}) w, {AW} aw FROM {{T}}), unnest(aw) u(a)", 30, 5),
    "phx": ("SELECT iid, c, phn AS key FROM {T} WHERE phn <> ''", 30, 30),
}
NB = 8
con = connect()
NL = con.execute("SELECT count(*) FROM L").fetchone()[0]
for name in sys.argv[1:]:
    sql, cap, k = KEYS[name]
    t = time.time()
    con.execute(f"CREATE OR REPLACE TEMP TABLE Lk AS SELECT DISTINCT iid, hash(c || '|' || key) AS k FROM ({sql.format(T='L')})")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE Lkf AS SELECT l.iid, l.k, d.idf FROM Lk l JOIN (
                      SELECT k, ln(1 + {NL}::DOUBLE / count(*)) AS idf FROM Lk GROUP BY k HAVING count(*) <= {cap}) d USING (k)""")
    con.execute("DROP TABLE Lk")
    con.execute(f"DROP TABLE IF EXISTS cand_{name}")
    for b in range(NB):
        body = f"""SELECT lid, rid, s FROM (
            SELECT l.iid AS lid, r.iid AS rid, sum(l.idf) AS s
            FROM (SELECT DISTINCT iid, hash(c || '|' || key) AS k FROM ({sql.format(T=f'(SELECT * FROM R WHERE iid % {NB} = {b})')})) r
            JOIN Lkf l USING (k) GROUP BY ALL)
            QUALIFY row_number() OVER (PARTITION BY rid ORDER BY s DESC, lid) <= {k}"""
        con.execute(f"CREATE TABLE cand_{name} AS {body}" if b == 0 else f"INSERT INTO cand_{name} {body}")
    n, hit = con.execute(f"SELECT (SELECT count(*) FROM cand_{name}), (SELECT count(*) FROM gt SEMI JOIN cand_{name} c USING (lid, rid))").fetchone()
    ngt = con.execute("SELECT count(*) FROM gt").fetchone()[0]
    log(json.dumps(dict(name=name, cap=cap, k=k, pairs=n, recall_alone=round(hit / ngt, 4), secs=round(time.time() - t))))
