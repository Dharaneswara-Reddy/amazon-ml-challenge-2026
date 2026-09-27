"""B14: label-free proxy of key usefulness per country on TEST (incl. France) vs TRAIN: share of rights with a selective
hs / px key, pairs per right, and share whose best key-candidate looks like a match (JW name>=.9 and shared number)."""
import duckdb, time
C = "artifacts/cache"; S = "artifacts/v3_research_b"
con = duckdb.connect(f"{S}/t.duckdb"); con.execute(f"SET memory_limit='3GB'; SET threads=4; SET preserve_insertion_order=false; SET temp_directory='{S}/tmp'")
NUMS = "list_slice(list_reverse(list_distinct(list_filter(string_split(addr_nums, ' '), x -> x <> ''))), 1, 5)"
AW = "list_distinct(list_filter(string_split(addr_norm, ' '), x -> length(x) >= 3 AND NOT regexp_matches(x, '^[0-9]+$')))"
PFX = "substr(replace(name_core, ' ', ''), 1, 6)"
KEYS = {"hs": (f"SELECT id, c, n || '#' || w AS key FROM (SELECT id, c, unnest({NUMS}) n, {AW} aw FROM {{T}}), unnest(aw) u(w)", 50),
        "px": (f"SELECT id, c, {PFX} || '#' || x AS key FROM (SELECT id, c, name_core, list_concat({NUMS}, {AW}) xs FROM {{T}} WHERE length(replace(name_core, ' ', '')) >= 6), unnest(xs) u(x)", 30)}
cols = "entity_id AS id, country_norm AS c, name_core, addr_norm, addr_nums"
for split in ("train", "test"):
    t = time.time()
    con.execute(f"CREATE OR REPLACE TABLE L AS SELECT {cols} FROM '{C}/norm_{split}_s1.parquet'")
    con.execute(f"CREATE OR REPLACE TABLE R AS SELECT {cols} FROM read_parquet(['{C}/norm_{split}_s2.parquet','{C}/norm_{split}_s3.parquet']) WHERE hash(entity_id) % 20 = 0")
    for name, (sql, cap) in KEYS.items():
        for side in ("L", "R"):
            con.execute(f"CREATE OR REPLACE TEMP TABLE {side}k AS SELECT DISTINCT id, c, hash(c || '|' || key) AS k FROM ({sql.format(T=side)})")
        con.execute(f"CREATE OR REPLACE TEMP TABLE kdf AS SELECT k, count(*) dl, ln(1 + 2e6 / count(*)) idf FROM Lk GROUP BY k HAVING count(*) <= {cap}")
        con.execute(f"""CREATE OR REPLACE TEMP TABLE cand AS SELECT lid, rid, s FROM (
            SELECT l.id lid, r.id rid, sum(idf) s FROM Rk r JOIN kdf USING (k) JOIN Lk l USING (k) GROUP BY ALL)
            QUALIFY row_number() OVER (PARTITION BY rid ORDER BY s DESC, lid) <= 5""")
        res = con.execute("""WITH top AS (SELECT c.rid, max(jaro_winkler_similarity(l.name_core, r.name_core)) jw,
                   bool_or(len(list_intersect(string_split(l.addr_nums,' '), string_split(r.addr_nums,' '))) > 0) num
                 FROM cand c JOIN L l ON l.id = c.lid JOIN R r ON r.id = c.rid GROUP BY c.rid)
            SELECT R.c, count(*), round(avg((top.rid IS NOT NULL)::INT), 3) AS has_key, round(avg(coalesce(top.jw >= .9 AND top.num, false)::INT), 3) AS good,
                   round((SELECT count(*) FROM cand x JOIN R y ON y.id = x.rid WHERE y.c = R.c)::DOUBLE / count(*), 2) AS pairs_per_r
            FROM R LEFT JOIN top ON top.rid = R.id GROUP BY R.c ORDER BY 1""").fetchall()
        print(split, name, "(country, n_right, share with selective key, share best cand JW>=.9 & shared num, kept pairs/right):", res, round(time.time() - t), flush=True)
