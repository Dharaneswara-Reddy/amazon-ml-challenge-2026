"""Operation frequency tables: distractor-parent pairs vs true pairs (fit group samples from f02)."""
import sys
from f_common import OUT, connect

con = connect()
con.execute(f"CREATE VIEW d AS SELECT *, 'dist' AS kind FROM '{OUT}/f02_ops_dist.parquet'")
con.execute(f"CREATE VIEW t AS SELECT *, 'true' AS kind, NULL::DOUBLE best_s, NULL::BIGINT n_cand FROM '{OUT}/f02_ops_true.parquet'")
con.execute("CREATE VIEW a AS SELECT * FROM d UNION ALL BY NAME SELECT * FROM t")
q = lambda s: con.execute(s).fetchall()
sec = sys.argv[1] if len(sys.argv) > 1 else "all"
if sec in ("all", "ops"):
    print("== name/addr op presence share (dist near = best_s>=1.7)")
    for r in q("""SELECT CASE WHEN kind='dist' AND best_s>=1.7 THEN 'dist_near' WHEN kind='dist' THEN 'dist_far' ELSE 'true' END k,
        count(*) n, avg(name_exact::INT) exact, avg(case_only::INT) caseonly, avg(nonlatin::INT) nonlat,
        avg((n_typo>0)::INT) typo, avg((n_abbrev>0)::INT) abbr, avg((n_trunc>0)::INT) trunc, avg((n_concat>0)::INT) concat,
        avg(n_reorder) reord, avg((n_repl>0)::INT) repl, avg((n_ins>0)::INT) ins, avg((n_del>0)::INT) del,
        avg(addr_empty::INT) aempty, avg((a_repl>0)::INT) a_repl, avg((a_ins>0)::INT) a_ins, avg((a_del>0)::INT) a_del,
        avg((len(rel)>0)::INT) num_changed, avg((l_only>0)::INT) num_ldrop, avg((r_only>0)::INT) num_radd
        FROM a GROUP BY 1 ORDER BY 1"""):
        print(" | ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r))
if sec in ("all", "num"):
    print("== number relation classes (per changed number pair)")
    for r in q("""WITH x AS (SELECT CASE WHEN kind='dist' AND best_s>=1.7 THEN 'dn' WHEN kind='dist' THEN 'df' ELSE 't' END k, unnest(rel) rel FROM a)
        SELECT rel, count(*) FILTER (k='dn'), count(*) FILTER (k='df'), count(*) FILTER (k='t') FROM x GROUP BY 1 ORDER BY 2 DESC"""):
        print(r)
    print("== signed offsets (class offset), top 40: offset, dist_near, true")
    for r in q("""WITH x AS (SELECT CASE WHEN kind='dist' AND best_s>=1.7 THEN 'dn' WHEN kind='dist' THEN 'df' ELSE 't' END k,
                 unnest(rel) rel, unnest(off) o FROM a)
        SELECT o, count(*) FILTER (k='dn') dn, count(*) FILTER (k='t') t FROM x WHERE rel='offset' AND abs(o) <= 40
        GROUP BY 1 ORDER BY 1"""):
        print(r)
    print("== offsets > 40 bucketed")
    for r in q("""WITH x AS (SELECT CASE WHEN kind='dist' AND best_s>=1.7 THEN 'dn' WHEN kind='dist' THEN 'df' ELSE 't' END k,
                 unnest(rel) rel, unnest(off) o FROM a)
        SELECT sign(o)*pow(10, floor(log10(abs(o)))) b, count(*) FILTER (k='dn'), count(*) FILTER (k='t') FROM x WHERE rel='offset' AND abs(o) > 40
        GROUP BY 1 ORDER BY 1"""):
        print(r)
    print("== number of changed pairs per record (dist_near vs true)")
    for r in q("""SELECT len(rel) m, count(*) FILTER (kind='dist' AND best_s>=1.7), count(*) FILTER (kind='true') FROM a WHERE NOT addr_empty GROUP BY 1 ORDER BY 1"""):
        print(r)
    print("== which L number index got the offset (dist near vs true)")
    for r in q("""WITH x AS (SELECT kind, best_s, nl, unnest(rel) rel, unnest(li) li FROM a)
        SELECT nl, li, count(*) FILTER (kind='dist' AND best_s>=1.7), count(*) FILTER (kind='true') FROM x WHERE rel='offset' AND nl<=4 GROUP BY 1,2 ORDER BY 1,2"""):
        print(r)
if sec in ("all", "name"):
    K = "CASE WHEN kind='dist' AND best_s>=1.7 THEN 'dn' WHEN kind='dist' THEN 'df' ELSE 't' END"
    print("== joint name ops (latin R names): repl>0, ins>0, del>0, reorder | dist_near share, true share")
    for r in q(f"""SELECT (n_repl>0)::INT, (n_ins>0)::INT, (n_del>0)::INT, n_reorder, count(*) FILTER (k='dn') / sum(count(*) FILTER (k='dn')) OVER (),
               count(*) FILTER (k='t') / sum(count(*) FILTER (k='t')) OVER () FROM (SELECT {K} k, * FROM a WHERE NOT nonlatin) GROUP BY ALL ORDER BY 5 DESC"""):
        print(r)
    for col in ("n_ins_w", "n_del_w", "n_repl_w", "a_repl_w", "a_ins_w", "a_del_w"):
        print(f"== top {col}: word, dist_near, true")
        for r in q(f"""WITH x AS (SELECT {K} k, unnest({col}) w FROM a) SELECT w, count(*) FILTER (k='dn') dn, count(*) FILTER (k='t') t
                  FROM x GROUP BY 1 ORDER BY greatest(dn, t) DESC LIMIT 45"""):
            print("  ", r)
    print("== replaced-word position in L name (dist_near): first/middle/last share; replacement count per record")
    for r in q(f"""SELECT n_repl, count(*) FILTER (k='dn'), count(*) FILTER (k='t') FROM (SELECT {K} k, * FROM a WHERE NOT nonlatin) GROUP BY 1 ORDER BY 1"""):
        print(r)
    print("== inserted position (0 first,1 mid,2 last) / duplicate-of-existing flag")
    for r in q(f"""WITH x AS (SELECT {K} k, unnest(n_ins_pos) p, unnest(n_ins_dup) dup FROM a) SELECT p, dup, count(*) FILTER (k='dn'), count(*) FILTER (k='t') FROM x GROUP BY ALL ORDER BY ALL"""):
        print(r)
