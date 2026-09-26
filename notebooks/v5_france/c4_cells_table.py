"""C4: persist the France same-core-name pair table (offsets d in {0} and the C1 baseline offsets) with street/city
cells, for label-free per-arm checks: accepted at d = 0 per cell vs est. true, and accepted at baseline offsets
(same name, house number off by an offset no copy/distractor uses => almost surely a false match)."""
from common import *
c = connect()
c.execute(f"""CREATE TABLE L AS SELECT entity_id id, name_core, try_cast(addr_house AS INT) h, addr_street st, addr_city ci
    FROM '{CACHE}/norm_test_s1.parquet' WHERE country_norm = 'france' AND try_cast(addr_house AS INT) IS NOT NULL AND name_core <> ''""")
c.execute(f"""CREATE TABLE R AS SELECT entity_id id, name_core, try_cast(addr_house AS INT) h, addr_street st, addr_city ci
    FROM read_parquet(['{CACHE}/norm_test_s2.parquet', '{CACHE}/norm_test_s3.parquet']) WHERE country_norm = 'france'
    AND try_cast(addr_house AS INT) IS NOT NULL AND name_core <> ''""")
c.execute(f"""COPY (SELECT l.id lid, r.id rid, r.h - l.h d,
      CASE WHEN jaro_winkler_similarity(l.st, r.st) >= 0.9 THEN 'same' WHEN jaro_winkler_similarity(l.st, r.st) >= 0.6 THEN 'mid' ELSE 'diff' END sb,
      (l.ci = r.ci)::INT ce
    FROM L l JOIN R r ON r.name_core = l.name_core AND abs(r.h - l.h) <= 30
    WHERE r.h - l.h = 0 OR r.h - l.h <= -3 OR r.h - l.h IN (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30))
    TO '{TMP}/fr_cells.parquet' (FORMAT parquet)""")
print(c.execute(f"SELECT d = 0, count(*) FROM '{TMP}/fr_cells.parquet' GROUP BY 1").fetchall())
