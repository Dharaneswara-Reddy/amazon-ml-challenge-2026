"""Export the unseen-country adaptation bundle for the e5 cross-encoder (D-031), label-free.

The saved e5 model (fine-tuned on train pairs) is adapted to a test-only country with pseudo-labels from a finished
stage-3 run, mixed with labelled train pairs so it keeps the original task:
  adapt.parquet  (ta, tb, label)  country test pairs with stage-3 p >= HI that were matched (label 1) and p <= LO (label 0),
                                  N each, plus N labelled pairs from the original train bundle
  score_test.parquet (lid, rid, ta, tb)  every test pair of the country that already has an e5 score (to re-score)
Usage: python export_fr_adapt.py STAGE3_DIR MATCHING_TSV E5_TRAIN_PARQUET E5_TEST_DIRS(+) OUT_DIR [COUNTRY] [N] [LO] [HI]
"""

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
C = ROOT / "artifacts" / "cache"


def main(s3: Path, matching: Path, e5_train: Path, e5_dirs: str, out: Path, country: str, n: int, lo: float,
         hi: float) -> None:
    """Write adapt.parquet and score_test.parquet into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='6GB'")
    ta = "l.business_name || ' | ' || coalesce(l.business_address, '') AS ta"
    tb = "r.business_name || ' | ' || coalesce(r.business_address, '') AS tb"
    raw = f"""JOIN read_parquet('{C}/raw_test_s1.parquet') l ON l.entity_id = q.lid
              JOIN read_parquet(['{C}/raw_test_s2.parquet', '{C}/raw_test_s3.parquet']) r ON r.entity_id = q.rid"""
    con.execute(f"""CREATE TABLE m AS SELECT source1_entity_id AS lid, unnest(string_split(matched_entity_ids, ',')) AS rid
                    FROM read_csv('{matching}', sep='\\t', header=true, quote='', all_varchar=true)
                    WHERE coalesce(matched_entity_ids, '') <> ''""")
    con.execute(f"""CREATE TABLE p AS SELECT q.lid, q.rid, q.p FROM read_parquet('{s3}/pred2_test.parquet') q
                    JOIN read_parquet('{C}/raw_test_s1.parquet') l ON l.entity_id = q.lid WHERE l.country = '{country}'""")
    pos = f"SELECT p.lid, p.rid FROM p SEMI JOIN m ON m.lid = p.lid AND m.rid = p.rid WHERE p.p >= {hi}"
    neg = f"SELECT lid, rid FROM p WHERE p <= {lo}"
    con.execute(f"""COPY (
        SELECT * FROM (SELECT {ta}, {tb}, 1::TINYINT AS label FROM ({pos}) q {raw} USING SAMPLE {n} ROWS (reservoir, 42))
        UNION ALL SELECT * FROM (SELECT {ta}, {tb}, 0::TINYINT AS label FROM ({neg}) q {raw} USING SAMPLE {n} ROWS (reservoir, 42))
        UNION ALL SELECT * FROM (SELECT ta, tb, label::TINYINT FROM read_parquet('{e5_train}') USING SAMPLE {n} ROWS (reservoir, 42))
      ) TO '{out / 'adapt.parquet'}' (FORMAT parquet, COMPRESSION zstd)""")
    files = ", ".join(f"'{Path(d) / 'ce_test.parquet'}'" for d in e5_dirs.split("+"))
    con.execute(f"""COPY (SELECT q.lid, q.rid, {ta}, {tb} FROM (SELECT DISTINCT e.lid, e.rid FROM read_parquet([{files}]) e
                          JOIN read_parquet('{C}/raw_test_s1.parquet') s ON s.entity_id = e.lid WHERE s.country = '{country}') q {raw}
      ) TO '{out / 'score_test.parquet'}' (FORMAT parquet, COMPRESSION zstd)""")
    for f in ("adapt", "score_test"):
        print(f, con.execute(f"SELECT count(*) FROM read_parquet('{out / (f + '.parquet')}')").fetchone()[0], flush=True)
    print(con.execute(f"SELECT label, count(*) FROM read_parquet('{out / 'adapt.parquet'}') GROUP BY 1 ORDER BY 1").fetchall())


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[1]), Path(a[2]), Path(a[3]), a[4], Path(a[5]), a[6] if len(a) > 6 else "France",
         int(a[7]) if len(a) > 7 else 150_000, float(a[8]) if len(a) > 8 else 0.005, float(a[9]) if len(a) > 9 else 0.995)
