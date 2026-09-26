"""Check whether test candidate pairs (pred2_test.parquet) ever cross country borders, and count pairs/entities
per country. Read-only against artifacts/ and code/business_entity_resolution output; writes nothing under
code/business_entity_resolution/ (frozen). Run: .venv/bin/python notebooks/v3_proof/00_crosscountry_check.py
"""
import duckdb

RUN = "artifacts/runs/20260925T161436_b538130_1242e7cd_v2"


def main() -> None:
    """Print per-country candidate-pair counts and confirm lid/rid countries always match."""
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'")
    con.execute("SET threads=2")
    con.execute("PRAGMA temp_directory='artifacts/v3_proof_tmp'")
    con.execute(
        """CREATE TEMP VIEW rcountry AS
           SELECT entity_id, country FROM read_parquet('artifacts/cache/raw_test_s2.parquet')
           UNION ALL
           SELECT entity_id, country FROM read_parquet('artifacts/cache/raw_test_s3.parquet')"""
    )
    rows = con.execute(
        f"""SELECT l.country AS lc, r.country AS rc, count(*)
            FROM read_parquet('{RUN}/pred2_test.parquet') p
            JOIN read_parquet('artifacts/cache/raw_test_s1.parquet') l ON l.entity_id = p.lid
            JOIN rcountry r ON r.entity_id = p.rid
            GROUP BY 1, 2 ORDER BY 3 DESC"""
    ).fetchall()
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
