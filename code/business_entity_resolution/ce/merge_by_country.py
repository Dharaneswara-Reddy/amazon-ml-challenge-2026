"""Merge two prediction outputs by Source-1 country (D-029).

Decisions never cross countries (every candidate pair shares a country, one-to-one assignment and the decision rule act
within a Source-1 entity / right record), so rows can be taken per country from two complete outputs of the same
candidate set: countries listed in COUNTRIES come from DIR_B, all others from DIR_A.
Usage: python merge_by_country.py DIR_A DIR_B OUT_DIR COUNTRY [COUNTRY ...]
Each DIR holds matching_results.tsv and candidate_pairs.tsv; OUT_DIR gets both files (candidate lists must agree).
"""

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
S1 = ROOT / "artifacts" / "cache" / "raw_test_s1.parquet"


def main(a: Path, b: Path, out: Path, countries: list[str]) -> None:
    """Write OUT_DIR/matching_results.tsv (per-country rows from A or B) and the shared candidate_pairs.tsv."""
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    def tsv(p: Path) -> str:
        """read_csv SQL for an output TSV (all text, empty lists kept as '')."""
        return f"read_csv('{p}', sep='\\t', header=true, quote='', all_varchar=true)"

    diff = con.execute(f"""SELECT count(*) FROM (SELECT * FROM {tsv(a / 'candidate_pairs.tsv')}
                           EXCEPT SELECT * FROM {tsv(b / 'candidate_pairs.tsv')})""").fetchone()[0]
    if diff:
        raise SystemExit(f"candidate_pairs.tsv differ in {diff} rows; outputs must come from the same candidates")
    cty = ", ".join(f"'{c}'" for c in countries)
    con.execute(f"""COPY (
        SELECT m.source1_entity_id, coalesce(m.matched_entity_ids, '') AS matched_entity_ids
        FROM (SELECT x.*, s.country FROM {tsv(a / 'matching_results.tsv')} x
              JOIN read_parquet('{S1}') s ON s.entity_id = x.source1_entity_id WHERE s.country NOT IN ({cty})
              UNION ALL
              SELECT y.*, s.country FROM {tsv(b / 'matching_results.tsv')} y
              JOIN read_parquet('{S1}') s ON s.entity_id = y.source1_entity_id WHERE s.country IN ({cty})) m
        ORDER BY m.source1_entity_id
      ) TO '{out / 'matching_results.tsv'}' (HEADER, DELIMITER '\\t', QUOTE '')""")
    con.execute(f"""COPY (SELECT source1_entity_id, coalesce(candidate_entity_ids, '') AS candidate_entity_ids
                         FROM {tsv(a / 'candidate_pairs.tsv')} ORDER BY source1_entity_id)
                   TO '{out / 'candidate_pairs.tsv'}' (HEADER, DELIMITER '\\t', QUOTE '')""")
    print(con.execute(f"SELECT count(*) FROM {tsv(out / 'matching_results.tsv')}").fetchone()[0], "rows written", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
