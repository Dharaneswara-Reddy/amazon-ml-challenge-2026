"""Lightweight, memory-bounded profile of the organizer dataset (DuckDB only; prints JSON)."""

import json
import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
TMP = ROOT / "artifacts" / "tmp"


def src(split: str, n: int) -> str:
    """Return a DuckDB read_csv expression for one source file."""
    p = DATA / split / f"{split}_source{n}.tsv"
    return (
        f"read_csv('{p}', delim='\\t', header=true, quote='', escape='', all_varchar=true, "
        "columns={'entity_id':'VARCHAR','business_name':'VARCHAR','business_address':'VARCHAR','country':'VARCHAR'})"
    )


def gt() -> str:
    """Return a DuckDB read_csv expression for the train ground truth."""
    p = DATA / "train" / "train_ground_truth.tsv"
    return (
        f"read_csv('{p}', delim='\\t', header=true, quote='', escape='', all_varchar=true, "
        "columns={'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'})"
    )


def q(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    """Run SQL and return rows as dicts."""
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


def file_profile(con: duckdb.DuckDBPyConnection, split: str, n: int) -> dict:
    """Row counts, emptiness, lengths, duplicates, country mix and postal presence for one file."""
    s = src(split, n)
    base = q(
        con,
        f"""
        SELECT count(*) rows, count(DISTINCT entity_id) distinct_ids,
          avg((coalesce(business_name,'')='')::INT) name_empty,
          avg((coalesce(business_address,'')='')::INT) addr_empty,
          quantile_cont(length(business_name),[0.5,0.95,0.999]) name_len,
          quantile_cont(length(business_address),[0.5,0.95,0.999]) addr_len,
          max(length(business_name)) name_len_max,
          avg(regexp_matches(business_name,'[\\x{{0900}}-\\x{{097F}}]')::INT) name_devanagari,
          avg(regexp_matches(business_address,'[\\x{{0900}}-\\x{{097F}}]')::INT) addr_devanagari,
          avg(regexp_matches(business_name,'[^\\x00-\\x7F]')::INT) name_non_ascii,
          avg((NOT regexp_matches(entity_id, '^S{n}-[0-9]+$'))::INT) bad_id_rate
        FROM {s}""",
    )[0]
    base["countries"] = q(
        con,
        f"""SELECT country, count(*) n,
              avg(regexp_matches(business_address,'(^|[^0-9])[0-9]{{6}}([^0-9]|$)')::INT) has6,
              avg(regexp_matches(business_address,'(^|[^0-9])[0-9]{{5}}([^0-9]|$)')::INT) has5
            FROM {s} GROUP BY 1 ORDER BY 2 DESC""",
    )
    base["dup_full_rows"] = q(
        con,
        f"""SELECT count(*) - count(DISTINCT (lower(business_name), lower(business_address), country)) d
            FROM {s}""",
    )[0]["d"]
    base["samples"] = q(con, f"SELECT business_name, business_address, country FROM {s} USING SAMPLE 8 ROWS")
    return base


def gt_profile(con: duckdb.DuckDBPyConnection) -> dict:
    """Match multiplicity, singleton rate, and whether right-side records map to several S1 entities."""
    con.execute(
        f"""CREATE TABLE IF NOT EXISTS gtp AS
            SELECT source1_entity_id s1, trim(m) r
            FROM {gt()}, unnest(string_split(coalesce(matched_entity_ids,''), ',')) t(m)
            WHERE trim(m) <> ''"""
    )
    out = q(
        con,
        f"""SELECT count(*) n_s1,
              avg((coalesce(matched_entity_ids,'')='')::INT) singleton_rate
            FROM {gt()}""",
    )[0]
    out["pairs"] = q(con, "SELECT count(*) n, count(DISTINCT r) distinct_right FROM gtp")[0]
    out["right_in_multiple_s1"] = q(
        con, "SELECT count(*) n FROM (SELECT r FROM gtp GROUP BY r HAVING count(DISTINCT s1) > 1)"
    )[0]["n"]
    out["multiplicity"] = q(
        con,
        """SELECT k, count(*) n FROM (
             SELECT s1, count(*) k FROM gtp GROUP BY s1) GROUP BY k ORDER BY k LIMIT 30""",
    )
    out["per_source_counts"] = q(
        con,
        """SELECT n2, n3, count(*) n FROM (
             SELECT s1, sum((r LIKE 'S2-%')::INT) n2, sum((r LIKE 'S3-%')::INT) n3
             FROM gtp GROUP BY s1) GROUP BY 1,2 ORDER BY 3 DESC LIMIT 15""",
    )
    return out


def coverage_and_country(con: duckdb.DuckDBPyConnection) -> dict:
    """Share of S2/S3 records matched to any S1, cross-country GT pairs, exact-name agreement."""
    out = {}
    for n in (2, 3):
        out[f"s{n}_matched_share"] = q(
            con,
            f"""SELECT avg((g.r IS NOT NULL)::INT) matched_share
                FROM {src('train', n)} s LEFT JOIN (SELECT DISTINCT r FROM gtp) g ON s.entity_id = g.r""",
        )[0]["matched_share"]
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE pair_sample AS
            SELECT * FROM gtp USING SAMPLE 200000 ROWS (reservoir, 42)"""
    )
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE right_all AS
            SELECT * FROM {src('train', 2)} WHERE entity_id IN (SELECT r FROM pair_sample)
            UNION ALL
            SELECT * FROM {src('train', 3)} WHERE entity_id IN (SELECT r FROM pair_sample)"""
    )
    out["pair_sample_stats"] = q(
        con,
        f"""SELECT count(*) n,
              avg((a.country = b.country)::INT) same_country,
              avg((lower(a.business_name) = lower(b.business_name))::INT) exact_name_ci,
              avg(jaro_winkler_similarity(lower(a.business_name), lower(b.business_name))) jw_name,
              avg(jaro_winkler_similarity(lower(a.business_address), lower(b.business_address))) jw_addr
            FROM pair_sample p
            JOIN {src('train', 1)} a ON a.entity_id = p.s1
            JOIN right_all b ON b.entity_id = p.r""",
    )[0]
    out["pair_examples"] = q(
        con,
        f"""SELECT a.business_name s1_name, b.business_name r_name,
                   a.business_address s1_addr, b.business_address r_addr, p.r
            FROM (SELECT * FROM pair_sample USING SAMPLE 25 ROWS (reservoir, 7)) p
            JOIN {src('train', 1)} a ON a.entity_id = p.s1
            JOIN right_all b ON b.entity_id = p.r""",
    )
    return out


def main() -> None:
    """Run all profile sections and dump JSON to artifacts/profile/profile.json."""
    TMP.mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "profile").mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "artifacts" / "profile" / "profile.duckdb"))
    con.execute(f"SET memory_limit='700MB'; SET threads=3; SET temp_directory='{TMP}'")
    con.execute("SET preserve_insertion_order=false")
    result: dict = {"files": {}}
    sections = sys.argv[1:] or ["files", "gt", "coverage"]
    t0 = time.time()
    if "files" in sections:
        for split in ("train", "test"):
            for n in (1, 2, 3):
                result["files"][f"{split}_s{n}"] = file_profile(con, split, n)
                sys.stderr.write(f"{split}_s{n} done {time.time() - t0:.0f}s\n")
    if "gt" in sections or "coverage" in sections:
        result["gt"] = gt_profile(con)
        sys.stderr.write(f"gt done {time.time() - t0:.0f}s\n")
    if "coverage" in sections:
        result["coverage"] = coverage_and_country(con)
        sys.stderr.write(f"coverage done {time.time() - t0:.0f}s\n")
    out = ROOT / "artifacts" / "profile" / f"profile_{'_'.join(sections)}.json"
    out.write_text(json.dumps(result, indent=1, default=str, ensure_ascii=False))
    sys.stderr.write(f"wrote {out} in {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
