"""Sample ~40 French test S1 entities with their chosen match(es) and top rejected candidates, raw text, for
manual failure-mode inspection (normalization of French legal forms, accents, elision, street types, bis/ter,
arrondissement numbers, postal/number formats). Read-only; writes a text report under notebooks/v3_proof/out/.

Sampling: stratified over calibrated best-p (post-shift, production) buckets so we see confident, uncertain, and
low-confidence entities, not just easy ones. Uses the *production* decision output already in output/ (matching
+ candidate TSVs) plus pred2_test.parquet raw scores recalibrated the same way as the pipeline, so displayed
p values match what actually drove the shipped decision.

Run: .venv/bin/python notebooks/v3_proof/03_france_text_sample.py
"""
import json
import random
from pathlib import Path

import duckdb
import numpy as np

from ber.decision import best_per_right, calibrate

RUN = Path("artifacts/runs/20260925T161436_b538130_1242e7cd_v2")
OUT = Path("notebooks/v3_proof/out")
GLOBAL_SHIFT = -0.49344231087999096
SEED = 20260926


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection under the shared-laptop resource limits."""
    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'")
    con.execute("SET threads=2")
    con.execute("PRAGMA temp_directory='artifacts/v3_proof_tmp'")
    return con


def main() -> None:
    """Write a text report of ~40 sampled French S1 entities: chosen match + top rejected candidates, raw text."""
    OUT.mkdir(parents=True, exist_ok=True)
    params = json.loads((RUN / "decision_params.json").read_text())
    con = connect()
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE fr AS
            SELECT dense_rank() OVER (ORDER BY p.lid) - 1 AS li,
                   dense_rank() OVER (ORDER BY p.rid) - 1 AS rc,
                   p.lid AS lid, p.rid AS rid, p.p AS p
            FROM read_parquet('{RUN}/pred2_test.parquet') p
            JOIN read_parquet('artifacts/cache/raw_test_s1.parquet') l ON l.entity_id = p.lid
            WHERE l.country = 'France'"""
    )
    d = con.execute("SELECT li, rc, lid, rid, p FROM fr").fetchnumpy()
    li, rc, p_raw = d["li"].astype(np.int64), d["rc"].astype(np.int64), d["p"].astype(np.float64)
    lid, rid = d["lid"], d["rid"]
    is_best = best_per_right(rc, li, p_raw)
    p_cal = calibrate(params, is_best, p_raw, shift=GLOBAL_SHIFT)  # matches production shipped decision

    # bucket best-per-right pairs by calibrated p to stratify the sample
    rng = random.Random(SEED)
    buckets = {"high (p>=0.8)": [], "mid (0.2-0.8)": [], "low (p<0.2)": []}
    best_idx = np.flatnonzero(is_best)
    for i in best_idx:
        key = "high (p>=0.8)" if p_cal[i] >= 0.8 else ("low (p<0.2)" if p_cal[i] < 0.2 else "mid (0.2-0.8)")
        buckets[key].append(int(li[i]))
    sample_lids: list[int] = []
    for key, lst in buckets.items():
        lst = sorted(set(lst))
        rng.shuffle(lst)
        sample_lids.extend(lst[:14])
    sample_lids = sorted(set(sample_lids))[:42]

    con.register("sample_li", {"li": np.asarray(sample_lids, dtype=np.int64)})
    lid_map = con.execute(
        "SELECT DISTINCT li, lid FROM fr WHERE li IN (SELECT li FROM sample_li)"
    ).fetchall()
    li_to_lid = {int(a): b for a, b in lid_map}

    lines: list[str] = []
    con.execute("CREATE OR REPLACE TEMP TABLE match_out AS SELECT * FROM read_csv('output/matching_results.tsv', delim='\t', header=true, all_varchar=true)")
    for li_val in sample_lids:
        s1id = li_to_lid[li_val]
        s1 = con.execute(
            f"SELECT business_name, business_address FROM read_parquet('artifacts/cache/raw_test_s1.parquet') WHERE entity_id = '{s1id}'"
        ).fetchone()
        matched_ids = con.execute(
            f"SELECT matched_entity_ids FROM match_out WHERE source1_entity_id = '{s1id}'"
        ).fetchone()
        matched_set = set((matched_ids[0] or "").split(",")) if matched_ids and matched_ids[0] else set()
        lines.append(f"\n=== S1 {s1id} | {s1[0]!r} | {s1[1]!r} ===")
        lines.append(f"  chosen matches: {sorted(matched_set) if matched_set else '(empty prediction)'}")
        mask_i = li == li_val
        idxs = np.flatnonzero(mask_i)
        order = idxs[np.argsort(-p_cal[idxs])][:6]
        for j in order:
            rid_j = rid[j]
            row = con.execute(
                f"""SELECT business_name, business_address, country FROM read_parquet('artifacts/cache/raw_test_s2.parquet') WHERE entity_id = '{rid_j}'
                    UNION ALL
                    SELECT business_name, business_address, country FROM read_parquet('artifacts/cache/raw_test_s3.parquet') WHERE entity_id = '{rid_j}'"""
            ).fetchone()
            chosen = "CHOSEN" if rid_j in matched_set else "rejected"
            lines.append(
                f"    [{chosen:8s}] p={p_cal[j]:.3f} raw={p_raw[j]:.3f} is_best={bool(is_best[j])} {rid_j} | {row[0]!r} | {row[1]!r}"
            )

    (OUT / "03_france_text_sample.txt").write_text("\n".join(lines))
    print("wrote", OUT / "03_france_text_sample.txt", "entities:", len(sample_lids))


if __name__ == "__main__":
    main()
