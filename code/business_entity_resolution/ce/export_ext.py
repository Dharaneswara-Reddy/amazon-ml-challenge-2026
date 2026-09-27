"""Export an extended-band scoring bundle (D-029) for a cross-encoder that was already fine-tuned (no train.parquet).

The first bundle scored pairs with IN_LO < p < IN_HI (stage-2 p). This one adds the pairs just outside that band,
OUT_LO < p <= IN_LO or IN_HI <= p < OUT_HI, for the tune and holdout groups (to fit / measure stage 3) and for the
test pairs of the given countries (default France, the unseen country where the cross-encoders help most).
  score_<g>.parquet (lid, rid, ta, tb)
Usage: python export_ext.py RUN_DIR OUT_DIR [IN_LO IN_HI OUT_LO OUT_HI] [COUNTRY ...]
"""

import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]
C = ROOT / "artifacts" / "cache"


def main(run: Path, out: Path, band: tuple[float, float, float, float], countries: list[str]) -> None:
    """Write score_{tune,holdout,test}.parquet with the extended-band pairs into ``out``."""
    in_lo, in_hi, out_lo, out_hi = band
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='6GB'")
    ta = "l.business_name || ' | ' || coalesce(l.business_address, '') AS ta"
    tb = "r.business_name || ' | ' || coalesce(r.business_address, '') AS tb"
    cty = ", ".join(f"'{c}'" for c in countries)
    for g in ("tune", "holdout", "test"):
        split = "test" if g == "test" else "train"
        where = f"((q.p > {out_lo} AND q.p <= {in_lo}) OR (q.p >= {in_hi} AND q.p < {out_hi}))"
        if g == "test":
            where += f" AND l.country IN ({cty})"
        dst = out / f"score_{g}.parquet"
        con.execute(f"""COPY (
            SELECT q.lid, q.rid, {ta}, {tb} FROM read_parquet('{run}/pred2_{g}.parquet') q
            JOIN read_parquet('{C}/raw_{split}_s1.parquet') l ON l.entity_id = q.lid
            JOIN read_parquet(['{C}/raw_{split}_s2.parquet', '{C}/raw_{split}_s3.parquet']) r ON r.entity_id = q.rid
            WHERE {where}
          ) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)""")
        print(g, con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0], flush=True)


if __name__ == "__main__":
    a = sys.argv
    nums = [float(x) for x in a[3:7]] if len(a) >= 7 else [0.01, 0.99, 1e-4, 0.9999]
    main(Path(a[1]), Path(a[2]), tuple(nums), a[7:] if len(a) > 7 else ["France"])
