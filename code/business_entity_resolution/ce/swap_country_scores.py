"""Replace one country's test cross-encoder scores with an adapted model's scores, quantile-matched (D-031).

The adapted scores only re-order that country's pairs: each pair gets the original score found at its rank, so the
marginal score distribution seen by stage 3 is unchanged (no blanket shift in acceptance). Tune/holdout files are the
union of the original dirs unchanged.
Usage: python swap_country_scores.py ORIG_DIRS(+) ADAPTED_CE_TEST OUT_DIR [COUNTRY]
Writes OUT_DIR/ce_{tune,holdout,test}.parquet (lid, rid, ce).
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
S1 = ROOT / "artifacts" / "cache" / "raw_test_s1.parquet"


def main(orig: str, adapted: Path, out: Path, country: str) -> None:
    """Write the swapped score files into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    dirs = [Path(d) for d in orig.split("+")]

    def files(g: str) -> str:
        """SQL list of the original ce_<g> files."""
        return "[" + ", ".join(f"'{d / f'ce_{g}.parquet'}'" for d in dirs) + "]"

    for g in ("tune", "holdout"):
        con.execute(f"COPY (SELECT lid, rid, ce FROM read_parquet({files(g)})) TO '{out / f'ce_{g}.parquet'}' (FORMAT parquet)")
    con.execute(f"""CREATE TABLE o AS SELECT e.lid, e.rid, e.ce, s.country = '{country}' AS tgt
                    FROM read_parquet({files('test')}) e JOIN read_parquet('{S1}') s ON s.entity_id = e.lid""")
    t = con.execute(f"""SELECT o.lid, o.rid, o.ce AS old, a.ce AS new FROM o JOIN read_parquet('{adapted}') a
                        USING (lid, rid) WHERE o.tgt ORDER BY o.lid, o.rid""").fetchnumpy()
    old, new = t["old"].astype(np.float64), t["new"].astype(np.float64)
    mapped = np.empty_like(old)
    mapped[np.argsort(new, kind="stable")] = np.sort(old)  # the k-th smallest adapted score gets the k-th smallest old one
    swapped = pa.table({"lid": t["lid"], "rid": t["rid"], "ce": pa.array(mapped.astype(np.float32))})
    con.register("sw", swapped)
    con.execute(f"""COPY (SELECT lid, rid, ce FROM o WHERE NOT tgt
                          UNION ALL SELECT lid, rid, ce FROM sw) TO '{out / 'ce_test.parquet'}' (FORMAT parquet)""")
    n_t = con.execute("SELECT count(*) FROM o WHERE tgt").fetchone()[0]
    rho = float(np.corrcoef(np.argsort(np.argsort(old)), np.argsort(np.argsort(new)))[0, 1])
    print(f"{country}: {len(old)} of {n_t} pairs swapped; rank correlation old vs adapted {rho:.4f}", flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], Path(a[2]), Path(a[3]), a[4] if len(a) > 4 else "France")
