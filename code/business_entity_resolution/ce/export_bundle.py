"""Export the cross-encoder bundle (D-028) for GPU fine-tuning/scoring in ce_colab.ipynb.

Pairs are text only ("name | address" of both sides); labels only for FIT-group training pairs (provided train data).
  train.parquet    (ta, tb, label)    all FIT-group pairs with OOF stage-1 p in (LO, HI) + a sample of easy pairs
  score_<g>.parquet (lid, rid, ta, tb) pairs of pred2_<g> with LO < p < HI for g in tune, holdout, test
Usage: python export_bundle.py RUN_DIR OUT_DIR [N_EASY]
"""

import sys
from pathlib import Path

import duckdb

LO, HI = 0.003, 0.997
ROOT = Path(__file__).resolve().parents[3]
C = ROOT / "artifacts" / "cache"


def main(run: Path, out: Path, n_easy: int) -> None:
    """Write train.parquet and score_{tune,holdout,test}.parquet into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=8; SET memory_limit='4GB'")

    def text(split: str, pairs_sql: str, cols: str, dst: Path) -> int:
        """Join raw texts to a pair query and write the selected columns to ``dst``; return rows."""
        con.execute(f"""COPY (
            SELECT {cols} FROM ({pairs_sql}) q
            JOIN read_parquet('{C}/raw_{split}_s1.parquet') l ON l.entity_id = q.lid
            JOIN read_parquet(['{C}/raw_{split}_s2.parquet', '{C}/raw_{split}_s3.parquet']) r ON r.entity_id = q.rid
          ) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)""")
        return con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]

    ta = "l.business_name || ' | ' || coalesce(l.business_address, '') AS ta"
    tb = "r.business_name || ' | ' || coalesce(r.business_address, '') AS tb"
    fit = "md5_number(lid) % 100 < 30"
    train_sql = f"""SELECT lid, rid, label FROM read_parquet('{run}/pred1_fit.parquet') WHERE {fit} AND p > {LO} AND p < {HI}
        UNION ALL
        SELECT * FROM (SELECT lid, rid, label FROM read_parquet('{run}/pred1_fit.parquet')
                       WHERE {fit} AND (p <= {LO} OR p >= {HI}) USING SAMPLE {n_easy} ROWS (reservoir, 42))"""
    print("train", text("train", train_sql, f"{ta}, {tb}, q.label::TINYINT AS label", out / "train.parquet"))
    for g in ("tune", "holdout", "test"):
        split = "test" if g == "test" else "train"
        q = f"SELECT lid, rid FROM read_parquet('{run}/pred2_{g}.parquet') WHERE p > {LO} AND p < {HI}"
        print(g, text(split, q, f"q.lid, q.rid, {ta}, {tb}", out / f"score_{g}.parquet"))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 600_000)
