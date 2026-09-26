"""P06: v2 feature set (ber.features.write_features, unchanged) on a new candidate file, for FIT-SUB / TUNE / HOLDOUT
with rival context, then append the candidate file's extra columns (new-pass flags/scores, jw_phn, jw_strip, cut_p).
Usage: python p06_features.py CAND_FILE TAG"""
import logging, json
from common import *
from ber.blocking import register_inputs
from ber.features import write_features, name_idf

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cand, tag = sys.argv[1], sys.argv[2]
    out = T / f"feat_{tag}"; out.mkdir(exist_ok=True)
    con = connect(out / "w.duckdb", mem="3GB", threads=8)
    register_inputs(con, str(C / "norm_train_s1.parquet"), [str(C / "norm_train_s2.parquet"), str(C / "norm_train_s3.parquet")])
    con.execute(f"CREATE OR REPLACE VIEW candidates AS SELECT * FROM read_parquet('{cand}')")
    t = time.time()
    idf = name_idf(con)
    log("idf", len(idf), time.time() - t)
    lo = {k: json.loads((C / f"learned/logodds_{k}.json").read_text()) for k in ("fitB", "fit")}
    label = f"SELECT s1_id, match_id FROM read_parquet('{C}/train_gt_pairs.parquet')"
    extra_cols = [c for c in con.execute("DESCRIBE candidates").fetchall()]
    extra = [c[0] for c in extra_cols if c[0] in {f"in_{p}" for p in PASSES} | {f"s_{p}" for p in PASSES} | {"n_pass_new", "in_v2", "jw_strip", "jw_phn", "cut_p", "lrank"}]
    for g, lok in (("fitsub", "fitB"), ("tune", "fit"), ("holdout", "fit")):
        raw, dst = out / f"{g}_raw.parquet", out / f"{g}.parquet"
        if dst.exists():
            continue
        n = write_features(con, raw, label, 100_000, 6, bucket_pred("c.lid", g), True, 16, idf, lo[lok])
        con.execute(f"""COPY (SELECT f.*, {', '.join('c.' + e for e in extra)} FROM '{raw}' f JOIN candidates c USING (lid, rid))
                        TO '{dst}' (FORMAT parquet, COMPRESSION zstd)""")
        raw.unlink()
        log(g, "rows", n, time.time() - t)
    con.close(); (out / "w.duckdb").unlink(missing_ok=True)
