"""Extract per-pair edit tokens once (signature pair_ops + textdiff align_tokens) so any vocabulary can be learned or
applied later with dict lookups only. Output: artifacts/v4_tmp/tok_{group}.parquet (lid, rid, label, in_group,
country, bkt, self-label signature columns, token lists)."""
from __future__ import annotations

import multiprocessing as mp
import time

import pyarrow as pa
import pyarrow.parquet as pq

from common import FEAT, R, S1, TMP, bucket, connect

SIGCOLS = ["sg_sole_off_in_s", "sg_num_changed", "sg_num_l_only", "sg_num_r_only", "sg_num_eq", "sg_r_addr_empty",
           "sg_n_ins", "sg_n_repl", "sg_n_del", "core_ratio"]
KEYS = ("ins", "del", "repl", "rtgt", "ains", "adel")


def work(rows):
    """Token lists for a chunk of (ln, la, rn, ra, lc, rc) rows."""
    from ber import signature, textdiff

    out = {k: [] for k in (*KEYS, "tdl", "tdr", "name_none", "addr_none")}
    for ln, la, rn, ra, lc, rc in rows:
        d = signature.pair_ops(ln or "", la or "", rn or "", ra or "")
        t = signature.edit_tokens(d)
        for k in KEYS:
            out[k].append(t[k])
        out["name_none"].append(d["name"] is None)
        out["addr_none"].append(d["addr"] is None)
        lt = lc.split() if lc else []
        rt = rc.split() if rc else []
        cl, cr = textdiff.align_tokens(lt, rt)
        out["tdl"].append([w for w in (textdiff._norm_token(x) for x, c in zip(lt, cl) if c == "unexplained") if True])
        out["tdr"].append([textdiff._norm_token(x) for x, c in zip(rt, cr) if c == "unexplained"])
    return out


def run(group: str, where: str, pool) -> None:
    """Write tok_{group}.parquet for rows of feature file ``group`` matching ``where``."""
    con = connect()
    t0 = time.time()
    sig = ", ".join(f"f.{c}" for c in SIGCOLS)
    q = f"""SELECT f.lid, f.rid, f.label, f.in_group, c.country_norm AS country, {bucket('f.lid')} AS bkt, {sig},
                   l.name_norm ln, l.addr_norm la, r.name_norm rn, r.addr_norm ra, l.name_core lc, r.name_core rc
            FROM read_parquet('{FEAT[group]}') f
            JOIN read_parquet('{S1}') c ON c.entity_id = f.lid
            JOIN read_parquet('{S1}') l ON l.entity_id = f.lid
            JOIN read_parquet({[str(p) for p in R]}) r ON r.entity_id = f.rid
            WHERE {where}"""
    rdr = con.execute(q).fetch_record_batch(200_000)
    writer = None
    n = 0
    for b in rdr:
        tab = pa.Table.from_batches([b])
        strs = list(zip(*(tab.column(c).to_pylist() for c in ("ln", "la", "rn", "ra", "lc", "rc"))))
        chunks = [strs[i:i + 20_000] for i in range(0, len(strs), 20_000)]
        parts = pool.map(work, chunks)
        cols = {k: sum((p[k] for p in parts), []) for k in parts[0]}
        base = tab.drop(["ln", "la", "rn", "ra", "lc", "rc"])
        for k, v in cols.items():
            base = base.append_column(k, pa.array(v))
        if writer is None:
            writer = pq.ParquetWriter(TMP / f"tok_{group}.parquet", base.schema, compression="zstd")
        writer.write_table(base)
        n += len(base)
    writer.close()
    con.close()
    print(group, n, round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    with mp.get_context("spawn").Pool(6) as pool:
        run("holdout", "c.country_norm = 'us'", pool)
        run("tune", "TRUE", pool)
        run("fit", "f.in_group = 1", pool)
