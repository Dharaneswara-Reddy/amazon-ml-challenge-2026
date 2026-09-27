"""Build compact pair tables (rule inputs) for train holdout+tune (labelled) and France test (unlabelled).

Output: artifacts/v4_tmp/pairs_train.parquet, pairs_fr.parquet with v3 stage-2 p, per-right rank/margin,
per-left rank, and the language-independent signature columns from the v3 feature files.
"""
import time

from h_common import C, RUN, TMP, connect

SG = ["sg_sole_off_in_s", "sg_off_in_s", "sg_num_changed", "sg_num_nl", "sg_num_nr", "sg_num_eq", "sg_n_repl",
      "sg_n_ins", "sg_n_del", "sg_n_typo", "sg_n_abbrev", "sg_n_trunc", "sg_n_reorder", "sg_name_exact",
      "sg_case_only", "sg_r_addr_empty", "sg_off_p12", "sg_off_m12", "sg_off_other", "sg_num_trunc",
      "sg_num_add_digit", "sg_first_off", "sg_edit_lo_max", "sg_ins_lo_max", "sg_repl_lo_max", "sg_num_l_only",
      "sg_num_r_only", "cut_p", "jw_core", "jw_addr", "jw_strip", "jw_phn", "core_ratio", "addr_ratio"]


def build(con, src_sql: str, dst: str) -> None:
    """Add per-right best / margin and per-left rank to a (lid, rid, p, label, country, SG...) source and write it."""
    con.execute(f"""COPY (
        WITH s AS ({src_sql}),
        r AS (SELECT *, row_number() OVER (PARTITION BY rid ORDER BY p DESC, lid) AS r_rank,
                     count(*) OVER (PARTITION BY rid) AS r_n,
                     row_number() OVER (PARTITION BY lid ORDER BY p DESC, rid) AS l_rank FROM s),
        m AS (SELECT rid, max(p) FILTER (WHERE r_rank = 2) AS p_second FROM r GROUP BY rid)
        SELECT r.*, coalesce(m.p_second, 0) AS p_second FROM r JOIN m USING (rid)
      ) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)""")


if __name__ == "__main__":
    con = connect()
    cols = ", ".join(f"f.{c}" for c in SG)
    t = time.time()
    parts = []
    for g in ("holdout", "tune"):
        parts.append(f"""SELECT f.lid, f.rid, f.label::INT AS label, p.p, s.country, '{g}' AS grp, {cols}
            FROM read_parquet('{C}/feat_v3_train_{g}.parquet') f
            JOIN read_parquet('{RUN}/pred2_{g}.parquet') p USING (lid, rid)
            JOIN read_parquet('{C}/raw_train_s1.parquet') s ON s.entity_id = f.lid""")
    build(con, " UNION ALL ".join(parts), f"{TMP}/pairs_train.parquet")
    print("train done", time.time() - t, flush=True)
    t = time.time()
    build(con, f"""SELECT f.lid, f.rid, NULL::INT AS label, p.p, s.country, 'test' AS grp, {cols}
            FROM read_parquet('{C}/feat_v3_test.parquet') f
            JOIN read_parquet('{C}/raw_test_s1.parquet') s ON s.entity_id = f.lid AND s.country = 'France'
            JOIN read_parquet('{RUN}/pred2_test.parquet') p USING (lid, rid)""", f"{TMP}/pairs_fr.parquet")
    print("fr done", time.time() - t, flush=True)
    for f in ("pairs_train", "pairs_fr"):
        print(f, con.execute(f"SELECT grp, country, count(*), count(DISTINCT lid), count(DISTINCT rid) FROM read_parquet('{TMP}/{f}.parquet') GROUP BY ALL").fetchall())
