"""P03: union of v2 candidates (cand_train, K=3) and the new passes, restricted to 'relevant' right records (those
with at least one union candidate whose S1 is in FIT-SUB, TUNE or HOLDOUT; every candidate of such a right is kept so
rivals are complete). Adds the cheap cut features of B §Q3(g) and the v2 blocking columns for new pairs.
Output: artifacts/v3_proof_tmp/union/b<k>.parquet (int ids)."""
from common import *
from ber.blocking import load_ranker, ranker_sql, addr_words_sql

NB = 8
con = connect()
out = T / "union"; out.mkdir(exist_ok=True)
t = time.time()
con.execute("CREATE OR REPLACE TABLE lsub AS SELECT iid FROM L WHERE bkt < 5 OR (bkt >= 30 AND bkt < 40)")
allp = " UNION ALL ".join(["SELECT lid, rid FROM v2"] + [f"SELECT lid, rid FROM cand_{p}" for p in PASSES])
con.execute(f"CREATE OR REPLACE TABLE rel AS SELECT DISTINCT rid FROM ({allp}) u WHERE lid IN (SELECT iid FROM lsub)")
nrel = con.execute("SELECT count(*) FROM rel").fetchone()[0]
log("relevant rights", nrel, time.time() - t)

V2F = ["in_tok", "in_q3", "in_sn_name", "in_sn_name_addr", "in_sn_addr_name", "in_sn_skel_nums", "in_sn_nums_addr",
       "in_sn_addr_skel", "in_nameonly"]
AWL, AWR = addr_words_sql("l.addr_norm"), addr_words_sql("r.addr_norm")
sl = lambda x: f"string_split({x}, ' ')"
jacc = lambda a, b: f"len(list_intersect({sl(a)}, {sl(b)}))::DOUBLE / greatest(len(list_distinct(list_concat({sl(a)}, {sl(b)}))), 1)"
score = ranker_sql(load_ranker(str(C / "learned/ranker.json")))
sflags = ", ".join(f"max(CASE WHEN p = '{p}' THEN s END) AS s_{p}" for p in PASSES)
for b in range(NB):
    npart = " UNION ALL ".join(f"SELECT lid, rid, '{p}' AS p, s FROM cand_{p} WHERE rid % {NB} = {b}" for p in PASSES)
    q = f"""
    WITH N AS (SELECT lid, rid, {sflags}, count(*)::INT AS n_pass_new FROM ({npart}) GROUP BY lid, rid),
         V AS (SELECT * FROM v2 WHERE rid % {NB} = {b}),
         U AS (SELECT coalesce(V.lid, N.lid) AS lid, coalesce(V.rid, N.rid) AS rid, (V.lid IS NOT NULL) AS in_v2,
                 {', '.join(f'coalesce(V.{f}, false) AS {f}' for f in V2F)},
                 V.tok_score, V.q3_score, coalesce(V.n_pass, 0) AS n_pass, V.block_score AS v2_block_score,
                 {', '.join(f'N.s_{p}' for p in PASSES)}, coalesce(N.n_pass_new, 0) AS n_pass_new
               FROM V FULL OUTER JOIN N ON V.lid = N.lid AND V.rid = N.rid)
    SELECT U.*, (g.lid IS NOT NULL) AS y,
      jaro_winkler_similarity(l.name_core, r.name_core) AS jw_core,
      jaro_winkler_similarity({AWL}, {AWR}) AS jw_addr,
      {jacc('l.addr_norm', 'r.addr_norm')} AS jacc_addr,
      (len(list_intersect({sl('l.addr_nums')}, {sl('r.addr_nums')})) > 0)::INT AS num_match,
      jaro_winkler_similarity(replace(l.name_core, ' ', ''), replace(r.name_core, ' ', '')) AS jw_strip,
      jaro_winkler_similarity(l.phn, r.phn) AS jw_phn,
      {jacc('l.name_core', 'r.name_core')} AS jacc_name,
      len(list_intersect({sl('l.addr_nums')}, {sl('r.addr_nums')})) AS n_num,
      len(list_filter({sl('r.addr_norm')}, x -> x NOT IN ('', ','))) AS r_addr_len,
      len(list_filter({sl('r.addr_nums')}, x -> x <> '')) AS r_nnum,
      l.bkt
    FROM U SEMI JOIN rel ON rel.rid = U.rid
    JOIN L l ON l.iid = U.lid JOIN R r ON r.iid = U.rid
    LEFT JOIN gt g ON g.lid = U.lid AND g.rid = U.rid"""
    con.execute(f"""COPY (SELECT * EXCLUDE (v2_block_score), coalesce(v2_block_score, {score.replace('coalesce(', 'coalesce(')}) AS block_score
                          FROM ({q})) TO '{out}/b{b}.parquet' (FORMAT parquet, COMPRESSION zstd)""")
    log("bucket", b, time.time() - t)
st = con.execute(f"""SELECT count(*), sum(in_v2::INT), sum(y::INT), count(DISTINCT rid) FROM '{out}/b*.parquet'""").fetchone()
log("union rows, v2 rows, true, rights", st)
