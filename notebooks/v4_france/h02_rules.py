"""Self-label rules measured on labelled TRAIN (holdout+tune) pairs per country, then applied to France test.

Precision = share of rule pairs whose label equals the rule's label; coverage = share of all pairs of that label
(within the candidate file) the rule captures. Also prints how many French test pairs each rule fires on.
"""
from h_common import TMP, connect

NUMEQ = ("coalesce(sg_num_changed,1)=0 AND coalesce(sg_num_l_only,1)=0 AND coalesce(sg_num_r_only,1)=0 "
         "AND coalesce(sg_num_eq,0)>=1")
NAMEQ = "coalesce(sg_n_repl,1)=0 AND coalesce(sg_n_ins,1)=0"
RULES = {  # name -> (label, SQL predicate over the pair table with r_rank, p, p_second)
    "D1 sole offset in S(>=3)": (0, "sg_sole_off_in_s = 1"),
    "D1b D1 & rid-best": (0, "sg_sole_off_in_s = 1 AND r_rank = 1"),
    "D1c D1 & name edit (ins/repl)": (0, "sg_sole_off_in_s = 1 AND sg_n_ins + sg_n_repl >= 1"),
    "D1d D1 & p<0.5": (0, "sg_sole_off_in_s = 1 AND p < 0.5"),
    "D2 rid whose best pair is D1 (all its pairs)": (0, "rid_d1"),
    "E p<0.01": (0, "p < 0.01"),
    "E2 p<0.05 & not rid-best": (0, "p < 0.05 AND r_rank > 1"),
    "T1 nums all equal & no name ins/repl": (1, f"{NUMEQ} AND {NAMEQ}"),
    "T2 T1 & rid-best": (1, f"{NUMEQ} AND {NAMEQ} AND r_rank = 1"),
    "T3 T2 & p>=0.5": (1, f"{NUMEQ} AND {NAMEQ} AND r_rank = 1 AND p >= 0.5"),
    "T4 nums all equal & rid-best & p>=0.5": (1, f"{NUMEQ} AND r_rank = 1 AND p >= 0.5"),
    "T5 T4 & margin>=0.5": (1, f"{NUMEQ} AND r_rank = 1 AND p >= 0.5 AND p - p_second >= 0.5"),
    "T6 T3 & margin>=0.5": (1, f"{NUMEQ} AND {NAMEQ} AND r_rank = 1 AND p >= 0.5 AND p - p_second >= 0.5"),
    "Tp rid-best & p>=0.95 & margin>=0.5 (model only)": (1, "r_rank = 1 AND p >= 0.95 AND p - p_second >= 0.5"),
    "Tn nums all equal & name exact/case-only & rid-best": (1, f"{NUMEQ} AND (sg_name_exact=1 OR sg_case_only=1) AND r_rank = 1"),
}


def view(con, name: str, path: str) -> None:
    """Pair table + rid_d1 flag (rid's rank-1 pair has the sole-offset signature)."""
    con.execute(f"""CREATE OR REPLACE VIEW {name} AS SELECT t.*, coalesce(d.rid_d1, false) AS rid_d1
        FROM read_parquet('{path}') t LEFT JOIN (SELECT rid, true AS rid_d1 FROM read_parquet('{path}')
        WHERE r_rank = 1 AND sg_sole_off_in_s = 1) d USING (rid)""")


if __name__ == "__main__":
    con = connect()
    view(con, "tr", f"{TMP}/pairs_train.parquet")
    view(con, "fr", f"{TMP}/pairs_fr.parquet")
    tot = dict(((c, y), n) for c, y, n in con.execute("SELECT country, label, count(*) FROM tr GROUP BY ALL").fetchall())
    nfr = con.execute("SELECT count(*) FROM fr").fetchone()[0]
    print("train totals", tot, "fr pairs", nfr)
    print("| rule | y | India n | India prec | India cov | US n | US prec | US cov | France n (share) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, (y, pred) in RULES.items():
        r = dict(((c), (n, k)) for c, n, k in con.execute(
            f"SELECT country, count(*), count(*) FILTER (WHERE label = {y}) FROM tr WHERE {pred} GROUP BY 1").fetchall())
        f = con.execute(f"SELECT count(*) FROM fr WHERE {pred}").fetchone()[0]
        cells = []
        for c in ("India", "US"):
            n, k = r.get(c, (0, 0))
            cells += [f"{n:,}", f"{k / max(n, 1):.4f}", f"{k / tot[(c, y)]:.3f}"]
        print(f"| {name} | {y} | " + " | ".join(cells) + f" | {f:,} ({f / nfr:.3f}) |", flush=True)
