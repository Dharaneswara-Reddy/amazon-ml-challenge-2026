"""Loss decomposition of a run's holdout predictions: where the macro-F0.5 points go (oracle fix per category)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "code/business_entity_resolution/src")
from ber.config import load_config
from ber.duck import connect
from ber.pipeline import _group_arrays, _fetch_numpy
from ber.decision import decide, per_entity_f05
from ber.split import group_where
from ber import io

run = Path(sys.argv[1])
cfg = load_config()
con = connect(cfg)
params = json.loads((run / "decision_params.json").read_text())
a = _group_arrays(con, cfg, run / "pred2_holdout.parquet", "holdout")
li, rid, p, lab, nt = a["lid_idx"], a["rid"], a["p"], a["label"], a["n_true"]
sel = decide(params, li, rid, p)
n = len(nt)
f = per_entity_f05(li, sel, lab, nt)
base = f.mean()
in_cand = np.bincount(li, weights=lab.astype(float), minlength=n)          # true pairs present after rival drop
# true pairs of holdout entities in the full pred file (before rival drop)
s1 = io.raw_parquet_path(cfg, "train", 1)
tot_in_file = _fetch_numpy(con, f"""SELECT count(*) AS c FROM read_parquet('{run / "pred2_holdout.parquet"}')
    WHERE label = 1 AND {group_where('lid', 'holdout')}""")["c"][0]
tp = (sel & lab).sum(); fp = (sel & ~lab).sum(); fn_dec = (~sel & lab).sum()
blk_miss = nt.sum() - tot_in_file; o2o_loss = tot_in_file - lab.sum()
def F(selected, label, ntrue):
    return per_entity_f05(li, selected, label, ntrue).mean()
fp_single = (sel & ~lab & (nt[li] == 0)).sum()
r = {
 "holdout_f05": base, "entities": n, "true_pairs": int(nt.sum()), "TP": int(tp), "FP": int(fp), "FP_on_singletons": int(fp_single),
 "decision_FN": int(fn_dec), "blocking_miss": int(blk_miss), "one_to_one_loss": int(o2o_loss),
 "oracle_perfect_decisions_on_candidates": F(lab.copy(), lab, nt),
 "fix_decision_FN": F(sel | lab, lab, nt),
 "fix_all_FP": F(sel & lab, lab, nt),
 "fix_FP_on_matched_entities": F(sel & (lab | (nt[li] == 0)), lab, nt),
 "fix_FP_on_singletons": F(sel & (lab | (nt[li] > 0)), lab, nt),
 "fix_blocking+o2o (all true pairs present, same decisions)": F(sel, lab, np.minimum(nt, in_cand)),
}
# per country
cc = _fetch_numpy(con, f"""SELECT s.idx, n.country_norm AS c FROM s1g s JOIN read_parquet('{io.cache_dir(cfg)}/norm_train_s1.parquet') n ON n.entity_id = s.s1_id ORDER BY s.idx""")
for c in sorted(set(cc["c"])):
    m = cc["c"] == c
    r[f"f05_{c}"] = f[m].mean(); r[f"n_{c}"] = int(m.sum())
# by entity size
for lo, hi, name in ((0, 0, "singleton"), (1, 1, "n=1"), (2, 3, "n=2-3"), (4, 1000, "n>=4")):
    m = (nt >= lo) & (nt <= hi)
    r[f"f05_{name}"] = f[m].mean(); r[f"share_{name}"] = m.mean(); r[f"loss_pts_{name}"] = (1 - f[m]).sum() / n * 100
print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, indent=1))
