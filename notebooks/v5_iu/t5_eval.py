"""Evaluate prediction sets on holdout with a tune-fitted decision; optional averaging (bagging) of several pred sets;
paired bootstrap over holdout entities vs a reference.

usage: t5_eval.py REF_NAME NAME1[+NAME2+...] ...   (NAME = artifacts/v5_iu/pred_NAME_{tune,holdout}.parquet; 'v4' = run pred2)
'A+B' averages the probabilities of A and B (logit mean) before the decision.
"""
import json, sys
import numpy as np
from common import OUT, RUN, con, arrays, note
from ber.decision import fit_decision, decide, per_entity_f05


def path(n, g):
    return RUN / f"pred2_{g}.parquet" if n == "v4" else OUT / f"pred_{n}_{g}.parquet"


def combo(spec, g):
    """Pred parquet for a spec; for 'A+B' write the logit-averaged file once."""
    parts = spec.split("+")
    if len(parts) == 1:
        return path(parts[0], g)
    out = OUT / f"pred_{spec.replace('+', '_')}_{g}.parquet"
    if not out.exists():
        c = con()
        sel = " + ".join(f"ln(greatest(t{i}.p,1e-7)/greatest(1-t{i}.p,1e-7))" for i in range(len(parts)))
        joins = " ".join(f"JOIN read_parquet('{path(p, g)}') t{i} ON t{i}.lid = t0.lid AND t{i}.rid = t0.rid" for i, p in enumerate(parts) if i)
        c.execute(f"""COPY (SELECT t0.lid, t0.rid, (1/(1+exp(-({sel})/{len(parts)})))::FLOAT p, t0.label
                      FROM read_parquet('{path(parts[0], g)}') t0 {joins}) TO '{out}' (FORMAT parquet)""")
    return out


def entity_f(spec):
    t, h = arrays(combo(spec, "tune"), "tune"), arrays(combo(spec, "holdout"), "holdout")
    P = fit_decision(t)
    return per_entity_f05(h["lid_idx"], decide(P, h["lid_idx"], h["rid"], h["p"]), h["label"], h["n_true"])


if __name__ != "__main__":
    pass
else:
    ref = sys.argv[1]
    fr = entity_f(ref)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(fr), size=(300, len(fr)))
    lines = []
    for spec in sys.argv[2:]:
        f = entity_f(spec)
        d = f - fr
        bs = d[idx].mean(1)
        line = (f"| {spec} | {f.mean():.5f} | {d.mean()*100:+.3f} | [{np.quantile(bs,.025)*100:+.3f}, {np.quantile(bs,.975)*100:+.3f}] |")
        print(line, flush=True)
        lines.append(line)
    note(f"### Paired bootstrap vs `{ref}` ({fr.mean():.5f}); decision expected_f fitted on tune; delta in F-points, 95% CI over entities\n\n"
         "| variant | holdout F0.5 | delta pts | 95% CI |\n|---|---|---|---|\n" + "\n".join(lines))
