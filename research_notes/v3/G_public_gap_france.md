# G — Public/local gap and France (research only, no pipeline changes)

Goal: explain local holdout macro F0.5 0.9715 vs public 0.961 for v2, with focus on France (test-only country,
15% of test S1, 259,452 entities). Run dir: `artifacts/runs/20260925T161436_b538130_1242e7cd_v2/`.
Scripts: `notebooks/v3_proof/00_crosscountry_check.py`, `01_country_calibration.py`, `02_expected_vs_realized.py`
(holdout section superseded — see correction below), `03_france_text_sample.py`, `04_holdout_by_country_faithful.py`,
`05_holdout_expected_vs_realized_faithful.py`. DuckDB temp: `artifacts/v3_proof_tmp/` (memory_limit 1500MB,
threads=2). Status: complete (Q1–Q5, Q7 measured; Q6 designed only, not executed).

## Finding 1 — Candidate generation (blocking) does not cross borders, and candidate density is similar per country

Query: joined `pred2_test.parquet` (29.9M pair scores) to S1 country and S2/S3 country (script:
`notebooks/v3_proof/00_crosscountry_check.py`, inlined below since it was a one-off check).

- Every one of the 29,908,767 test candidate pairs has `lid.country == rid.country`: India 14,152,695, US
  11,451,093, France 4,304,979. Blocking is entirely within-country (expected — normalize/block presumably keys on
  `country_norm`), so France's candidates cannot be diluted or enriched by cross-border noise.
- Candidates per S1 entity: India 14,152,695 / 809,986 = **17.47**, US 11,451,093 / 663,106 = **17.27**, France
  4,304,979 / 259,452 = **16.59**. Candidate density is close across countries (France is not starved of
  candidates at the blocking stage) — the gap, if real, is downstream of blocking (features/model/calibration/prior),
  not a blocking-recall artifact at this coarse level. (Per-pass blocking recall on France still needs checking
  against true matches, which test lacks — see Finding placeholder for indirect evidence via distractor/expected-F.)

## Finding 2 — France's own confidence distribution is much less peaked; the global EM prior shift over-corrects France

Script: `notebooks/v3_proof/01_country_calibration.py` (writes `notebooks/v3_proof/out/01_country_calibration.json`).
Per country, on **test**, pre-shift production-calibrated `p` (isotonic `cal_best`/`cal_other`, no logit shift yet):

| country | pairs | best-per-right | mean best p | median best p | share uncertain (0.05<p<0.95) | share best p < 0.10 | EM prior (own country) |
|---|---:|---:|---:|---:|---:|---:|---:|
| India  | 14,152,695 | 4,717,565 | 0.557 | 0.996 | **1.6%** | 42.9% | 0.5526 |
| US     | 11,451,093 | 3,817,031 | 0.581 | 0.999 | **1.9%** | 40.2% | 0.5773 |
| France |  4,304,979 | 1,434,993 | 0.620 | 0.996 | **5.6%** | 32.2% | 0.6112 |

- France has **3–3.5x the share of ambiguous mid-confidence pairs** (5.6% vs 1.6–1.9%) — the model is much less decisive
  on France, consistent with normalization edge cases (accents, legal forms, elision) blurring name/address
  similarity features rather than a clean match/non-match separation.
- Counter to the "France has more distractors" hypothesis at face value: the share of best-per-right candidates that
  still look like clear non-matches (best p < 0.10) is *lower* for France (32.2%) than India/US (40–43%). France
  doesn't have more obvious junk candidates; it has more candidates the model can't confidently call — a
  discrimination problem, not a volume-of-noise problem.
- train_prior_best (train, pooled India+US) = 0.68466. Per-country EM prior estimated straight from calibrated `p`
  (no shift applied) on test: India 0.5526 (implied logit delta **-0.564**), US 0.5773 (**-0.464**), France 0.6112
  (**-0.323**). The **production pipeline applies one pooled delta of -0.493 to all three countries** (EM run once
  on best-per-right p pooled across all test). Relative to each country's own EM estimate: India is *under*-shifted
  (needs -0.564, gets -0.493 → 0.07 logit too weak), US is close (needs -0.464, gets -0.493 → 0.03 logit too
  strong), and **France is clearly over-shifted** (needs only -0.323, gets -0.493 → 0.17 logit too strong, i.e. the
  pipeline pushes France's probabilities down by ~1.6x more than France's own data implies it should).
  Note EM's own-country estimate for France, run on only 1.43M best-per-right pairs vs 4.7M+3.8M for India/US, is
  noisier, but the direction (France needs a much smaller downward shift) is consistent with France's higher raw
  best-p mean/EM-prior above, not an artifact of sample size alone.

## Finding 3 — The model's own expected F0.5 already flags France as worse, even before any shift

Same script, `mean_expected_f_at_kstar` (model's internal expected F0.5 at its chosen prefix size, computed from the
production calibrator with **no logit shift**, i.e. the "does the model itself know" question in isolation from the
prior-shift knob):

| country | mean expected F0.5 (pre-shift) | share k*=0 | mean k* |
|---|---:|---:|---:|
| India  | 0.9694 | 6.5% | 3.20 |
| US     | 0.9705 | 6.0% | 3.29 |
| France | **0.9488** | 5.6% | 3.25 |

France's pre-shift expected F0.5 is already **~2 points lower** than India/US, purely from the calibrated
probabilities the model assigns to French candidates (more mass in the uncertain middle → lower expected precision
at any chosen k). This happens *before* the pooled EM shift is even applied, so yes — the model's own confidence
signal already "knows" France is harder; the pooled/global shift then additionally over-corrects France on top of
that (Finding 2), which should be expected to compound the loss unless F0.5's precision-weighting happens to prefer
the extra pull toward fewer predictions.

## Correction — a bug in the first holdout-by-country check (02_expected_vs_realized.py), and the faithful redo

`02_expected_vs_realized.py`'s holdout section computed `is_best` from a naive per-country subset of
`pred2_holdout.parquet`, ignoring that the production `_group_arrays` (ber/pipeline.py) drops/reassigns pairs whose
right record's *true* best candidate (over ALL rows, including rows for entities outside the holdout split) lies
outside the current group — "rival" handling (D-003). Skipping that inflated `is_best`/`cal_best` assignment for
many rival-losing pairs and produced a nonsense realized macro F0.5 of **~0.40** (vs the known local 0.9715). That
number is **wrong and discarded**; do not cite it. `04_holdout_by_country_faithful.py` and
`05_holdout_expected_vs_realized_faithful.py` redo it correctly (import `ber.pipeline._group_arrays` directly, same
function `fit_decision`/`compare_decisions` use) and reproduce the known pooled holdout **0.97151** exactly, so the
correct-path results below are trustworthy.

## Finding 3b — Holdout is genuinely well-calibrated (expected ≈ realized); per-country holdout split

`05_holdout_expected_vs_realized_faithful.py`, production params (shift=0.0, the fitted optimum), faithful rival
handling:

| country | n entities | mean expected F0.5 (k*) | mean realized F0.5 | expected − realized | macro F0.5 (incl. 0-cand) |
|---|---:|---:|---:|---:|---:|
| India  | 44,353 | 0.9718 | 0.9678 | +0.0040 | **0.9667** |
| US     | 66,096 | 0.9728 | 0.9755 | −0.0028 | **0.9748** |
| pooled | 110,449 | 0.9724 | 0.9724 | +0.00004 | **0.9715** (matches known local score) |

Holdout expected≈realized confirms the calibrator is well-behaved in-distribution (as its docstring claims) — the
gaps are ≤0.004 F0.5 points either direction. This is the honest baseline against which to read France's pre-shift
expected F0.5 of 0.9488 (Finding 3): it is **not** an artifact of a systematically biased "expected F" metric —
on data the calibrator has seen the shape of, expected F really does track realized F this tightly. So the ~2-point
gap between France's expected F and India/US's is a real signal the model has about France, *conditional on the
calibrator generalizing as well to France as it does within-distribution to India/US* — which is untested, since
train (and hence calibration fitting) has zero French entities. That last caveat is the main source of remaining
uncertainty: France could realize *worse* than its own expected F if isotonic `cal_best`/`cal_other`, fit only on
India/US score distributions, extrapolate over-confidently onto France's differently-shaped raw-score distribution
(Finding 2's 5.6% "uncertain" share suggests the raw scores *do* land in a different place for France).

## Finding 4 — Manual inspection of 42 sampled French test S1 entities: the dominant failure mode is structured
## "sibling company" distractors at nearby street numbers, not name-normalization breakage

Script: `notebooks/v3_proof/03_france_text_sample.py` → `notebooks/v3_proof/out/03_france_text_sample.txt` (42
entities stratified by calibrated best-p: high/mid/low, production shift applied, raw text + p per candidate).

**Most true matches score p≈1.000 and are chosen correctly** — character-level noise that superficially looks
France-specific (accent drift `é/è/ê`, `Jean Jaurés`→`JEAN JAURSÉ`, `çentre`, `Prïmaire`, `Çompagnie`→`Corgnie`,
`Fedhderation`, `Anlee`/`Allee`, `Etile`/`Émile`, abbreviations `R.`/`Rue`/`RUE`, `Bd`/`Boulevard`, house-number
markers `N°`/`Nº`/`No.`/`#`) is handled fine by the existing fuzzy features — these are not where matches are lost.

**The pervasive, distinctly-structured problem: near-duplicate "sibling" businesses at a nearby but different
street number**, sharing most of the name plus a suffix word (Développement / Participations / Groupe / Holding /
Amis / Cie / France / [no legal-form qualifier]), that must be *rejected*. Examples (calibrated p, production
shift, all correctly rejected in this sample but often close):
- S1-1998821 'Bourguignonne **Amis** SAS' @ 20 Rue De Candale vs true @ 19 → p=0.352
- S1-433848327 'Lège-Cap-Ferret **Federation**' (no SARL) @ 48 Allee... vs true @ 10 → p=0.262
- S1-834133116 'Club des Etoile **France**' @ 207 Rue Pierre de Roubaix vs true @ 206 → p=0.557 (the chosen match
  itself, `clubetoile.com`, has **no house number in its address at all** — genuinely thin margin)
- S1-930072393 'Defense Sportive **Groupe** SCI' @ 43 Rue Des Gants vs true @ 38 → p=0.690
- S1-97529486 'Dunkerque Institut **Holding** Sas' @ 8 Rue Oze Novembre vs true (unnumbered) → p=0.597
- S1-506530735 'Drexarc' (name-obscured but at the **exact same address**, 2 la Bate) → p=0.715, correctly rejected
  only on name grounds since the address is identical

**Of the 42 sampled entities, ~25 (60%) have at least one such sibling reject with calibrated p in [0.05, 0.9]** —
this is exactly the "uncertain pair" mass quantified in Finding 2 (5.6% of France pairs vs 1.6–1.9% India/US), now
confirmed to be a real, systematic pattern rather than measurement noise. It reads as a deliberately harder
distractor design for France (same-name-family shell/holding/subsidiary companies a few doors down the same
street), which the global logit shift (Finding 2, over-corrects France by ~0.17 logit) squeezes from both
directions at once: real matches near the shift boundary get pushed toward rejection (hurting recall) while
sibling near-misses sitting at p≈0.35–0.7 pre-shift are pushed down too (helping precision) — net effect on F0.5
is ambiguous from inspection alone and needs the leaderboard probe (Finding 6) or in-distribution labels to settle.
A secondary, non-France-specific complication also visible here: many right records replace the business name with
a domain-like or handle-like string (`clubetoile.com`, `#amicaleyoga`, `voixfrressas.com`, `@algeriegaulois`),
which forces the decision onto address fields alone — raising the stakes of exactly the street-number precision
the sibling-distractor pattern is designed to defeat.

## Finding 5 — Distractor rate synthesis (answers Q5)

Combining Findings 1–4: France does **not** have a higher raw volume of obvious junk candidates per S1 (candidate
density 16.6/entity vs 17.3–17.5 for India/US, Finding 1; share of best-per-right pairs with p<0.10 is *lower* for
France, 32.2% vs 40–43%, Finding 2). What France has more of is **hard, structurally-designed near-miss
distractors** — sibling companies one street-number off — that land in the calibrated-probability grey zone
(Finding 2's 5.6% uncertain share; Finding 4's manual confirmation). The pooled global EM prior-shift (`-0.493`)
is calibrated against a mix dominated 85%/15% by India+US vs France, and per-country EM estimates diverge:
India needs *more* shift (-0.564) than applied, US is close (-0.464 vs -0.493 applied), **France needs
substantially less (-0.323 vs -0.493 applied) — the pipeline over-corrects France's probabilities downward by
about 0.17 logit units (~1.6x the country-appropriate correction)**, on top of a raw (pre-shift) discrimination
gap the model already shows for France (Finding 3).

## Finding 6 — Leaderboard probe design (NOT executed — costs one upload; formula and uncertainty only)

**Design.** Take the exact v2 submission (commit `fc7aa50`, `output/matching_results.tsv`) and produce one
variant: for every French `source1_entity_id`, replace `matched_entity_ids` with the empty string (keep
`candidate_pairs.tsv` and every non-French row byte-identical, so this isolates France's contribution and nothing
else). Upload it under a fresh code zip built from the same commit (rules require the code zip to match the
uploaded TSV's producing commit — since we are not changing code, re-zip `fc7aa50` unchanged and only swap the
TSV). This costs one of the 5 daily upload slots.

**Backing out France's public F0.5.** Let `n_FR=259,452`, `n_other=1,732,544-259,452=1,473,092` (India+US), and let
`s_hat` = singleton share among France S1 (share with `n_true=0`, unknown on test — see estimate below). Under the
"empty France" probe, every French row scores exactly its singleton-or-not indicator: singletons (no true match)
score **1.0** on empty prediction (correct behaviour), non-singletons score **0.0** (empty prediction on an entity
that has a true match). So:

```
public_score_v2          = (n_other * F_other + n_FR * F_FR_v2) / n_total                (known: 0.961)
public_score_probe        = (n_other * F_other + n_FR * s_hat) / n_total                  (measured by the probe)
```

Two unknowns (`F_other`, `F_FR_v2`) but the probe removes `F_FR_v2` entirely and replaces it with the *known*
constant `s_hat`, so from the probe score alone:

```
F_other = (public_score_probe - n_FR/n_total * s_hat) / (n_other/n_total)
```
Subtracting the probe equation from the v2 equation eliminates `F_other` directly:
```
n_FR * (F_FR_v2 - s_hat) = n_total * (public_score_v2 - public_score_probe)
F_FR_v2 = s_hat + (n_total / n_FR) * (public_score_v2 - public_score_probe)
        = s_hat + 6.677 * (0.961 - public_score_probe)
```
(`n_total/n_FR = 1,732,544/259,452 = 6.6772`.) This needs only the probe score and `s_hat`; `F_other` is a
byproduct, useful to sanity-check against local India+US holdout (0.9667/0.9748 pooled ≈0.971, Finding 3b).

**Estimating `s_hat`.** Not directly observable, but the model's own expected empty-share is available without any
upload: from `02_expected_vs_realized.py`, `share_empty_all_entities` for France under the production shift is
**5.9%** (own-country-shift variant: 5.8%) — this is P(model predicts empty), not P(truly singleton), but since
holdout showed cal_empty is well-calibrated in-distribution (Finding 3b), `s_hat ≈ 0.06` is a reasonable central
estimate, with material uncertainty (French singleton rate could genuinely differ from what a model trained with
zero French labels predicts; ±0.03 plausible, i.e. `s_hat` anywhere in ~[0.03, 0.09]).

**Uncertainty on `F_FR_v2`.** The multiplier `6.677` amplifies noise: since the public leaderboard score is
presumably reported to 3 decimals (0.961), rounding alone gives `public_score_probe` a de facto ±0.0005
granularity, i.e. ±0.0033 on `F_FR_v2` from rounding alone — small. The dominant uncertainty is `s_hat`: a ±0.03
error there is a ±0.03 error on `F_FR_v2` directly (the `s_hat + ...` term), which is large relative to the
effect size being measured. **Recommendation if this probe is run:** report `F_FR_v2` as a range over
`s_hat ∈ [0.03, 0.09]` rather than a point estimate, and treat `F_other` as the more load-bearing sanity check
(if `F_other` comes back well below ~0.96, that itself is news — it would mean India/US public also underperforms
local, and France is not the sole story).

## Finding 7 — Ranked fixes (with evidence)

1. **Fit the EM prior shift per country, not pooled** (evidence: Finding 2 — France's own EM-implied shift is
   -0.323 vs the applied -0.493; India's is -0.564 vs applied -0.493). This is a small, safe, code-level change to
   `predict_stage`'s prior-shift block (`em_prior_shift` called once on `best_p[is_best]` pooled → call it three
   times, once per `country`, using the same `best_p` array already computed, and apply each country's own delta
   to its own rows before `decide()`). Directly measurable locally: rerun `predict_stage` and compare local France
   proxy metrics (mean k*, expected F, see `02_expected_vs_realized.py`'s `own_shift` column) and, if a probe
   (Finding 6) confirms France is the main loss, this is the cheapest lever to pull first — no retraining needed.
2. **Add explicit sibling/near-duplicate discriminative features**, since Finding 4 shows the dominant French
   failure mode is same-name-family companies at a *different but nearby* house number. Candidate features (all
   derivable from train/test text only, no external data, satisfying the competition's no-external-data rule):
   absolute and relative difference between parsed house numbers (`nums_shared`/`nums_conflict` exist already —
   check whether they saturate/lose signal when the base name match is otherwise near-perfect); a feature flagging
   "candidate name equals S1 name plus one extra suffix token" (Groupe/Holding/Développement/Participations/
   Amis/Cie/France/SNC-vs-SARL-etc.) as a *negative* signal distinct from ordinary legal-form variation. This is
   feature engineering + retraining, medium effort, but targets the exact evidence in Finding 4.
3. **Refit (or widen) the isotonic calibrators to be more conservative under distribution shift**, since Finding
   3b's caveat is untested: the calibrators have never seen a French raw-score distribution, and Finding 2 shows
   that distribution is meaningfully different (5.6% vs 1.6–1.9% uncertain mass). A cheap partial mitigation:
   shrink calibrated `p` toward 0.5 by a country-agnostic amount only for scores outside the range of raw scores
   seen in training (already-plausible given `Calibrator` clamps outside its knot range) — needs testing on
   holdout to avoid regressing India/US.
4. **Run the leaderboard probe (Finding 6)** to convert "France is probably the main gap" into a number, before
   spending more engineering effort — it's cheap (one upload) and resolves the biggest open uncertainty (whether
   India/US public actually matches local, an assumption the whole analysis currently rests on unverified).
5. Lower priority: investigate `nums_shared`/`nums_conflict`/`postal_state` feature distributions specifically on
   the ~60% of France entities with sibling near-misses (not done here — time-boxed out) to see whether the
   features already carry enough signal and the loss is purely a shift/calibration problem (fixes 1+3) or whether
   the model genuinely cannot separate these cases from the given features (fix 2 becomes higher priority).

_Status: core questions (1–5, 7) answered with measurements above; Q6 is a design only (not executed, per
instructions). Time-boxed out: per-pass blocking recall specifically for France (Finding 1 only shows aggregate
candidate density, not recall against true matches, which test lacks — would need the leaderboard probe or a
proxy via train-like synthetic French holdout to measure directly)._
