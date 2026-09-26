# J — France covariate shift: adversarial validation, shift-robust features, France-only decisions (v5 research)

Research/prototype only; `code/business_entity_resolution/` untouched. Scripts: `notebooks/v5_france/` (logs in
`notebooks/v5_france/out/`). DuckDB temp `artifacts/v5_tmp/`, memory_limit 3GB, 5 threads; LightGBM 5 threads.
Baseline = v4 run `artifacts/runs/20260926T063247_1a0d20f_99304c0b_v4/` (submission #6, public 0.982191, France ~0.950).
Findings are appended as each number is measured.

## D0 — entity-level weight of the rejected clean slice (d0_entity.py; out/d0.log)
- #6 per test country: empty-prediction share France 5.30% / India 5.93% / US 5.78%; matches per S1 3.41 / 3.32 / 3.37.
  Train truth: singleton (n_true=0) share 5.59% (both countries), mean n_true 3.46. So France is NOT under-predicting in
  aggregate (it predicts more matches per S1 than India/US, closer to the train truth).
- Rejected clean France pairs (same core name, identical numbers, r_rank=1, not accepted): 17,053 pairs (14,022 with p<0.1)
  on **11,425 distinct S1 (4.4% of French S1)**; only 593 of those S1 have an empty #6 prediction, most already have
  2-5 matches (1 rejected pair per S1 for 8,247 of them).
- Upper bound of the slice: if ALL were true and fixing cost nothing, gain ≈ 593×~1 + 10.8k×~0.07-0.09 ≈ 1.4-1.6k
  entity-units = **+0.006 France F0.5 ≈ +0.0009 overall**. The slice cannot close the France gap (0.95 → 0.98 needs
  ≈7.8k entity-units) on its own; if train rates hold (~40-70% true at these frequencies) the real gain is smaller.

## A — adversarial validation (a1_sample.py, a2_adversarial.py; out/a2.log, out/a2_adversarial.json)
Samples: 400k France test pairs, 408k India+US test pairs, 216k train pairs (fit+tune+holdout, in_group), v4 FEATURES.
LightGBM 150 rounds, 70/30 split.
- **France vs train: AUC 1.0000; France vs India+US test: AUC 1.0000; India+US test vs train (reference): AUC 0.965**
  (reference is driven by s_phx / nd_*_unexp_idf / s_n2 / name-freq: IDF-type columns differ between splits even in-domain).
- France is separable by many independent columns: dropping the top-5 / 10 / 20 gain features leaves AUC 0.9999 /
  0.9997 / 0.9994. So there is no single "shift feature" to fix: pair-level covariate shift is total, and only shift
  on the pairs that decide matches (true copies / their rivals) matters (next section).
- Top separators (gain) France vs train: l_n_cand 0.19, sg_rtgt_lo_max 0.11, r_n_cand 0.11, **admin_eq 0.11**, sg_num_nl
  0.07, jacc_addr, sg_del_lo_min, sg_a_del_lo_max, nd_l_logodds_max, s_phx. Univariate (France vs train, median / p90):
  l_n_cand 24 vs 9 (p90 75 vs 21), r_n_cand 5 vs 2 (13 vs 5), l_rank 11 vs 5, r_rank 3 vs 1, lrank 3 vs 1,
  **admin_eq median 0 vs 1 (AUC 0.25)**, compound-key scores s_hs / s_na / s_px median 0 vs 13-14, block_score -3.4 vs -1.8,
  address similarities lower (jw_addr 0.78 vs 0.92, addr_token_sort 0.75 vs 0.89), sg_a_repl / sg_a_del higher.
  The France pair population is dominated by many more (weaker) candidates per S1 and per right, with no admin
  (region) agreement and weaker compound-key blocking evidence.

## A3 — SHAP of the v4 models on the clean slice (a3_shap.py; out/a3.log, out/a3_shap.json)
Clean slice = numbers identical, core_ratio>=0.95, r_rank=1. (DuckDB applies USING SAMPLE before WHERE, so group sizes
are the in-sample counts: FR_rej (p2<0.1) 295, FR_acc (p2>=0.9) 11,633, TR_true 11,970, TR_false 37, TR_hf_true 874;
the means below are still unbiased for each group; TR_hf_false n=9 is ignored.)
- Train clean slice (tune+holdout, 333k pairs): 99.71% true, 0.25% get p2<0.1; with r_name_freq>=200: 25.9k pairs,
  **99.10% true**, mean p2 0.991. So common name alone does NOT make the model reject clean pairs on train.
- France rejected pairs differ from train true pairs mainly in the ADDRESS, not the name: median street_jw 0.625 vs 1,
  street_token_set 0.46 vs 1, addr_token_set 0.69 vs 1, sg_a_repl 2 vs 0 (two address tokens replaced), plus the
  crowd context (r_n_cand 6 vs 1, l_n_cand 31 vs 8, lrank 2 vs 1, l_rank 7 vs 1, r_name_freq 354 vs 5, block_score
  -0.57 vs 5.58, compound-key scores s_na/s_hs/s_px 0 vs 42-82).
- **Stage-1 contributions FR_rej minus TR_true (logit): cut_p -5.02 (!), lrank -1.08, street_jw -0.97, sg_a_repl -0.93,
  r_n_cand -0.68, addr_token_set -0.66, street_token_set -0.66, l_rank -0.64, sg_n_repl -0.54, l_n_cand -0.53,
  sg_edit_lo_max -0.46, r_name_freq -0.46, block_score -0.37, legal_eq -0.31, r_same_name_cands -0.27, r_core_l_freq -0.19.**
  Total stage-1 logit -6.2 vs +11.7. The single biggest term is **cut_p** (the block3 cut GBDT score, a label-trained
  India/US model output used as a feature; median 0.68 on FR_rej vs 0.999): the stage-1 model gives it +5 logit on
  train true pairs, ~0 here. Crowd/context features together ≈ -3.5 logit; address-difference features ≈ -3.2.
- Stage 2 just propagates it: s2_p1 -8.9 and s2_r_margin -8.6 (p1 median 0.002; the right's best p1 is a different S1
  with an equally tiny p1), s2_l_gap -1.05; other stage-2 terms are small (street_jw -0.28, r_name_freq -0.08).
- So the rejection is a stage-1 decision, driven by (i) cut_p (learned, shifted), (ii) the crowd/rank/count context
  (shifted), (iii) a genuinely different street (street_jw ~0.6, 2 address tokens replaced) — which is real evidence
  against a match when the name is shared by hundreds of French businesses.

## C — are French "same name + same number + different street" pairs true? Label-free collision estimate (c1_collision.py, c2_autocorr.py; out/c1.log, out/c2.log)
Method: join S1 x (S2+S3) on equal name_core within a country; count pairs by house-number offset d = h_R - h_L and
street Jaro-Winkler bucket (same >= 0.9, mid 0.6-0.9, diff < 0.6) x city equality. Chance collisions of different
same-named businesses are smooth in d: their count at d = 0 is estimated from offsets no copy/distractor uses
(d <= -3, even d >= 6), scaled by the house-number autocorrelation ratio A(0)/mean A(base) (1.61 train, 1.70 France,
1.32 US: small numbers are over-represented). **Validated on a 10% train sample with labels**: estimated true share
0.834 vs real 0.806 (diff street, other city), 0.655 vs 0.679 (mid street, other city), 0.999 vs 0.997 (diff, same city).

| cell (street, city) | train n(d=0) / real true | France n(d=0) | France chance est. | **France est. true share** | US test est. true share |
|---|---|---|---|---|---|
| diff, other city | 34,559 / 0.806 | 54,905 | 33,360 | **0.39** | 0.84 |
| diff, same city | 12,223 / 0.997 | 37,005 | 27,978 | **0.24** | 1.00 |
| mid, other city | 2,584 / 0.679 | 36,619 | 33,506 | **0.09** | 0.75 |
| mid, same city | 4,983 / 0.997 | 56,131 | 62,116 | **≈0** | 1.00 |
| same street, other city | 47,755 / 0.998 | 103,787 | 324 | 0.997 | 1.00 |
| same street, same city | 145,850 / 0.999 | 401,806 | 670 | 0.998 | 1.00 |

**The hypothesis "the model wrongly rejects thousands of true French copies" is refuted.** In France, because names
are templated (hundreds of "<city> club" per city) and house numbers are small, two different businesses share
name + number by chance very often: with a different street the pair is true only 24-39% of the time, with a
"similar" street in the same city ~0% (vs 80-100% in train and in the US test). The v4 model's rejection of the
clean-but-different-street slice (street_jw median 0.625) is mostly CORRECT; forcing those pairs in would add FPs.
Corollary: any feature change that makes France look "train-like" on frequency / crowd context would push these
pairs toward acceptance and is expected to HURT. The useful direction is the opposite: check whether the model
ACCEPTS French pairs in the chance-dominated cells (next).

## C3 — does v4 over- or under-accept in those cells? (c3_accepted.py; out/c3.log)
Same-core-name, same-number pairs; accepted = in #6 output; est. true from C2.

| cell | France n / in cand / **accepted** / est. true | US test (25% S1) n / accepted / est. true |
|---|---|---|
| diff street, other city | 54,905 / 37,033 / **14,928** / 21,545 | 9,498 / 7,947 / 7,955 |
| diff street, same city | 37,005 / 25,008 / **8,234** / 9,027 | 4,542 / 4,539 / 4,539 |
| mid street, other city | 36,619 / 22,742 / **307** / 3,113 | 1,698 / 1,307 / 1,280 |
| mid street, same city | 56,131 / 37,494 / **941** / ≈0 | 5,399 / 5,390 / 5,397 |
| same street (both) | 505,593 / 505,168 / **503,928** / 504,599 | 209,157 / 209,104 / 209,105 |

- On the US test the model's accepted count equals the estimated true count in every cell (within 30 pairs), which
  also validates the estimator. In France the model accepts about as many pairs as are estimated true in the
  different-street cells (23.2k accepted vs 30.6k est. true, of which only ~60-70% are candidates at all), i.e. it
  is roughly count-calibrated there; the possible recall gap is a few thousand pairs at most and the possible FP
  excess (mid street, same city) is <= ~1k pairs. **There is no large exploitable error in the exact-name cells**;
  the France gap must come mostly from pairs whose names differ (edited copies vs same-template different
  businesses) and from blocking, not from over-rejection of clean pairs.

## F — France-only decision variants on v4 stage-2 scores (f_decision.py; out/f_decision.log)
Changing only the France decision (v4 pred2, v4 decision params). Columns: pairs added/removed vs v4 and how many of
them are near-certain TRUE (same core name + same number + same street), near-certain FALSE (same core name, house
number off by a non-copy offset), chance-collision cells (same name+number, other street; ~0-40% true), +1/+2 vs
-1/-2 single-number changes (excess = nudged distractors), generator offsets >=3 (distractors).

| variant | accepted | empty S1 | changed pairs | sure-true | sure-false | collision | +1/+2 | -1/-2 | gen>=3 |
|---|---|---|---|---|---|---|---|---|---|
| v4 (shift 0) | 884,146 | 5.30% | - | | | | | | |
| shift -0.75 | 869,083 | 5.48% | -15,063 | 294 | 143 | 307 | 476 | 45 | 203 |
| shift -0.43 (= France EM estimate) | 874,385 | 5.40% | -9,761 | 191 | 90 | 206 | 358 | 29 | 149 |
| shift -0.25 | 878,088 | 5.35% | -6,058 | 118 | 47 | 129 | 240 | 19 | 92 |
| shift +0.25 | 888,866 | 5.23% | +4,720 | 65 | 40 | 123 | 226 | 16 | 108 |
| shift +0.50 | 893,365 | 5.17% | +9,219 | 133 | 84 | 251 | 528 | 37 | 211 |
| empty_weight 0.6 / 1.0 / 1.5 | | | +264 / -140 / -462 | | | | | | |
| max_k 8 / 24 | | | -510 / 0 | 53 / 0 | | | | | |

- The marginal pairs of any shift are a mix: per 10k marginal pairs, ~200 near-certain true vs ~600-800 near-certain
  false (distractor excess + generator offsets + wrong-offset same-name), with ~90% not characterizable. Their F0.5
  value is ambiguous (precision counts double, so removing a 50/50 pair is worth doing; removing a 70/30 true pair is
  not). The v4 decision is near a flat optimum; a France shift in [-0.43, 0] changes <1.1% of French matches.
  Expected effect of -0.43 (re-enabling the EM shift for France): removes ~500 labelled-false and ~190 labelled-true
  pairs among 9.8k; direction mildly positive if the uncharacterized 9k are <~65% true, else negative — not a
  confident submission. max_k 8 removes 53 near-certain true pairs: keep 16.

## G — coordinator check: very short (len<=2) S2/S3 names (g_short.py; out/g_short.log)
- **Concentrated in France**: S2 1.39% France vs 0.003% India / 0.018% US; S3 1.67% France vs 0.29% / 0.04% (train
  India 0.37%, US 0.05%). So the train->test doubling is France (21,930 French very-short rights are candidates).
- They are initialisms of the S1 name, i.e. true-copy abbreviation noise ("SP" = Site Parents, "LC" = Lille Club,
  "RR" = Representatif Residence), usually with the same number and street. #6 accepts 16,387 of 21,930 (75%;
  India 99%, US 94%). None are in the rejected clean slice (core_ratio >= 0.95 excludes them).
- Examples of rejections look like real recall losses when the initials are ambiguous among templated names
  ("LC" / 13 Rue Aristide Briard vs "Lart Club" / 13 Rue Aristide Briand, p=0.39). Up to ~5.5k French pairs are
  rejected here; a France-shift-aware initialism feature (does the right name equal the initials of the S1 core
  name) is a possible, unmeasured recall lever.

## E — proxy protocol and BASE (e_proxy.py; out/e_BASE.log, out/e_BASE.json)
Single-stage LightGBM (ber DEFAULT_PARAMS, lr 0.2, seed 42) on in_group rows of 35% of the v4 FIT S1 (hash bucket,
1.89M rows), early-stopped on tune_in; full tune/holdout files scored; ber choose_decision (tune only); holdout
macro F0.5 overall / India / US / templated-name S1 (S1 core-name frequency >= 50 and >= 200 in train S1).
France: all 5.35M French pairs scored, decided with the fitted params, no EM shift (as v4).
- **BASE: holdout 0.98586** (India 0.98394 n=44,353; US 0.98714 n=66,096; namefreq>=50 0.97874 n=19,531;
  namefreq>=200 0.98309 n=3,620), 308 trees. France: model-expected F0.5 0.9715 (self-assessed, not a real score),
  matches/S1 3.384, empty 5.16%, accepted +1/+2 678 vs -1/-2 450, generator offsets 301; vs #6: +12,716 / -18,832
  pairs (single-stage proxy vs the 2-stage v4). v4-rejected clean slice (14,022 pairs): median p 0.0026, 75 accepted.
  Same-name cells accepted at d=0: diff/other-city 14,934, diff/same-city 8,259, mid 319 / 1,174, same street 503,889
  (matches v4 within 1-2%, so the proxy reproduces v4's France behaviour on these cells).

## H — French name-edit distractors among #6 accepted pairs (h_genwords.py; out/h_genwords.log)
Pairs (r_rank<=2) whose right name contains a word the S1 name lacks; numeq = all address numbers identical.
- Pure French generator words are rejected almost perfectly even with identical numbers: accepted / numeq pairs:
  holding 44 / 17,895, participations 24 / 19,295, distribution 25 / 18,555, international 26 / 19,193, snc 2 / 8,818.
  (So French name-only distractors are frequent — ~18k per word — and the self-labelled vocab handles them.)
- groupe 6,341 / 32,439, developpement 5,817 / 29,486, france 6,273 / 58,919 accepted (~11-20%), the same order as
  the neutral legal-form insertions (sarl 7,445, sas 7,544, sci 7,466). Offset check: with holding's number profile
  (numeq : other = 0.46), pure-distractor groupe / developpement / france would show 21.3k / 19.6k / 24.4k numeq pairs;
  observed 32.4k / 29.5k / 58.9k, i.e. ~11k / ~10k / ~35k numeq pairs in excess = these words also occur as TRUE-copy
  noise ("(France)" is part of many S1 names). Accepted counts (~6k each) are below those excesses, so there is no
  sign of a large accepted-distractor pool here either (at most a few thousand pairs if the model picks the wrong ones).

### E1 — NOCUT: drop cut_p (out/e_NOCUT.log, out/k_NOCUT.log)
- Holdout **0.98580 vs BASE 0.98586 (-0.00006)**; India 0.98375 (-0.00019), US 0.98717 (+0.00003), namefreq>=50
  0.97851 (-0.00023), >=200 0.98297 (-0.00012). No meaningful in-domain cost: the rest of the features carry cut_p's information.
- France: matches/S1 3.372 (BASE 3.384), empty 5.38% (5.16%), accepted +1/+2 568 vs -1/-2 469 (excess 99; BASE 228),
  generator offsets 299 (301). v4-rejected clean slice: median p 0.0061 (0.0026), 164 accepted (75): cut_p was the
  largest SHAP term on that slice, but removing it barely moves it (other features absorb the evidence).
- Changed French matches vs BASE: +9,805 / -12,848. Label-free profile of ADDED: 186 near-certain true, 92 off-offset
  false, 439 collision-cell, +1/+2 184 vs -1/-2 41, gen offsets 140, pure generator word 70; REMOVED: 161 / 82 / 244 /
  294 vs 22 / 142 / 45. Balanced: NOCUT reshuffles ~4% of French matches with no measurable net direction.
  (For scale, BASE-proxy vs #6 itself differs by +12.7k / -18.8k.)

### E2 — PCT: per-country percentile normalization of 10 context/frequency features (out/e_PCT.log, out/k_PCT.log)
l_name_freq, r_name_freq, r_core_l_freq, r_same_name_cands, l_n_cand, r_n_cand, l_rank, r_rank, lrank, block_score
-> mid-rank ECDF within (world, country) (train world: India / US reference; test world: France reference; references =
the A1 adversarial samples). Makes France's crowd context look like train's.
- Holdout **0.98594 vs BASE 0.98586 (+0.00008)**; India 0.98383 (-0.00011), US 0.98735 (+0.00021); **France-like
  in-domain slices slightly worse**: namefreq>=50 0.97850 (-0.00024), >=200 0.98279 (-0.00030).
- France: **matches/S1 3.519 (BASE 3.384; +38.6k / -3.7k pairs vs BASE)**, empty 4.81%, model-expected F0.5 0.9676
  (BASE 0.9715: the model itself is less sure). v4-rejected clean slice: median p 0.050, 884 accepted (BASE 75).
- Label-free profile of the 38,568 ADDED pairs: only 426 near-certain true; FP markers 287 off-offset same-name,
  **2,039 chance-collision cells** (mid-street same-city/other-city accepted 2,889 vs BASE 1,493, where C2 estimates
  ~0-9% true), 329 pure generator-word insertions (holding/participations/...); +1/+2 203 vs -1/-2 147. REMOVED 3,700:
  296 +1/+2 vs 11 -1/-2 and 169 generator offsets (i.e. PCT drops ~450 distractors), 59 sure-true.
- Verdict: PCT does what the hypothesis asked (accepts more crowded French pairs) but C showed those pairs are
  mostly chance collisions; FP markers outnumber TP markers ~6:1 among added pairs, and France-like train slices get
  worse. **Expected to HURT France** (+38.6k matches ≈ +0.15 per French S1 on a set whose precision is unknown and
  likely well below 2/3, the F0.5 break-even). Not recommended.

### E3 — COMBO (drop cut_p + drop admin_eq/admin_diff + PCT) (out/e_COMBO.log, out/k_COMBO_NOADMIN.log)
- Holdout **0.98567 (-0.00019 vs BASE)**, rule switched to threshold; India 0.98372, US 0.98698 (-0.00016).
- France: matches/S1 **3.597** (+58.6k / -3.4k vs BASE), model-expected F0.5 0.9622; v4-rejected clean slice median p
  0.61, 4,364 accepted. ADDED profile: 505 near-certain true vs FP markers 415 off-offset, **6,327 chance-collision**,
  1,125 pure generator-word distractors, +1/+2 408 vs -1/-2 162, 313 generator offsets. Clearly harmful. Not recommended.

### E4 — NOADMIN: drop admin_eq / admin_diff (out/e_NOADMIN.log, out/k_COMBO_NOADMIN.log)
- Holdout **0.98590 (+0.00004 vs BASE)**; India 0.98390, US 0.98725, namefreq>=50 0.97856, >=200 0.98306 (all within noise).
- France: matches/S1 3.427 (BASE 3.384; +16,105 / -4,968), empty 5.12%, model-expected F0.5 0.9727 (BASE 0.9715).
  ADDED: **350 near-certain true** (vs 58 removed), 82 off-offset, 220 collision, +1/+2 228 vs -1/-2 29, 140 gen offsets,
  33 generator words; REMOVED: 83 off-offset, 345 collision, 231 vs 21, 85 gen, 42 generator words. FP markers are
  balanced (added ≈ removed), TP markers favour NOADMIN (+292). The only arm with a favourable label-free profile.

## M — why admin matters in France (m_admin.py; out/m_admin.log)
- **France admin_diff = 1 on 119,092 of 505,168 near-certain true pairs (24%)**: the S1 carries the REGION and the copy
  the DEPARTMENT ("Lille, Hauts-de-France" vs "Lille, Nord"; "Mérignac, Nouvelle-Aquitaine" vs "MERIGNAC, Gironde").
  14.5% of all French candidate pairs have admin_diff = 1. In train, admin_diff = 1 is real counter-evidence (clean
  same-name same-number holdout pairs: 94.4% true vs 99.8% with admin_diff = 0), so the model penalizes it.
- v4 still accepts 99.7% of those strong pairs, but for weaker French true copies (name edits, abbreviations,
  initialisms, street typos) the false "different region" signal costs logit. This is a genuine, France-specific,
  mechanism-level shift (unlike the crowd features), and it is fixable transductively without external data.

## Ranked candidates for tomorrow (all measured with the single-stage proxy; France has no labels)

Summary table (holdout = in-domain macro F0.5, proxy protocol; France = label-free):

| arm | holdout (Δ vs BASE) | India / US | namefreq>=200 | France matches/S1 | France Δ pairs vs BASE | label-free verdict |
|---|---|---|---|---|---|---|
| BASE | 0.98586 | 0.98394 / 0.98714 | 0.98309 | 3.384 | - | reproduces v4 cells |
| NOADMIN | 0.98590 (+0.00004) | 0.98390 / 0.98725 | 0.98306 | 3.427 | +16.1k / -5.0k | **favourable** (+292 sure-true, FP markers neutral) |
| NOCUT | 0.98580 (-0.00006) | 0.98375 / 0.98717 | 0.98297 | 3.372 | +9.8k / -12.8k | neutral reshuffle |
| PCT | 0.98594 (+0.00008) | 0.98383 / 0.98735 | 0.98279 | 3.519 | +38.6k / -3.7k | harmful (collision FPs) |
| COMBO | 0.98567 (-0.00019) | 0.98372 / 0.98698 | 0.98301 | 3.597 | +58.6k / -3.4k | clearly harmful |

1. **Admin equivalence for France (region vs department), self-labelled** — best candidate. Measured proxy of the
   blunt version (drop admin_eq/admin_diff): in-domain +0.00004, French matches +11k net with 350 near-certain-true
   additions vs 58 removed and balanced FP markers. Expected France +0.001..+0.004 (overall +0.0002..+0.0006);
   low risk in-domain. Implementation (ber): in `features.py` SQL (lines ~219-220) compute admin_eq / admin_diff
   through an equivalence table `admin_equiv(country, a, b)` learned per country from self-labelled true pairs
   (same core name, identical address numbers, street Jaro-Winkler >= 0.9 — the D-025 self-label rule plus street):
   count (l.addr_admin, r.addr_admin) with both non-empty and different; keep pairs with count >= 50 and >= 10% of
   a's self-labelled pairs; then admin_eq = 1 and admin_diff = 0 when (ladm, radm) or (radm, ladm) is in the table.
   Learn it in a small stage next to learn_selfvocab_stage (train from train candidates, France from French test
   candidates; transductive, no labels, no external data), store as learned/admin_equiv.json, apply at feature
   time to train and test rows alike, then retrain v4 unchanged. Fallback if time is short: drop the two columns
   from FEATURES (exactly the measured NOADMIN arm, but for all countries).
2. **v4 as is (#6) + France EM shift -0.43** (decision only: remove France from `decision.shift_skip_countries`):
   removes 9.8k French matches containing ~570 labelled-false vs 191 labelled-true; direction unclear (break-even is
   the unlabelled 9k being <~65% true). Low priority; only if a slot is spare.
3. **NOCUT** (drop cut_p): in-domain neutral, France reshuffle with no measurable net direction. Not worth a slot alone.
4. Unmeasured idea: an **initialism feature** (right name == initials of S1 core-name tokens) for the 21.9k French
   very-short rights (only 75% accepted vs 99% India), where ambiguous initials of templated names lose true copies.

**Do not submit**: PCT / COMBO or any change that makes France's crowd/frequency context look like train (measured
+38.6k / +58.6k French matches dominated by chance-collision and generator-word FPs; France-like train slices worse).

## Bottom line
- The "model wrongly rejects clean French copies" hypothesis is refuted by a validated label-free estimator: in
  France same name + same number + different street is mostly a chance collision (24-39% true; ~0% with a similar
  street in the same city), because names are templated and house numbers small. v4 accepts about the estimated
  true count in every such cell.
- Shift-robust normalization of the crowd features does not help and likely hurts. The one France-specific
  mechanism found that is both real and fixable is admin disagreement (department vs region, 24% of near-certain true
  pairs). Expected gains are ~0.001-0.004 France, far from the ~0.03 needed for 0.98; a large part of France's gap is
  likely irreducible ambiguity of templated same-name businesses (≈40k same-name same-number other-street pairs, most
  of them chance) plus blocking/name-edit errors that no label-free check here can see.
- Proxy caveats: single-stage, 35% of fit, single seed; in-domain differences < ~0.0002 are noise.
