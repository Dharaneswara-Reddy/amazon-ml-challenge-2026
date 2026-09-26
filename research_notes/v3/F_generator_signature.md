# v3 research F: reverse-engineering the distractor generator (signatures -> features, measured on holdout)

Scripts: notebooks/v3_proof/f*.py (outputs in notebooks/v3_proof/out/, git-ignored scratch; DuckDB temp in
artifacts/v3_proof_tmp/). Analysis samples use the FIT group only (md5 buckets 0-29 of the S1 / parent id);
holdout is touched only for the final measurement. Findings are appended as they are made.

## Samples (f01_sample.py, f02_ops.py, sigdiff.py)
- **dist**: 120,562 unmatched right records (md5(rid)%100<15) whose parent = the cand_train candidate with max
  jw_core+jw_addr is in the fit group. `dist_near` = parent score >= 1.7 (81,404 pairs; parent confidently identified),
  `dist_far` = the rest (39,158; often no real parent in the candidates, e.g. a random different business).
- **true**: 114,697 GT pairs, S1 in fit, md5(match_id)%100<5 (in candidates or not).
- Diff engine (`sigdiff.py`): accent fold + lowercase, letter/digit swap repair, token multiset diff; leftovers paired
  into typo (ratio>=75) / abbreviation (same first letter + in-order subsequence) / truncation / concatenation, the
  rest = replaced (del+ins pairs), inserted, deleted. Address digit runs: equal runs (leading zeros stripped) removed,
  leftovers paired (same length, min |diff|) and classified pad / trunc_last / drop_first / drop_inner / add_digit /
  transpose / offset (signed R-L).

## Finding 1: the distractor number nudge is an exact, uniform draw from 10 offsets (f03_tables.py num)
Distractors look like "parent S1 -> distractor edit -> the SAME copy-noise generator as true copies" (all-caps
addresses, state full names, `#`/`-` decorations, "Corp Corp", leading `>>` appear on distractors too).

Records with >=1 changed address number (after removing equal numbers): **dist_near 93.0%, true 7.7%**.
Changed pairs per record: dist_near 1 -> 74,834, 2 -> 845; true 1 -> 8,772, 2 -> 87 (exactly one number is nudged).

Signed offset R-L of the changed number (class `offset`, same-length pairing):

| offset | +1 | +2 | +3 | +4 | +5 | +7 | +9 | +11 | +13 | +21 | -1 | -2 | +6/+8/+10/+12 | other in [-40,40] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dist_near | 7,091 | 7,115 | 6,951 | 7,043 | 7,064 | 6,904 | 6,059 | 6,742 | 6,823 | 6,759 | 120 | 76 | 33-90 each | 0-43 each |
| true | 299 | 311 | 31 | 21 | 16 | 29 | 9 | 11 | 14 | 3 | 302 | 273 | 8-39 each | 0-33 each |

**Distractor offset set S = {+1,+2,+3,+4,+5,+7,+9,+11,+13,+21}, drawn uniformly (~7k each), always positive.**
True-copy number noise = symmetric +-1/+-2 typos (~300 each), digit truncation (trunc_last 2,068, drop_first 1,770
vs 720/110 in dist), and scattered single-digit substitutions (+-10, +-20, +-100 ...). Offset in S\{1,2} occurs on
134 true pairs vs ~54k near distractors. In multi-number addresses the nudged number is usually the first
(house) number but can be any (e.g. `576/3398 -> 576/3409`, `24-2-1373 -> 24-2-1376`, `8200-8300 -> 8200-8307`).

## Finding 2: distractor name edits come from small fixed vocabularies (f03_tables.py name)
Name op presence (share of pairs; Latin R names; ops after typo/abbrev/truncation pairing):

| pattern (replaced, inserted, deleted, reordered) | dist_near | true |
|---|---|---|
| no token op at all | 2.1% | **37.8%** |
| insert only | **45.5%** | 15.8% |
| replace only | **32.4%** | 12.6% |
| replace + insert | 8.9% | 1.4% |
| delete only | 1.9% | **17.9%** |
| replace + delete | 4.2% | 6.5% |
| reorder only | 0.3% | 6.9% |
| insert + reorder | 3.2% | 0.2% |

Joint with the number nudge: **91.5% of dist_near have a name edit AND a changed number**, 1.5% number only,
6.5% name only, 0.5% neither. True copies: 5.0% have both (mostly name noise + digit truncation), 30.5% neither.
**R address empty: 0.0% of dist_near (0.9% of dist_far) vs 4.4% of true copies** - the distractor generator never
blanks the address.

Distractor name operations (counts in dist_near vs true):
1. **Legal-form swap** (a legal form replaced by a DIFFERENT legal/org word): llc->inc 1,229/0, llc->ltd 1,228/0,
   llc->corp 1,211/0, llc->co 1,197/2, llc->group 1,105/0, llc->holdings 1,080/1, llc->partners 1,080/0,
   inc->{corp,ltd,co,llc,partners,holdings,group} ~700-785 each / 0-1, private->public 816/0, private->llp 514/0,
   pvt->public 226/0, limited->{enterprises,industries,exports,ventures,overseas,group,holdings,infratech,llp} ~90-150/0-5.
   True copies drop or add legal forms (limited deleted 5,732, llc deleted 5,584, inc inserted 1,406) but
   essentially never swap one for another.
2. **Suffix/org word appended**: group 2,880/12, holdings 2,834/6, partners 1,993/155, industries 1,267/10,
   overseas 1,235/2, ventures 1,227/11, exports 1,199/0, infratech 1,198/0, enterprises 1,191/87, public 337/3
   (and ltd/inc/co/corp/llc ~2.1-2.6k each, but those are also common true-copy inserts: 760-1,400).
3. **Location qualifier inserted** (a closed list of 21 words, ~600 each = uniform draw): east, west, north, south,
   uptown, downtown, midtown, eastgate, westgate, riverside, southside, northside, summit, central, greater, metro,
   lakeside, valley, coastal, harbor, highland -> 0-5 occurrences on true copies each.
4. **Content word replaced by another business word** (perez>wood, auto>home, logistics>builders, gems>infratech,
   telecom>exports, block>motors, health>lumonyx pseudo-word) - targets include solutions, technologies, trading,
   foods, consulting, agro, marketing, media, developers, engineering (50-150 each, ~0-10 on true).
True-copy name noise: legal form dropped (limited/llc/inc/private/ltd), prefixes (dr, mr, smt, sri, shri, the),
generic words (center 3,017 / services 2,103 / service 1,005 as replacement of private/pvt/limited), accents,
case, letter/digit swaps, truncation, concatenation / domain style, reordering, transliteration.
Address words: the op tables of address tokens (state code <-> full name, unit/null/pmb/po box, city renames) are
the SAME for distractors and true copies (copy noise applied to both) - the address signature is the number only.

## Finding 3: distractors are generated from the S1 record, then the normal copy noise is applied (f04_sibling.py)
Raw case-sensitive similarity (name ratio + address ratio) of a record to its S1 vs to its closest sibling copy:
distractors are closer to a sibling in 64% of 24,577 cases, true copies in 59% of 45,155 (control) - the same
pattern (siblings share S2/S3 formatting such as all-caps addresses). No sign that distractors are copies of a
specific noisy copy. Model: **distractor = copy_noise(edit(S1))** with edit = {name edit from the vocabularies of
finding 2} + {one address number + offset from S}, never an emptied address. So a pair feature must look *through*
the copy noise: typo/abbrev/truncation/case/accents/legal-drop are shared noise, while legal-SWAP, vocab inserts
and positive offsets in S are the generator's distractor fingerprint.

## Finding 4: separability on real candidate pairs (f08_separability.py, TUNE group, 1.16M pairs with p2 >= 1e-3)
Features from `f05_feats.py` (definitions below). "offset in S" here = exactly one changed number and its signed
offset in {+3,+4,+5,+7,+9,+11,+13,+21} (+1/+2 excluded because true copies also make +-1/+-2 typos).
Name-edit log-odds = smoothed log(dist_near count / true count) of the inserted word / replacement pair, learned on
the fit-group samples only.

| signature | pairs | P(non-match) | pairs 0.05<=p2<=0.95 | P(nm) there | pairs p2>0.95 | P(nm) there | non-matches with p2>0.5 |
|---|---|---|---|---|---|---|---|
| sole offset in S (>= +3) | 41,501 | 0.974 | 10,096 | 0.961 | 868 | 0.239 | 1,716 |
| offset -1/-2 | 80,918 | 0.530 | 31,123 | 0.578 | 23,982 | 0.005 | 2,190 |
| number truncated / digit dropped | 110,878 | 0.400 | 32,590 | 0.575 | 51,828 | 0.003 | 2,477 |
| name edit log-odds >= 3 (legal swap / vocab insert) | 100,953 | 0.274 | 6,914 | 0.712 | 71,418 | 0.002 | 1,039 |
| legal-swap pair (repl lo >= 5) | 91,683 | 0.217 | 5,469 | 0.650 | 69,892 | 0.0015 | 711 |
| vocab insert (ins lo >= 5) | 79,194 | 0.092 | 3,900 | 0.515 | 70,012 | 0.001 | 541 |
| **name edit lo >= 3 AND offset in S** | **12,953** | **0.9944** | 1,991 | 0.9945 | 101 | 0.406 | 400 |
| name edit lo >= 3 OR offset in S | 129,501 | 0.426 | 15,019 | 0.842 | 72,185 | 0.004 | 2,355 |
| no name token op and numbers equal | 546,254 | 0.0055 | 3,722 | 0.290 | 540,671 | 0.0001 | 319 |
| R address empty | 75,743 | 0.561 | 29,161 | 0.614 | 21,171 | 0.005 | 2,172 |
All band pairs: P(nm) 0.109; 6,812 non-matches have p2 > 0.5 (the FP pool). The two signatures cover 2,355 of them (35%).

Caveats that stop a >99%-precision rule on its own:
- On candidate pairs (harder than a random sample), true copies ALSO show small signed offsets smoothly spread
  over +-1..+-20 (digit substitutions: p2>0.3 band, true/non-match: +3 204/377, +6 181/3, +8 126/2, +10 263/6,
  +13 85/465, +20 226/10, +21 17/453). The non-match mass sits exactly on S; off-S offsets (+6,+8,+10,+12,+14..20)
  are ~98% TRUE. So "offset in S" alone is ~0.97 non-match overall, ~0.70 non-match at p2>0.3.
- A legal swap such as `llc -> inc` is also produced on true pairs when copy noise drops one legal form and adds
  another (P(nm)=0.22 overall). Only the conjunction name-edit-from-distractor-vocab AND offset-in-S is rule-like
  (0.9944), and it covers only 12,953 pairs, most already scored low by v2.
- The generator's distractor = name edit + number nudge (finding 2) - but copy noise afterwards can hide either
  half (truncating the nudged number, dropping the inserted suffix), so a sizeable remainder stays ambiguous.

## Finding 5: measured holdout gain (f06_measure.py; log notebooks/v3_proof/out/f06_a.log)
Protocol: LightGBM (lr 0.05, 31 leaves, min_data 200, 400 rounds, seed 42) on [logit(p2) + feature set], trained on
TUNE pairs with p2 >= 1e-3 (1.16M; pairs below keep p2 - they hold 58 of ~360k tune positives); tune predictions
cross-fitted in 2 folds by md5(lid); holdout model trained on all tune; decision rule (expected-F0.5, fit_decision)
refitted on the corrected tune p; holdout scored once with _group_arrays semantics. Vocabularies/offset set come from
the fit-group samples only. Baseline = raw p2 through the same decision refit (reproduces the v2 manifest exactly).

| variant | holdout macro F0.5 | gain (pts) | TP | FP | in-candidate FN | India | US |
|---|---|---|---|---|---|---|---|
| v2 baseline (raw p2) | 0.971512 | - | 354,880 | 1,090 | 5,026 | 0.966651 | 0.974774 |
| + number signature only (17 feats) | 0.972751 | +0.124 | 355,570 | 909 | 4,313 | 0.967859 | 0.976033 |
| **+ full signature (46 feats: number + name ops + vocab log-odds + address ops)** | **0.973839** | **+0.233** | 356,275 | **731** | **3,678** | 0.969154 | 0.976983 |
(A's post-hoc number features gave +0.056; the signed offset-set membership + "which number changed" index doubles it,
and the name-op features add as much again.) Top gains in the full model: logit_p2 0.77, first_off_idx 0.10,
off_in_S 0.05, edit_lo_max 0.02, first_off 0.011, n_repl 0.009.

## Finding 6: which v2 holdout errors the signature explains / fixes (f09_explain.py; A's categories)
"Signature" = sole number offset in S (>= +3) OR name-edit log-odds >= 3 (Latin R names only; the non-Latin rows
below are counted by NaN semantics and are not signature cases). "Fixed" = full-signature model (finding 5).

| v2 FP tag (A) | FPs | fixed | carry signature | fixed among signature |
|---|---|---|---|---|
| name_core equal, numbers differ (nudge) | 416 | 285 | 211 | 198 |
| name other noise | 273 | 128 | 98 | 92 |
| R address empty | 170 | 41 | 6 | 5 |
| name very different | 98 | 34 | 4 | 3 |
| non-Latin R name | 80 | 34 | (n/a) | - |
| one word added/dropped | 36 | 21 | 21 | 19 |
| name and numbers equal | 17 | 5 | 6 | 4 |
| **total** | **1,090** | **548** | 346 (Latin) | **321 (93%)** |

| v2 decision-FN tag (A) | FNs | recovered | carry signature |
|---|---|---|---|
| R address empty | 1,864 | 57 | 5 |
| name_core equal, numbers differ (truncation / +-1 true noise) | 1,402 | **891** | 52 |
| name other noise | 810 | 395 | 20 |
| name very different | 435 | 98 | 1 |
| non-Latin R name | 233 | 99 | - |
| name and numbers equal | 110 | 89 | 0 |
| one word added/dropped | 98 | 62 | 5 |
| **total** | **4,952** | **1,691** | |
The model introduces 189 new FPs (net FP 1,090 -> 731). So the signature explains ~1/3 of v2's FPs (346 carry it,
93% of those are removed) and, by its ABSENCE, rescues true copies whose numbers differ in a non-distractor way
(offset outside S, truncation) - the larger half of the gain. It does not help the R-address-empty ambiguity
(1,864 FNs + 170 FPs: that is "which of several same-name S1s", not distractor-vs-copy) nor pseudo-brand names.

### 5b. Control and "in-model" proxy (log out/f06_b.log)
| variant | holdout macro F0.5 | gain | TP | FP | in-cand FN | India | US |
|---|---|---|---|---|---|---|---|
| logit(p2) + all 83 v2 features (re-stacking control) | 0.971526 | +0.001 | 354,794 | 1,075 | 5,043 | 0.966651 | 0.974798 |
| **logit(p2) + 83 v2 features + 46 signature features** | **0.974062** | **+0.255** | 356,126 | **632** | 3,803 | 0.969517 | 0.977111 |
Re-stacking the v2 features alone gains nothing, so the +0.23..0.26 pts are the signature features' information.
Top gains: first_off_idx 0.15, edit_lo_max 0.03, num_changed 0.03, off_in_S 0.02 (v2 features all < 0.01).
(Peak RSS of this run was 3.1 GB, above the 2.5 GB target; the smaller variants stay at 1.6 GB.)

## Finding 7: per source / country (fit samples; share of pairs with the op)
| kind | country | src | n | exact | case-only | non-Latin | typo | abbrev/trunc | reorder | repl | ins | del | addr empty | number changed | digit trunc/drop | offset in S>=3 | +-1/2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dist_near | India | S2 | 15,404 | 0.001 | 0.000 | 0.161 | 0.045 | 0.151 | 0.063 | 0.372 | 0.502 | 0.094 | 0.000 | 0.838 | 0.009 | 0.621 | 0.165 |
| dist_near | India | S3 | 12,755 | 0.001 | 0.000 | 0.110 | 0.038 | 0.147 | 0.077 | 0.400 | 0.529 | 0.107 | 0.000 | 0.831 | 0.009 | 0.600 | 0.165 |
| dist_near | US | S2 | 26,348 | 0.001 | 0.000 | 0.000 | 0.049 | 0.055 | 0.042 | 0.481 | 0.568 | 0.037 | 0.000 | 0.981 | 0.013 | 0.718 | 0.182 |
| dist_near | US | S3 | 26,897 | 0.001 | 0.000 | 0.000 | 0.044 | 0.055 | 0.042 | 0.482 | 0.568 | 0.038 | 0.000 | 0.980 | 0.013 | 0.725 | 0.183 |
| true | India | S2 | 22,086 | 0.029 | 0.061 | 0.234 | 0.055 | 0.193 | 0.097 | 0.178 | 0.122 | 0.175 | 0.037 | 0.060 | 0.027 | 0.001 | 0.011 |
| true | India | S3 | 23,790 | 0.027 | 0.068 | 0.131 | 0.049 | 0.167 | 0.083 | 0.217 | 0.155 | 0.215 | 0.041 | 0.070 | 0.024 | 0.002 | 0.012 |
| true | US | S2 | 33,226 | 0.062 | 0.122 | 0.000 | 0.047 | 0.117 | 0.077 | 0.201 | 0.162 | 0.255 | 0.049 | 0.090 | 0.040 | 0.001 | 0.010 |
| true | US | S3 | 35,595 | 0.058 | 0.115 | 0.000 | 0.041 | 0.105 | 0.071 | 0.192 | 0.192 | 0.242 | 0.047 | 0.081 | 0.037 | 0.001 | 0.009 |
- **S2 and S3 use the same noise model** (differences <= 0.04 on every op, except more non-Latin names in S2 India
  0.234 vs 0.131). No source-specific features needed.
- The distractor generator is the same in both countries; India distractors change a number less often (0.83 vs
  0.98: many Indian addresses have no house number) and use the India suffix vocabulary (public / llp /
  enterprises / industries / exports / ventures / overseas / infratech) instead of the US one.
- An "exact name" copy (0.1% of distractors vs 3-6% of true copies) is also a (weak) true signal.

