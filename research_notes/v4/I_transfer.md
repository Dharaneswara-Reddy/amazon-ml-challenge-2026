# v4 research I: generalizing to an unseen country (simulated: train India only, evaluate on US)

Scripts: `notebooks/v4_transfer/` (common.py, t01_tokens.py, t02_vocab.py, t03_arms.py). DuckDB temp and scratch
parquet: `artifacts/v4_tmp/`. Research only, no change to `code/business_entity_resolution/`.
Findings are appended as each number is measured.

## Protocol
- Data: v3 feature files (147 FEATURES) `feat_v3_train_{fit,tune,holdout}.parquet`; country of the S1 entity from
  `norm_train_s1.parquet` (blocking never crosses countries, so a rival lid always has the rid's country).
- Model: single-stage LightGBM, `ber.model.DEFAULT_PARAMS` + learning_rate 0.2, seed 42, 6 threads, early stopping
  (50 rounds) on the in_group rows of the SOURCE country's tune group. Training rows = in_group rows of **fitA**
  (md5 buckets 0-14) of the source country (memory/time budget; the same subset in every arm).
- Decision: `ber.decision.fit_decision` (expected-F0.5 rule; the tune scores of both rules are close in v3) fitted on
  the SOURCE tune group with `_group_arrays` semantics (all rows of tune incl. rival-context rows scored).
- Score: macro F0.5 over the **US entities of the holdout group** (66k entities), `_group_arrays` semantics restricted
  to one country: entities = holdout US S1 ids (n_true from gt, incl. entities with no candidate), pairs = rows of
  the holdout file whose lid is such an entity and whose rid's best lid (over all holdout-file rows, context incl.)
  is too. FP / FN are pair counts (FN includes true matches missing from the candidates).
- Uncertain share = share of best-per-right pairs of the scored set with 0.05 < raw p < 0.95.

## Honest "unseen US" features (what the model would see for a new country)
Label-learned inputs of the v3 features:
1. token log-odds table (`logodds_*.json`, textdiff): `nd_l_logodds_max/_sum`, `nd_r_logodds_max/_sum`;
2. signature edit vocab (`sigvocab_*.json`): `sg_ins_lo_max/_sum`, `sg_repl_lo_max/_sum`, `sg_rtgt_lo_max`,
   `sg_del_lo_min`, `sg_edit_lo_max`, `sg_a_ins_lo_max`, `sg_a_del_lo_max`;
3. blocking linear ranker (`block_score`) and cut model (`cut_p`), plus the candidate set itself.
(1)-(2) are RECOMPUTED exactly (same `pair_ops`/`align_tokens` code) with vocabularies learned from India fit pairs
only (out-of-fold fitA<->fitB for the India training rows, full India fit for India tune and all US rows).
(3) is NOT redone: candidates, `block_score` and `cut_p` of US rows still come from models that saw US labels, so
the measured A-B gap is a LOWER bound on the true unseen-country gap (France also has 3.7 vs 1.6 candidates per right).
Name-frequency / IDF / crowd features are computed from the unlabeled data of the country itself, so they are
legitimately available for France and are left as they are.

## Results
### R0: recomputation check and self-labelled vocabularies (t01, t02)
- Token extraction (pair_ops / align_tokens once per pair, 9.0M pairs, ~15 min on 6 procs) + vocab application
  reproduces the feature-file vocab columns EXACTLY (13/13 columns, 300k US holdout rows, match 1.00000 with the
  pipeline's `sigvocab_fit` / `logodds_fit`). So the recomputed "unseen" columns are exactly what the pipeline would
  produce with another vocabulary.
- Self-label rule, no labels: distractor = `sg_sole_off_in_s = 1` (sole changed number moved by +3,+4,+5,+7,+9,+11,
  +13,+21); true = all address numbers equal (`sg_num_changed = 0 AND l_only = 0 AND r_only = 0 AND num_eq >= 1`).
  Measured on fit pairs (labels used only for this check): distractor rule precision **99.4% India / 99.5% US**
  (187k / 308k pairs); "true" rule only **72% / 70%** true (768k / 1.28M pairs; equal numbers also hold for other
  businesses in the same building). Still, the US self-labelled vocab (no US labels) ranks exactly the generator's
  vocabulary on top: inserted riverside, eastgate, lakeside, northside, midtown, uptown, ... (lo ~3.4-3.7 vs 5.5-7
  with true labels, the contamination compresses the scale), replaced llc>holdings, inc>holdings, corp>holdings,
  lp>group. India true-label vocab: 49k tokens; US self: 22k; US true: 58k.

- **A** (in-domain: train India+US fitA, pooled vocab, decision on India+US tune): US holdout F0.5 **0.98723**, FP 556, FN 7,584, uncertain 0.0331; +EM shift: 0.98722 (delta +0.022, FP 575, FN 7,532); iters 339, source tune F0.5 0.98585

- **B** (unseen US: train India fitA, India-only vocab (OOF), India tune; US rows India vocab): US holdout F0.5 **0.95850**, FP 7,379, FN 8,316, uncertain 0.0843; +EM shift: 0.95676 (delta +0.225, FP 7,990, FN 7,855); iters 209, source tune F0.5 0.98388

- **B_zero** (unseen US: train India fitA, India-only vocab (OOF), India tune; US rows India vocab; eval vocab None zero=['nd_l_logodds_max', 'nd_l_logodds_sum']): US holdout F0.5 **0.92415**, FP 16,986, FN 7,337, uncertain 0.1494; +EM shift: 0.91491 (delta +0.614, FP 19,653, FN 6,605); iters 209, source tune F0.5 0.98388

- **B_pooledvocab** (unseen US: train India fitA, India-only vocab (OOF), India tune; US rows India vocab; eval vocab None zero=[]): US holdout F0.5 **0.97858**, FP 2,477, FN 8,080, uncertain 0.0460; +EM shift: 0.97830 (delta +0.083, FP 2,599, FN 7,924); iters 209, source tune F0.5 0.98388

- **C2a_USS** (unseen US: train India fitA, India-only vocab (OOF), India tune; US rows India vocab; eval vocab holdout_USS zero=[]): US holdout F0.5 **0.97356**, FP 3,614, FN 8,186, uncertain 0.0647; +EM shift: 0.97310 (delta +0.127, FP 3,841, FN 7,935); iters 209, source tune F0.5 0.98388

- **C2b_MRG** (unseen US: train India fitA, India-only vocab (OOF), India tune; US rows India vocab; eval vocab holdout_MRG zero=[]): US holdout F0.5 **0.97365**, FP 3,585, FN 8,210, uncertain 0.0646; +EM shift: 0.97321 (delta +0.125, FP 3,807, FN 7,962); iters 209, source tune F0.5 0.98388

- **C2_oracle_US** (unseen US: train India fitA, India-only vocab (OOF), India tune; US rows India vocab; eval vocab holdout_US zero=[]): US holdout F0.5 **0.98142**, FP 1,830, FN 8,090, uncertain 0.0443; +EM shift: 0.98133 (delta +0.070, FP 1,912, FN 7,956); iters 209, source tune F0.5 0.98388

- **C2c** (self-labelled vocab everywhere: India rows INS (OOF), US rows USS (no labels)): US holdout F0.5 **0.97891**, FP 1,596, FN 10,290, uncertain 0.0554; +EM shift: 0.97894 (delta +0.021, FP 1,600, FN 10,259); iters 214, source tune F0.5 0.98318

### R1: the gap and the vocabulary remedies (A, B, C2)
- **A (in-domain) US holdout 0.98723** (FP 556, FN 7,584 of which most are candidates-missing, uncertain 3.3%).
- **B (unseen US) 0.95850**: -2.87 pts, FP x13 (7,379), uncertain best-per-right share 8.4% vs 3.3%. This reproduces
  the France symptoms (France public ~0.92, uncertain share 5.6% vs 1.6-1.9% for India/US in research G). The
  per-country EM prior shift makes it WORSE (0.95676): it moves US UP (delta +0.23) because the extra mid-scored
  pairs look like extra positives, i.e. EM cannot tell "shifted prior" from "worse discrimination".
- Decomposition: with the same B model, US rows scored with the pooled India+US vocab (B_pooledvocab) give 0.97858,
  with a true-label US vocab (C2_oracle_US) 0.98142. So ~2.0-2.3 of the 2.87 pts are the vocabulary features alone;
  the remaining ~0.6 pt is the model not having seen US data at all.
- Zeroing the vocab columns (the "cheap approximation") is far too pessimistic: 0.92415 (FP 16,986). A model trained
  with vocab features expects distractor tokens to carry log-odds; blanking them makes every distractor look clean.
  Do not use zeroing to reason about France.
- **C2a: same B model, US rows with the self-labelled US vocab (no US labels) 0.97356** (+1.51 pts, FP 3,614).
  C2b (India vocab overlaid by US self vocab) 0.97365 - the same.
- **C2c: self-labelled vocab EVERYWHERE (India training rows too, OOF; US rows US-self vocab): 0.97891** (+2.04 pts,
  71% of the A-B gap; FP 1,596, FN 10,290, uncertain 5.5%). Training on self-labelled vocab makes the feature
  distribution identical in construction for the source and the target country, which beats plugging a
  self-labelled target vocab into a model trained on a true-label vocab (C2a) by 0.5 pt. EM shift: +0.00004.

- C3 pseudo labels on US fitA (labels used ONLY for this report): {"n_rows": 1631841, "n_pos": 654840, "n_neg": 861480, "pos_precision": 0.985538452140981, "neg_precision": 0.9986314249895528, "coverage": 0.9292081765319048, "true_pos_covered": 0.9500613137482647}

- **C3** (self-training on B: India fitA + confident pseudo-labelled US fitA pairs): US holdout F0.5 **0.96255**, FP 6,443, FN 7,951, uncertain 0.0330; +EM shift: 0.96192 (delta +0.151, FP 6,643, FN 7,755); iters 167, source tune F0.5 0.98338

### R2: pseudo-label self-training (C3, on top of B)
- US fitA pairs (1.63M, no labels) scored by the B model; pseudo 1 = p > 0.98 and best for its right record and not a
  sole S-offset; pseudo 0 = p < 0.02 or (sole S-offset and p < 0.5). Coverage 92.9%; pseudo-positive precision
  98.55%, pseudo-negative precision 99.86% (labels used only to report this). Retrained on India fitA + these rows.
- **C3 US holdout 0.96255** (+0.40 pt over B; FP 6,443). The uncertain share collapses to 3.3% (= in-domain level)
  but the errors stay: the ~1.5% wrong pseudo-positives are exactly the distractors the source model already
  accepts (vocab-blind), and self-training teaches the model to accept them confidently. Confidence is not accuracy:
  on France, a drop of the uncertain share after self-training would NOT indicate a fix.

- **C2d** (self-labelled vocab v2 (pos rule + core_ratio>=0.6) everywhere: India INS2 (OOF), US USS2): US holdout F0.5 **0.98071**, FP 1,951, FN 8,243, uncertain 0.0467; +EM shift: 0.98039 (delta +0.069, FP 2,051, FN 8,084); iters 243, source tune F0.5 0.98341

- C3s pseudo labels on US fitA (labels used ONLY for this report): {"n_rows": 1631841, "n_pos": 626512, "n_neg": 895759, "pos_precision": 0.998446957121332, "neg_precision": 0.9975674260599112, "coverage": 0.9328549779053229, "true_pos_covered": 0.9208677257089356}

- **ASELF** (in-domain cost of self-labelled vocab: train India+US fitA, self vocab per country everywhere): US holdout F0.5 **0.98759**, FP 512, FN 7,508, uncertain 0.0318; +EM shift: 0.98758 (delta +0.021, FP 526, FN 7,462); iters 400, source tune F0.5 0.98589

### R3: a stricter self-label rule, and the in-domain cost of self-labelled vocab
- Self-label v2: positive = all numbers equal AND `core_ratio >= 0.6` (87% / 91% true among pairs with a name edit,
  vs 37% / 50% without the name condition); negatives unchanged (sole S-offset, 99.5%).
- **C2d: self-labelled vocab v2 everywhere (India rows INS2 OOF, US rows USS2): 0.98071** (FP 1,951, FN 8,243,
  uncertain 4.7%) = **77% of the A-B gap recovered**, within 0.07 pt of plugging a TRUE-label US vocab into B
  (0.98142). With v2 the self-labelled vocab is practically as good as a labelled one.
- **ASELF: in-domain (India+US train) with self-labelled vocab v1 everywhere: 0.98759 vs A 0.98723** (FP 512 vs 556).
  Replacing the label-learned vocabularies by self-labelled ones costs nothing in-domain (+0.04 pt, noise level), so
  the switch is safe for India/US and is what makes the vocab features transfer.

- **C3s** (self-training on C2c: India fitA + confident pseudo-labelled US fitA pairs): US holdout F0.5 **0.97994**, FP 1,463, FN 9,993, uncertain 0.0302; +EM shift: 0.97992 (delta +0.023, FP 1,478, FN 9,959); iters 217, source tune F0.5 0.98241

- **C1v** (India-only train, vocab log-odds columns dropped): US holdout F0.5 **0.97745**, FP 2,371, FN 9,021, uncertain 0.0609; +EM shift: 0.97698 (delta +0.082, FP 2,606, FN 8,678); iters 288, source tune F0.5 0.98297

- **C1vf** (India-only train, vocab log-odds + name-frequency columns dropped): US holdout F0.5 **0.97576**, FP 2,613, FN 8,929, uncertain 0.0621; +EM shift: 0.97550 (delta +0.086, FP 2,756, FN 8,724); iters 273, source tune F0.5 0.98282

### R4: combination and leaner models (C3s, C1)
- **C3s: pseudo-label self-training on top of C2c** (US fitA pseudo labels from the C2c model): pseudo-positive
  precision 99.84% (vs 98.55% from B), coverage 93.3%. US holdout **0.97994** (+0.10 pt over C2c, FP 1,463,
  FN 9,993). With transferable vocab features the pseudo labels are clean, and self-training then adds a little.
- **C1v: drop the 13 vocab columns** (India-only train): **0.97745** (FP 2,371). Better than keeping India-learned
  vocab (B 0.95850) and than a target vocab plugged into a true-vocab model (C2a 0.97356), but 0.3 pt below
  self-labelled vocab everywhere (C2d 0.98071).
- **C1vf: also drop l_name_freq / r_name_freq / r_core_l_freq: 0.97576** - the frequency features help transfer
  (they are computed from the country's own unlabelled data); keep them.

## Results table (US holdout, 66,096 US entities; FP/FN = pairs; uncertain = best-per-right 0.05<p<0.95)

| arm | description | US F0.5 | +EM shift | FP | FN | uncertain | gap recovered |
|---|---|---|---|---|---|---|---|
| A | in-domain, India+US train, label vocab | **0.98723** | 0.98722 | 556 | 7,584 | 3.3% | ref |
| ASELF | in-domain, self-labelled vocab v1 everywhere | **0.98759** | 0.98758 | 512 | 7,508 | 3.2% | (no in-domain cost) |
| B | unseen US: India train, India label vocab | **0.95850** | 0.95676 | 7,379 | 8,316 | 8.4% | 0% |
| B_zero | B, US vocab columns zeroed (approximation) | 0.92415 | 0.91491 | 16,986 | 7,337 | 14.9% | - |
| B_pooledvocab | B model, US rows with pooled India+US label vocab | 0.97858 | 0.97830 | 2,477 | 8,080 | 4.6% | (leaky ref) |
| C2_oracle_US | B model, US rows with TRUE-label US vocab | 0.98142 | 0.98133 | 1,830 | 8,090 | 4.4% | (leaky ref) 80% |
| C1v | India train, vocab columns dropped | 0.97745 | 0.97698 | 2,371 | 9,021 | 6.1% | 66% |
| C1vf | C1v + name-frequency columns dropped | 0.97576 | 0.97550 | 2,613 | 8,929 | 6.2% | 60% |
| C2a | B model, US rows with self-labelled US vocab | 0.97356 | 0.97310 | 3,614 | 8,186 | 6.5% | 52% |
| C2b | B model, India vocab overlaid by US self vocab | 0.97365 | 0.97321 | 3,585 | 8,210 | 6.5% | 53% |
| C2c | self-labelled vocab v1 everywhere (India OOF, US self) | 0.97891 | 0.97894 | 1,596 | 10,290 | 5.5% | 71% |
| **C2d** | **self-labelled vocab v2 everywhere** | **0.98071** | 0.98039 | 1,951 | 8,243 | 4.7% | **77%** |
| C3 | B + pseudo-label self-training | 0.96255 | 0.96192 | 6,443 | 7,951 | 3.3% | 14% |
| C3s | C2c + pseudo-label self-training | 0.97994 | 0.97992 | 1,463 | 9,993 | 3.0% | 75% |

Gap recovered = (arm - B) / (A - B), A - B = 2.87 pts. "+EM shift" = production per-country EM prior shift (C4a).
C4a (EM prior shift per country): never helps in the unseen setting; in B it hurts 0.17 pt because worse
discrimination is read as a higher prior (delta +0.23). C4 oracle (decision refit on US tune with true labels) and
ASELF2 (in-domain cost of v2 vocab) were stopped for RAM by the coordinator; not measured.

## Recommendation for France (what to implement in the ber pipeline)
1. **Self-labelled, per-country vocabularies instead of label-learned ones (highest value, measured 77% of the
   unseen-country gap, no in-domain cost).** In `learn_logodds_stage`, learn BOTH `sigvocab` and the textdiff token
   log-odds with self-labels instead of GT labels: negative = `sg_sole_off_in_s = 1`, positive = all address numbers
   equal (`sg_num_changed = 0 AND sg_num_l_only = 0 AND sg_num_r_only = 0 AND sg_num_eq >= 1`) AND
   `core_ratio >= 0.6`; other pairs ignored. Learn one vocab PER COUNTRY from that country's candidate pairs
   (train countries: from train candidates; France: from the France TEST candidates - unlabeled, transductive, legal:
   no external data, no labels), and apply each pair's own-country vocab at feature time, for training rows too (the
   training rows must see self-labelled vocab or the features shift between train and France, C2a vs C2c: -0.5 pt).
   The pair-level inputs (`sg_sole_off_in_s`, number counts, `core_ratio`) are vocabulary-free, so this is a
   two-pass feature build: pass 1 computes those + the edit tokens, pass 2 fills the 13 vocab columns.
   Keep the OOF discipline for training rows (fitA vocab for fitB rows etc.; labels are not used, but it mirrors
   what was measured).
2. **Then optionally pseudo-label self-training on France pairs** (C3s: +0.10 pt on top of the self vocab): score
   France test pairs, pseudo 1 = p > 0.98 and best for its right record and not a sole S-offset, pseudo 0 = p < 0.02
   or sole S-offset with p < 0.5, add to the stage-1 training set, retrain. Only after (1): on top of a vocab-blind
   model it amplifies the model's own errors (C3: +0.4 pt only, confident-but-wrong).
3. **Do not rely on the EM prior shift for France**: in the unseen setting it never helped and hurt 0.17 pt when the
   features were degraded (delta +0.23 read "less separable" as "more positives"). Consider disabling it for a
   country whose uncertain share is far above train's, or cap |delta|.
4. **Do not drop features to "be leaner"**: dropping the vocab columns (C1v 0.97745) is worse than self-labelled
   vocab; dropping name-frequency columns costs a further 0.17 pt.
5. Diagnostics: the uncertain share tracked the damage in every arm without self-training (8.4% B, 6.5% C2a, 4.7%
   C2d vs 3.3% in-domain) - use France's uncertain share (5.6% in v2) as the proxy after implementing (1); it is NOT
   a valid proxy after self-training (C3 has 3.3% at 0.9626).

Caveats: B keeps candidates, `block_score` and `cut_p` from models that saw US labels, so the real unseen gap is
larger than 2.87 pts; the remedies above address the vocab part only (France also keeps 3.7 vs 1.6 candidates per
right after the cut model - a blocking-side remedy would be separate). Training used fitA only (1.07M India pairs /
2.7M pooled) with single-stage LightGBM lr 0.2, so absolute numbers are below the v3 two-stage pipeline; the deltas
are what transfers. Single seed; differences under ~0.05 pt are noise level.

- **C4o_C2d** (oracle calibration: C2d model, decision refit on US tune with true labels): US holdout F0.5 **0.98131**, FP 1,173, FN 9,752, uncertain 0.0467; +EM shift: 0.98130 (delta +0.005, FP 1,177, FN 9,739); iters 243, source tune F0.5 0.00000

### R5: C4 upper bound (measured after the table)
- **C4o_C2d: C2d model, decision rule refit on the US tune group with TRUE US labels: 0.98131** (+0.06 pt over C2d,
  FP 1,173, FN 9,752). Even a perfect target-country calibration adds only 0.06 pt once the features transfer, so
  target-country calibration (EM shift, pseudo-labelled decision refit) is not worth building for France; the gap
  is in discrimination (features), not in calibration. ASELF2 (in-domain cost of the v2 self-label rule) was not
  rerun (RAM limit); ASELF (v1 rule) showed no in-domain cost.
