# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Straw Hat Coders
**Team Members:** Venkata Karthik Kandagaddala, Venkata Dharaneswara Reddy Palle, Kuppam Bhargav Reddy (Amrita Vishwa Vidyapeetham (AVV), Bengaluru)
**Final submission:** #20 (27 September 2026) — public leaderboard macro F0.5 **0.986182**; local holdout (India/US) **0.99008**

---

## 1. Executive Summary
We built a staged entity-resolution pipeline:
rule-based normalization → multi-pass blocking (IDF token/trigram passes, sorted-neighbourhood keys, compound-key
passes, a crowded-key pass and a learned candidate cut; **98.3% recall at 8.2 candidates per Source-1 entity**) →
a two-stage LightGBM matcher over 147 hand-built pair features plus out-of-fold "competition" features →
three fine-tuned open-weight cross-encoders (multilingual-e5-small, MIT; bge-reranker-v2-m3, Apache-2.0;
Qwen2.5-1.5B-Instruct with LoRA, Apache-2.0) re-scoring the uncertain pairs → a stage-3 LightGBM combiner with
competition context → a macro-F0.5-optimal decision rule with a per-country prior correction.

Key ideas: (1) reverse-engineering the dataset's **distractor generator** (house numbers nudged by a fixed offset set,
name edits from small vocabularies) into language-independent features; (2) **self-labelled per-country
vocabularies** that transfer to France (absent from training) without any labels; (3) a **leakage-free decision
layer** that maximizes expected macro F0.5 per entity; (4) **loss decomposition on a held-out entity group** to decide
what to work on next (blocking misses vs. rejected true pairs vs. false merges).

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale.** Train: 2.21 M Source-1 (S1) businesses, 10.32 M S2/S3 records; test: 1.73 M S1, 9.97 M S2/S3.
  Every S2/S3 record matches at most one S1; all true pairs share a country; 5.6% of S1 have no match; 3.46 matches per
  S1 on average. Test adds **France (15% of test S1), absent from train**, and has ~23% more records per S1
  (more distractors).
- **Names are repetitive, name + address is nearly unique.** ~50% of S1 share their normalized name with another S1
  (chains; in France templated names such as "lille club" occur hundreds of times), so blocking and matching must use
  addresses; names alone give 46% blocking recall.
- **Noise of true copies:** abbreviations and legal-suffix drift (Pvt/Private, Ltd/Limited, SARL/SAS…), token
  reordering, typos, transliteration from nine Indic scripts, initialisms ("LC" = "Lille Club"), dropped or truncated
  digits, ±1/±2 number typos, replaced street names (8% of true copies), empty addresses (4.4% of true copies).
  97.7% of records with an empty address are true copies of some S1.
- **Distractors are near-duplicates made by a generator** (measured on train, confirmed on French test pairs): one
  address number is nudged **upward by an offset drawn from {+1,+2,+3,+4,+5,+7,+9,+11,+13,+21}** and the name is edited
  from small vocabularies (legal-form swaps such as llc→inc / sarl→snc, inserted words such as "holdings",
  "industries", "groupe"); 91.5% of distractors carry both edits; the generator never empties an address. True copies
  never show this signature, so it is a strong, language-independent signal.
- **Postal codes are almost absent**; addresses must be parsed into house number / unit / street / city / admin parts.

### 2.2 Solution Strategy
**Approach Type:** Hybrid — multi-pass blocking + gradient-boosted matcher with stacking + fine-tuned transformer
cross-encoders + optimization-based decision layer.
**Core Innovation:** turning the data generator's signature and self-labelled per-country vocabularies into features
that generalize to an unseen country, plus a decision layer that maximizes expected macro F0.5.

**Validation protocol (no leakage).** S1 entities are split by md5 bucket of their id into **fit** (30%, two folds for
out-of-fold stacking), **tune** (5%: early stopping, calibration, stage-3 training by 5-fold cross-fitting, decision
rule choice) and **holdout** (5%, scored once per model version). All candidates of an S1 stay in its group. Every
learned table (vocabularies, token log-odds, blocking ranker, cut model) is fitted on the fit group only; the
cross-encoders are fine-tuned on fit-group pairs only. France has no labels: France-specific choices were checked on
France-like holdout slices (shared names, crowded candidate sets), with an unseen-country simulation (fit on India,
score on US and vice versa), and finally on the public leaderboard.

---

## 3. Candidate Generation (Blocking)
- **Normalization first:** Unicode/accent folding, Indic scripts transliterated, legal forms split from the core name
  (hand-written map for English/French plus an Indic legal-form token map learned from train), consonant skeleton and
  phonetic key of the name, address abbreviations expanded (English and French: r./rue, av/avenue, bd, ch, st/saint…),
  numbers extracted with leading zeros removed, address decomposed into house / unit / street / city / admin.
- **Blocking keys used:**
  1. IDF-weighted token pass on country-scoped name + address tokens (document-frequency caps), and a character-trigram
     IDF pass on the core name (typos, transliteration).
  2. Six sorted-neighbourhood keys (name; name+address; address+name; consonant skeleton+numbers; numbers+address;
     address+skeleton) plus a name-only window for records with empty or short addresses.
  3. **Compound-key passes:** 6-character prefix of the space-stripped name × address token, address number × address
     word, unordered pairs of name words, name word × address word, and the whole phonetic name — combinations of
     individually common tokens are rare ("1940" + "fisher").
  4. **Crowded-key pass:** for records with short addresses, number × address-word keys shared by 50–500 businesses
     (dropped by the cap of pass 3) are joined and ranked by name similarity (Jaro-Winkler of core and phonetic names),
     keeping the top 3 — this recovers transliterated names with truncated addresses
     ("vijn injiniyring | 601, mumbai" = "Vision Engineering Pvt Ltd | 601, Maple Classic…, Mumbai").
  5. A linear ranker (logistic, fitted on the fit group) orders each record's candidates; a small LightGBM **cut model**
     keeps "linear top-1 per record OR cut probability ≥ 1e-3".
- **Candidate pairs generated:** 18.2 M in train (8.2 per S1), **19.4 M in test (11.2 per S1)**; the exact set scored
  by the matcher is submitted in `output/candidate_pairs.tsv`.
- **How true matches were not lost:** recall is measured on every train run (a recall gate stops the pipeline below
  94%). The first passes reached 94.5%; compound keys and the learned cut raised it to 98.2% while halving candidates
  per S1 (14.1 → 8.1); the crowded-key pass to **98.28% on train (98.31% on the holdout group)**. The remaining misses
  are mostly records with an empty address and a name shared by dozens of S1, which no key can disambiguate.

---

## 4. Matching Model
**Features used (147 per pair in stage 1):**
- **Name:** rapidfuzz ratio / Jaro-Winkler / token-set / token-sort / partial ratio on normalized name, core name
  (legal form removed) and consonant skeleton; token alignment classes (exact, typo, abbreviation, transliteration,
  legal-form difference, unexplained tokens and their IDF); legal-form and business-type agreement; initialism flags;
  name frequencies (crowded names).
- **Address:** token Jaccard, ratio, Jaro-Winkler and partial ratio; parsed street similarity; house / unit / number
  relations (equal, truncated, digit dropped, signed offset); city agreement.
- **Distractor signature (46):** signed number offsets and membership of the generator offset set, which number
  changed, counts of name/address insert/replace/delete edits and the log-odds of the edited tokens.
- **Self-labelled per-country vocabularies:** edit-token and unexplained-token log-odds learned from pairs labelled by
  model-free rules (distractor = sole number change by a generator offset, 99.5% precise; true = all numbers equal and
  core-name ratio ≥ 0.6, 87–91% precise), per country — India/US from train candidates, France from test candidates.
  No labels are used, and the same self-labelling is applied to training rows, so the features mean the same in every
  country.
- **Blocking evidence and context:** pass flags and scores, cut probability, candidate counts and ranks per record and
  per entity.

**Model type:**
- **Stage 1:** LightGBM (binary, learning rate 0.2, 63 leaves, min 200 rows per leaf, bagging 0.8), early-stopped on
  tune; out-of-fold predictions on the two fit folds for stacking.
- **Stage 2:** LightGBM on the 147 features + 17 competition features (the pair's margin over the record's other
  candidate S1, its share of the record's probability mass, rank and gaps within the entity) + 15 entity features
  (179 in total), trained on the out-of-fold stage-1 predictions.
- **Cross-encoders** (open weights, fine-tuned only on fit-group train pairs, run in free Colab/Kaggle GPU notebooks,
  no hosted API), each re-scoring the pairs whose stage-2 probability is uncertain:
  - `intfloat/multilingual-e5-small` (MIT, 117.7 M params): 2 epochs on 725 k pairs; scores 0.003 < p < 0.997, and,
    reusing the saved model, also 1e-4 < p ≤ 0.003 and 0.997 ≤ p < 0.9999 (3.8 M test+validation pairs in total).
  - `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568 M params): up to 1 M pairs within a 170-minute budget; scores
    0.001 < p < 0.999.
  - `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, 1.54 B params) + LoRA (r = 16, all projection layers) with a
    classification head, prompt "Record A: … / Record B: … / Do A and B describe the same business?": up to 400 k
    pairs within a 150-minute budget; scores 0.01 < p < 0.99.
  - Input text of every model: `business_name | business_address` of both records.
- **Stage 3:** a LightGBM combiner over logit(stage-2 p), the three cross-encoder scores (missing outside a model's
  band) and **competition context for every score** — its rank, gap to the best and gap to the best *other* candidate
  within the S1 entity and within the S2/S3 record, candidate counts, and flags for an empty address and the source.
  Trained on the tune group with 5-fold cross-fitting by entity (out-of-fold predictions fit the decision rule); rows
  whose S1 lies in the holdout group are excluded from training. The context targets the largest in-candidate error —
  records with an empty address whose name is shared by several businesses (77% of rejected true pairs). It won on the
  holdout, on every France-like holdout slice (+0.0002 … +0.0006) and in the unseen-country simulation
  (India→US +0.0003, US→India +0.0006).
- All models are MIT or Apache-2.0 licensed and far below 8 B parameters.

**Threshold selection method:**
1. **One-to-one assignment:** each S2/S3 record can only go to its best-scoring S1.
2. **Decision rule chosen on the tune group** by realized macro F0.5 (rule fitted on one half of tune, chosen on the
   other): either a global probability threshold or an **expected-F0.5 rule** that, per S1, keeps the top-k candidates
   maximizing expected F0.5 under isotonic-calibrated probabilities, with a model of blocking misses and a calibrated
   singleton probability. The final model uses the expected-F0.5 rule.
3. **Per-country prior correction on test:** the test set contains ~23% more records per S1 (more distractors), so
   the match prior is lower than in training. For India and the US an EM estimate of the test prior shifts the logits
   (−0.67 / −0.61). France has no labels; its EM estimate (−0.56) improved the public score, and stronger fixed shifts
   improved it further (−1.0, −1.5), so the final submission uses **−1.5 for France**. This is the one setting chosen
   on the public leaderboard; it only removes French matches (the removed pairs are 15–24× richer in the distractor
   house-number offsets than the accepted ones).

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro):** local holdout (5% of train S1, never used for fitting; India/US) and public leaderboard.

| Version | Main change | Holdout | Public |
|---|---|---|---|
| v1 rule baseline | best blocking candidate + threshold | 0.8075 | 0.749 |
| v2 | LightGBM + stacking + expected-F0.5 decision | 0.9715 | 0.961 |
| v3 | compound-key blocking + learned cut, signature features | 0.9876 | 0.977648 |
| v3 + French adaptation | transductive French vocabulary | — | 0.981323 |
| v4 | self-labelled per-country vocabularies | 0.9879 | 0.982191 |
| v5 + e5 cross-encoder | stage-2 entity features, initialisms, e5-small stage 3 | 0.98918 | 0.984041 |
| + bge-reranker | two cross-encoders | 0.98929 | 0.984453 |
| + Qwen2.5-1.5B | three cross-encoders | 0.98937 | 0.985004 |
| #14 | stage-3 competition context, wider e5 band (India/US) | 0.98971 | 0.985379 |
| #16 | crowded-key blocking pass (recall 98.31%), full retrain | 0.99008 | 0.985878 |
| #17 | competition-context stage 3 for France too | 0.99008 | 0.985971 |
| #18 | France prior correction, EM estimate (−0.56) | 0.99008 | 0.986121 |
| #19 | France prior correction −1.0 | 0.99008 | 0.986161 |
| **#20 (final)** | France prior correction −1.5 | **0.99008** | **0.986182** |

- **Loss decomposition** (holdout, before the last blocking pass; total loss 0.0106): true pairs never generated by
  blocking 0.0069 (the ceiling with perfect decisions was 0.99306), true pairs present but rejected 0.0032, false
  merges 0.0005. This is why the late work went into blocking recall and into stage-3 competition context rather than
  into the pair classifier.
- **Local vs public:** a leaderboard probe with all French rows emptied showed India+US public ≈ local holdout, so the
  gap is France. From the public scores France is estimated at ≈0.92 (v3) → ≈0.95 (self-labelled vocabularies) →
  ≈0.96–0.97 (cross-encoders, context model, prior correction).
- **Common false positives (wrong merges):** same-name siblings at the same house number in another street (frequent
  in France, where names are templated and house numbers small — a label-free estimate shows such pairs are true only
  24–39% of the time in France vs. 80–100% in India/US); "same address, different name" records.
- **Common false negatives (missed matches):** records with an **empty address** and a name shared by several S1:
  which business they belong to cannot be read from the text; and the ~1.7% of true pairs never generated by blocking
  (mostly the same population).

---

## 6. Conclusion
Error decomposition drove every step: blocking recall was the first ceiling (0.980 with perfect decisions at 94.5%
recall), the unseen French domain the second (found with a leaderboard probe and reduced without labels through
self-labelled vocabularies), and text understanding the third (small multilingual cross-encoders gave the largest late
gains, especially in France). The final gains came from giving the combiner competition context, from a targeted
blocking pass for transliterated names with truncated addresses, and from correcting the French match prior. Lessons:
measure the ceiling before optimizing the model, validate on a separate entity group (and on slices that resemble the
unseen domain), and exploit the data-generating process when it is observable.

---

## 7. Other Relevant Information
- **Data:** only the provided train/test TSVs. No external data, registries, geocoding, web lookups or hosted LLM APIs.
  Abbreviation maps and stop lists are hand-written or learned from the training data.
- **Models and licences:** LightGBM (MIT); multilingual-e5-small (MIT, 117.7 M); bge-reranker-v2-m3 (Apache-2.0,
  568 M); Qwen2.5-1.5B-Instruct (Apache-2.0, 1.54 B) — all open weights run offline, fine-tuned only on the provided
  training data, each ≤ 8 B parameters.
- **Compute:** a 12-thread / 15 GB CPU laptop for the whole pipeline (DuckDB + LightGBM); free Google Colab T4 and
  Kaggle T4×2 notebooks for cross-encoder fine-tuning and scoring.
- **Reproducibility:** fixed seeds (42), deterministic LightGBM, fixed row order in stage 3; every run writes a manifest
  (git commit, config, metrics). `code/business_entity_resolution/README.md` gives the exact end-to-end commands; the
  final output is produced by `run_x.sh` and the France prior correction (`configs/variant_x_predict_fr15.yaml`).

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `src/ber/` (package `ber`, CLI entry point `ber`: prepare, learn-maps, normalize,
block, block3, learn-selfvocab, features, train, predict, validate-output), `configs/` (base config and the final
variant configs), `ce/` (cross-encoder bundle exports, Colab/Kaggle notebooks, stage-3 combiner, per-country merge),
`tests/` (unit tests + an end-to-end smoke test checked by the organizer validator), `README.md`, `requirements.txt`.
Every function has a docstring describing inputs and outputs.

### B. Additional Results
- Architecture diagram: `final_architecture.png`.
- Decision log with the evidence for every design choice (D-001 … D-032) and the full submission history are kept
  in the team repository and can be shared on request.
