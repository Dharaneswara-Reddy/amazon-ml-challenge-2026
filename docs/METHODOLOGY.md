# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Straw Hat Coders
**Team Members:** Venkata Karthik Kandagaddala, Venkata Dharaneswara Reddy Palle, Kuppam Bhargav Reddy (Amrita Vishwa Vidyapeetham (AVV), Bengaluru)
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We built a staged entity-resolution pipeline: rule-based normalization → multi-pass blocking (IDF token/trigram
passes, sorted-neighbourhood keys, compound-key passes and a learned candidate cut, 98.2% recall at ~8 candidates per
Source-1 entity) → a two-stage LightGBM matcher over 147 hand-built pair features and out-of-fold "competition"
context → a transformer cross-encoder (multilingual-e5-small, MIT, fine-tuned only on the provided train pairs)
a bge-reranker-v2-m3 cross-encoder (Apache-2.0) and a Qwen2.5-1.5B-Instruct LoRA classifier (Apache-2.0) re-scoring
the uncertain pairs → a macro-F0.5-optimal decision rule. Key innovations: reverse-engineering the dataset's
**distractor generator** (house numbers nudged by a fixed offset set, name edits from small vocabularies) into features,
**self-labelled per-country vocabularies** that transfer to France (unseen in training) without labels, and a
**leakage-free decision layer** tuned for macro F0.5 on a separate entity group.

---

## 2. Methodology

### 2.1 Problem Analysis
- **Scale.** Train: 2.21 M Source-1 (S1) businesses, 10.32 M S2/S3 records; test: 1.73 M S1, 9.97 M S2/S3.
  Every S2/S3 record matches at most one S1; all true pairs share a country; 5.6% of S1 have no match; 3.46 matches per
  S1 on average. Test adds **France (15% of test S1), absent from train**, and has ~23% more records per S1.
- **Names are repetitive, name + address is nearly unique.** ~50% of S1 share their normalized name with another S1
  (chains; in France templated names such as "lille club" occur hundreds of times), so blocking and matching must use
  addresses; names alone give 46% blocking recall.
- **Noise of true copies:** abbreviations and legal-suffix drift (Pvt/Private, Ltd/Limited, SARL/SAS…), token
  reordering, typos, transliteration from nine Indic scripts, initialisms ("LC" = "Lille Club"), dropped or truncated
  digits, ±1/±2 number typos, replaced street names (8% of true copies), empty addresses (4.4% of true copies).
- **Distractors are near-duplicates made by a generator** (measured on train, confirmed on French test pairs): one
  address number is nudged **upward by an offset drawn uniformly from {+1,+2,+3,+4,+5,+7,+9,+11,+13,+21}** and the
  name is edited from small vocabularies (legal-form swaps such as llc→inc / sarl→snc, inserted words such as
  "holdings", "industries", "groupe", "holding"); 91.5% of distractors carry both edits; the generator never empties an
  address. True copies never show this signature, so it is a strong, language-independent signal.
- **Postal codes are almost absent**; addresses must be parsed into house number / street / city / admin units.

### 2.2 Solution Strategy
**Approach Type:** Hybrid — multi-pass blocking + gradient-boosted matcher with stacking + transformer cross-encoder +
optimization-based decision layer.
**Core Innovation:** turning the data generator's signature and self-labelled, per-country vocabularies into
features that generalize to an unseen country, plus a decision layer that maximizes expected macro F0.5.

Validation protocol (no leakage): S1 entities are split by md5 bucket into **fit** (30%, split into two folds for
out-of-fold stacking), **tune** (5%, early stopping, calibration and decision-rule choice) and **holdout** (5%, scored
once per model version). Every learned table (vocabularies, token log-odds, blocking ranker and cut model) is fitted on
the fit group only.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:**
  1. IDF-weighted token pass on country-scoped name + address tokens (document-frequency caps), and a character-trigram
     IDF pass on the core name (typos, transliteration).
  2. Six sorted-neighbourhood keys (name; name+address; address+name; consonant skeleton+numbers; numbers+address;
     address+skeleton) plus a name-only window for records with empty or short addresses.
  3. **Compound-key passes (v3):** 6-character prefix of the space-stripped name × address token, address number ×
     address word, unordered pairs of name words, name word × address word, and the whole phonetic (consonant-skeleton)
     name — combinations of individually common tokens are rare ("1940" + "fisher").
  4. **Crowded keys (final, D-030):** for right records with short addresses, number x address-word keys shared by
     50-500 businesses (dropped by the df cap above) are joined and ranked by name similarity (top 3), recovering
     transliterated names with truncated addresses ("vijn injiniyring, 601, mumbai"). Holdout recall 98.22% -> 98.31%.
  5. A linear ranker (logistic, fitted on the fit group) orders each right record's candidates; a small LightGBM
     **cut model** keeps "linear top-1 per right record OR cut probability ≥ 1e-3".
- **Candidate pairs generated:** 17.9 M in train (8.1 per S1), 18.9 M in test (10.9 per S1); submitted in
  `output/candidate_pairs.tsv`.
- **How true matches were not lost:** recall is measured on every train run (a recall gate stops the pipeline below
  94%). v2 passes reached 94.5%; the compound-key passes and learned cut raised it to **98.2% on train (98.2% on the
  holdout group)** while *halving* candidates per S1 (14.1 → 8.1). The remaining misses are mostly records with an
  empty address and a name shared by many S1, which no key can disambiguate.

---

## 4. Matching Model
**Features used (147 per pair):**
- Name features: rapidfuzz ratio / Jaro-Winkler / token-set / token-sort / partial ratio on normalized name, core name
  (legal form removed) and consonant skeleton; token-alignment classes (exact, typo, abbreviation, transliteration,
  legal-form difference, unexplained tokens and their IDF); legal-form and business-type agreement; **initialism**
  flags; name frequencies (crowded names).
- Address features: token Jaccard, ratio, Jaro-Winkler and partial ratio on the address; parsed street similarity;
  house/unit/number relations (equal, truncated, digit dropped, offset); city agreement.
- **Distractor-signature features (46):** signed number offsets and membership of the generator offset set, which
  number changed, counts of name/address insert/replace/delete operations and the log-odds of the edited tokens.
- **Self-labelled per-country vocabularies:** edit-token and unexplained-token log-odds learned from pairs labelled by
  model-free rules (distractor = sole number change by a generator offset, 99.5% precise; true = all numbers equal and
  core-name ratio ≥ 0.6, 87–91% precise), per country — India/US from train candidates, France from test candidates
  (no labels are used; the same self-labelling applies to training rows, so the features mean the same in every country).
- Blocking evidence and context: pass flags and scores, cut probability, candidate counts and ranks per right record and
  per entity.

**Model type:**
- Stage 1: LightGBM (binary, learning rate 0.2, 63 leaves), early-stopped on tune.
- Stage 2: LightGBM stacked on stage-1 out-of-fold predictions and "competition" features (the pair's margin over the
  right record's other candidates, its rank within the entity, entity-level sums and gaps).
- Stage 3: **cross-encoder** `intfloat/multilingual-e5-small` (MIT, 117.7 M parameters) fine-tuned for 2 epochs on
  725 k fit-group pairs (on a free Colab T4, in-notebook, no API); it re-scores the 1.7 M uncertain pairs
  (0.003 < p < 0.997). A second cross-encoder, `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568 M; a multilingual reranker),
  fine-tuned on 1 M fit-group pairs on a free GPU notebook, re-scores 2.2 M pairs (0.001 < p < 0.999). A third model, `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0, 1.54 B) with
  LoRA adapters (r = 16) and a classification head, is fine-tuned on 400 k fit-group pairs presented as
  "Record A / Record B / same business?" and scores the pairs with 0.01 < p < 0.99. A small LightGBM combines
  logit(stage-2 p) with the three cross-encoder scores, cross-fitted on tune (5 folds by S1).
- Final stage 3 (D-029): each score also gets **competition context** — its rank, gap to the best and gap to the best
  other candidate within the Source-1 entity and within the Source-2/3 record, candidate counts, and flags for an empty
  right address and the source. This targets the largest in-candidate error: records with an empty address whose name
  is shared by several businesses (77% of rejected true pairs). e5 also re-scores the pairs just outside its first band
  (1e-4 < p <= 0.003, 0.997 <= p < 0.9999) with the saved fine-tuned model. Holdout 0.98937 -> 0.98971; in an
  unseen-country simulation (stage 3 fitted on India, scored on US and vice versa) the context gained +0.0003/+0.0006.
  For France the final output keeps the #12 stage 3: label-free checks showed the new model would add French pairs
  carrying the distractor generator's house-number offsets. Because every candidate pair and every decision stays
  within one country, the two outputs are merged per country.
- All models are MIT/Apache-2.0 licensed and far below 8 B parameters.

**Threshold selection method:** one-to-one assignment (each right record to its best S1), then a decision rule chosen
on the tune group by realized macro F0.5: either a global probability threshold or an **expected-F0.5 rule** that,
per S1, keeps the top-k candidates maximizing expected F0.5 under isotonic-calibrated probabilities (with a model of
blocking misses and a calibrated singleton probability). On test, a per-country EM prior shift corrects for the higher
distractor rate (India/US); it is disabled for France, where it was measured harmful in an unseen-country simulation.

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro), local holdout (5% of train S1, never used for fitting):**

| Version | Main change | Holdout | Public |
|---|---|---|---|
| v1 rule baseline | best blocking candidate + threshold | 0.8075 | 0.749 |
| v2 | LightGBM + stacking + expected-F0.5 decision | 0.9715 | 0.961 |
| v3 | compound-key blocking + learned cut, signature features | 0.9876 | 0.977648 |
| v3 + French adaptation | transductive French vocabulary | — | 0.981323 |
| v4 | self-labelled per-country vocabularies | 0.9879 | 0.982191 |
| v5 | stage-2 entity features, initialisms | 0.9882 | not uploaded |
| v5 + cross-encoder | multilingual-e5-small stage 3 | 0.98918 | 0.984041 |
| v5 + two cross-encoders | + bge-reranker-v2-m3 | 0.98929 | 0.984453 |
| v5 + bge + Qwen | bge-reranker + Qwen2.5-1.5B LoRA | 0.98939 | 0.984756 |
| v5 + three cross-encoders | e5-small + bge-reranker + Qwen2.5-1.5B | 0.98937 | 0.985004 |
| #14 | + stage-3 competition context, wider e5 band (India/US); France as #12 | 0.98971 | 0.985379 |
| **final (#16)** | + crowded-key blocking pass (recall 98.31%), full retrain | **0.99008** | **0.985878** |

- Holdout loss decomposition of #12 (0.98937): true pairs never generated by blocking cost 0.0069 (ceiling with
  perfect decisions 0.99306), rejected true pairs 0.0032, false matches only 0.0005.
- A leaderboard probe (France rows emptied) showed India+US public ≈ local holdout, so the local→public gap is
  essentially France, which improved from ≈0.92 to ≈0.95 with the self-labelled vocabularies.
- **Common false positives (wrong merges):** same-name siblings at the same house number in another street (frequent in
  France, where names are templated and house numbers small); "same address, different/random name" records
  (generator aliases) at very high model confidence.
- **Common false negatives (missed matches):** right records with an **empty address** and a name shared by several S1
  (65% of rejected true pairs on holdout; the text cannot tell which S1), and the ~1.8% of true pairs never generated
  by blocking (mostly the same population).

---

## 6. Conclusion
Careful error decomposition drove every step: blocking recall was the first ceiling (0.980 with perfect decisions on
94.5%-recall candidates), the unseen French domain the second (measured with a leaderboard probe and fixed without
labels through self-labelled vocabularies), and text understanding the third (a small multilingual cross-encoder gave
the largest late gain). Lessons: measure the ceiling before optimizing the model, validate on a separate entity group,
and exploit the data-generating process when it is observable.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — `src/ber/` (package `ber`, CLI entry point `ber`), `configs/base.yaml`,
`ce/` (cross-encoder bundle export, Colab/Kaggle notebooks, stage-3 combiner), `tests/` (unit tests + an end-to-end
smoke test validated by the organizer validator), `README.md` (exact reproduce sequence), `requirements.txt`.
Reproduce: `ber prepare → learn-maps → normalize → block (sample, ranker, train, test) → block3 → learn-selfvocab →
features → train → predict`; then `ce/export_bundle.py` → `ce/ce_colab.ipynb` (GPU) → `ce/stage3.py` →
`ber predict --stage3`. Every run writes a manifest (git commit, config, metrics) under `artifacts/runs/`.

### B. Additional Results
- Architecture diagram: `docs/diagrams/final_architecture.png` (source `final_architecture.dot`).
- Blocking recall by pass and the cut rules: `research_notes/v3/B_blocking_recall.md`, `E_proof_end_to_end.md`.
- Distractor generator analysis: `research_notes/v3/F_generator_signature.md`.
- France analyses and the unseen-country simulation: `research_notes/v4/`, `research_notes/v5/`.
- Every decision with its evidence: `docs/DECISIONS.md` (D-001 … D-030); submission history `submissions/log.md`.
