# Decision log

Append-only. Each record says what was decided, why, what was rejected, and the evidence. To change a decision, add a
new record that supersedes the old one; never edit history. Status: **active** · **proposed** (v2, awaiting approval) ·
**superseded**. Referenced from [`DESIGN_V2.md`](DESIGN_V2.md) and recorded in each run's `manifest.json`.

---

### D-001 · DuckDB + Parquet cache as the data engine — active
- **Decision:** convert TSVs once to zstd Parquet; do joins, windows, and aggregates in DuckDB with a memory limit and
  disk spill; do Python-side work (normalization, rapidfuzz) on streamed Arrow batches.
- **Why:** 24 M records / 2.4 GB of text on a 15 GB laptop with 3–9 GB free.
- **Rejected:**
  - pandas: loads everything into RAM
  - polars: in-memory by default
  - Spark: setup cost, overkill for one machine
- **Evidence:** `prepare` 60 s at 664 MB peak; normalization 11 min at under 1 GB.

### D-002 · Country is a hard partition for candidate generation — active
- **Decision:** candidates are only ever generated within the same country label.
- **Why:** it cuts the search space about 2× and removes cross-country false positives at no recall cost.
- **Rejected:** country as a soft feature only.
- **Evidence:** 100% of 200 k sampled true pairs share the country.
- **Risk noted:** a mislabelled country would be unrecoverable. None observed.

### D-003 · Each noisy record is assigned to at most one Source-1 entity (best candidate) — active
- **Decision:** after scoring, a Source-2/3 record keeps only its highest-probability entity. An entity may keep many
  records.
- **Why:** it's an exact property of the data, and removes rival-entity false positives.
- **Rejected:**
  - no constraint: more false positives
  - Hungarian / global bipartite matching: entities are one-to-many, and research shows global weight maximisation can
    add false positives in ER
- **Evidence:** 0 of 7.64 M matched records appear under two entities.

### D-004 · Grouped validation by Source-1 entity via md5 buckets; learned statistics only from the fit group — active (layout revised in v2)
- **Decision:** assign entities to groups by md5(entity_id) mod 100. Fit, tune (threshold / early stopping), and
  holdout (final score) are disjoint. Any statistic learned from labels (token log-odds, abbreviation maps, calibration)
  comes only from fit (out-of-fold within fit) or tune.
- **Why:** pairs of one entity share strings and group features, so a pair-level split leaks. md5 is deterministic and
  order-independent.
- **Rejected:**
  - random pair split: leaks
  - 5-fold CV of everything: 5× compute for about ±0.0006 of extra precision
- **v2 layout:** fitA 0–14, fitB 15–29, tune 30–34, holdout 35–39. v1 layout was fit 0–9, tune 10–14, holdout 15–19.

### D-005 · Blocking keys combine name and address — active
- **Decision:** sorted-neighbourhood keys like `name|sorted-address-words`, `address|name`, `skeleton|house-numbers`;
  the token pass uses name and address tokens.
- **Why:** half of Source-1 names are shared by other entities, so name-only keys drown the true match among namesakes.
- **Rejected:** name-only keys (v1 blocking).
- **Evidence:**
  - name-only passes: 41–58% recall
  - composite 6-key union: 92.8% (2% sample)
  - full train after ranking: 94.0%, vs 46% for v1 blocking

### D-006 · Block v2: add char-3-gram TF-IDF and name-only passes; replace hand-weighted ranking with a learned ranker; choose K from the recall curve — proposed
- **Decision:**
  - add a chunked sparse top-k TF-IDF pass over name_core + street + city, and a name-only pass for records with
    empty or one-component addresses
  - rank the union with a small LightGBM trained on fit-group candidates
  - K = the smallest cap within 0.2 points of the recall plateau on tune
- **Why:** blocking caps the score (oracle on today's candidates is 0.977). Misses concentrate in transliterated names
  with truncated or empty addresses. The hand-weighted score loses about 0.4 points of reachable recall.
- **Rejected:**
  - raising K alone: the research shows recall gaps are key-coverage problems
  - dense embedding ANN: pretrained-model status is unconfirmed and CPU cost is higher
- **Evidence:** Sparkly (VLDB 2023) TF-IDF top-10 recall 86–99.96%; meta-blocking literature (Papadakis/JedAI);
  forensics miss examples.

### D-007 · Pairwise matcher = LightGBM — active
- **Decision:** gradient-boosted trees (LightGBM, MIT) on engineered pair features.
- **Why:**
  - it already reaches 0.9555 holdout
  - CPU-fast
  - license-safe
  - interpretable importances for error analysis
- **Rejected (for now):** transformer cross-encoders and LLMs (see D-014).
- **Evidence:** Kaggle Foursquare 2022 7th place 0.9935 IoU with LightGBM only; our v1 run.

### D-008 · Two-stage stacking with competition and cross-source features — proposed
- **Decision:** stage-2 LightGBM on stage-1 probability plus context features:
  - rank and margin within the noisy record's and the entity's candidates
  - the entity's confident-candidate counts per source
  - agreement between an entity's Source-2 and Source-3 candidates

  Trained on out-of-fold stage-1 predictions.
- **Why:** many errors are relative ("lost to a rival", "near-duplicate beside the true match") and single-pair
  features can't see them.
- **Rejected:** hand-written post-processing rules (brittle, untunable).
- **Evidence:** Foursquare winners' rank / margin features; stage-2 branch simulations +0.4–1.6 points.

### D-009 · Decision rule chosen on tune: global threshold vs per-entity expected-F0.5 — proposed
- **Decision:** implement both, pick the one with the higher tune F0.5, report it on the holdout. Expected-F uses
  isotonic-calibrated probabilities and scores the empty set with P(no true match).
- **Why:** macro F0.5 per entity rewards predicting nothing for doubtful singletons and trading a borderline 4th match
  against the confidence of the first three.
- **Rejected:** fixed 0.5 threshold; per-country thresholds (France has no labels).
- **Evidence:** Jansche 2007 and Dembczyński GFM (exact expected-F set selection); stage-2 branch simulation.

### D-010 · Prior-shift correction for test — proposed
- **Decision:** estimate the test positive rate among candidates with Saerens EM on test predictions, rescale the
  probabilities before the decision rule, and log the estimate.
- **Why:** test has 5.8 noisy records per entity vs 4.68 in train, in every country. If the matches per entity are
  unchanged, a larger share of candidates are distractors and the model is over-confident on test.
- **Rejected:** assuming no shift (the rule dropped 0.807 → 0.749 public).
- **Evidence:** forensics composition; Saerens et al. 2002; BBSE 2018.

### D-011 · Address admin units (state / région / département) split out and down-weighted; no French admin map — proposed
- **Decision:** parse the admin component separately. Similarity features focus on house number + street + city; admin
  equality is its own low-weight feature. No hand-written département↔région table.
- **Why:** French records use a région on one side and a département or nothing on the other. France is 15% of test and
  was under-matched (20.6% empty). French postal standards don't use the région at all.
- **Rejected:** hand-mapping 13 régions / 96 départements (gray zone under the no-external-data rule, and unnecessary).
- **Evidence:** forensics samples; research note on French address standards.

### D-012 · Indic legal-form abbreviations learned from train pairs — proposed
- **Decision:** build the transliterated-token → canonical-token map (e.g. "pra li" → "private limited") by aligning
  tokens of true pairs in the fit group; keep only high-support, high-precision mappings.
- **Why:** "प्रा. लि." and similar forms survive transliteration as unmatched tokens, and those names dominate blocking
  misses.
- **Rejected:**
  - hand-writing forms for 9 scripts (slow, incomplete)
  - an external transliteration dictionary (not allowed)
- **Evidence:** forensics blocking-miss examples; rules allow maps derived from provided data.

### D-013 · Distractor-aware features: token alignment classes and learned token log-odds — proposed
- **Decision:**
  - classify each name token difference as exact / typo / abbreviation / transliteration / legal form / unexplained
  - add the IDF mass and learned log-odds (out-of-fold, fit group) of unexplained tokens
  - add house-number relation classes and relative numeric distance
- **Why:** 26.5 k of the rule's 29.5 k false positives are near-duplicate distractors that differ by one swapped or added
  word or a nudged number, and generic similarities can't tell that from typo noise.
- **Rejected:** only more generic similarity scores.
- **Evidence:** importance-weighted Jaccard +5 F1; learned token weighting (HierGAT) up to +8.7 F1 over Ditto; Magellan
  vs Ditto on dirty data 37 vs 86 F1.

### D-014 · Transformer / LLM reranker deferred — active
- **Decision:** don't build a cross-encoder or LLM stage now. Revisit only if stage-1 + stage-2 plateau below 0.975
  holdout **and** a local pilot scores 50 k pairs fast enough to cover the uncertain band (1–3 M pairs) in under 2 h.
- **Why:**
  - CPU throughput for cross-encoders is unproven (no primary benchmark found)
  - the organizers haven't confirmed that pretrained models are allowed
  - GBDT-only reached 0.9935 in the closest competition
- **If revived:** multilingual-e5-small (MIT) or paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0), both ≤ 118 M params.

### D-015 · No external data, gazetteers, geocoders, or libpostal — active
- **Decision:** everything learned comes only from the provided TSVs. Hand-written rules only for generic
  abbreviations and legal forms.
- **Why:** disqualification rule. libpostal is trained on OpenStreetMap, which counts as external data.

### D-016 · Transliteration with anyascii (+ anusvara fix) and a consonant skeleton — active
- **Decision:** anyascii for all non-Latin scripts; a consonant skeleton key for transliteration-robust comparison.
- **Why:** it's script-generic (9 scripts seen) and measured best.
- **Rejected:** indic_transliteration schemes: 67.6–68.6 vs 75.4 mean token-set ratio on 2,711 Indic↔Latin true
  pairs; Tamil 46–48 vs 63.
- **Evidence:** `docs/normalization_findings.md`. The skeleton lifted Indic-pair name ratio from 73.8 to 92.1 with
  hard negatives flat (58.4 → 59.2).

### D-017 · Rule baseline retired — active
- **Decision:** the rule is kept only as a sanity baseline.
- **Why:** 0.807 holdout / 0.749 public vs LightGBM 0.9555 holdout.

### D-018 · Fast iteration loop on samples before full runs — active
- **Decision:**
  - blocking ideas are tested on a 2% noisy-record sample (about 1 minute per variant)
  - model ideas are tested on the tune and holdout groups
  - only winners get full-data runs
- **Why:** a full blocking run takes 40–60 min on the laptop. The sample recall numbers matched full-data recall (95.2%
  sample vs 94.0% full).

### D-019 · Every heavy stage runs per hash bucket; big inputs are streamed Parquet views — active
- **Decision:** no global window, sort, or aggregate over more than ~10 M rows; per-bucket processing; views instead
  of in-memory tables; DuckDB spill capped at 30 GB.
- **Why:** 5 incidents: quadratic join, 60 GB spill, writer out of memory ×2, test features 30 GB spill.

### D-020 · Decisions and experiments are logged in the system — active
- **Decision:**
  - this file records every design decision
  - each run's `manifest.json` records commit, config, decision IDs, per-stage counts / times / memory, metrics, and
    the chosen decision parameters
  - `docs/EXPERIMENTS.md` gets one row per run, with the delta vs the best
  - stages log *why* they filtered (chosen K and its recall, chosen threshold and its tune score, EM prior)
- **Why:** you asked for every decision to be traceable; it also stops us from repeating failed ideas.

---

### Implementation status — 25 Sep 2026, commit 3c7f168+
- **Implemented:** D-006 (q3 + name-only passes, linear ranker, K from recall curve), D-008 (stage 2 wired into
  `ber train`), D-009 (rule chosen on tune halves via `decision.choose_decision`), D-010 (EM prior shift in
  `ber predict`), D-011 (address components, learned admin vocabulary), D-012 (`ber learn-maps`), D-013
  (`ber.textdiff` + `ber learn-logodds`), D-019 (bucketed features), D-020 (manifest decision IDs + EXPERIMENTS.md).
- **Verified:** 168 tests, incl. an end-to-end smoke run of the whole v2 chain on synthetic data through the organizer
  validator (`--check-ids`).
- **Not yet run on real data.** The runbook is `scripts/run_v2.sh prep | train | predict`, and training waits for
  the user's approval.
- **Found and fixed while integrating:**
  - `compare_decisions` picked the rule by holdout score. Replaced with tune-half selection (D-004).
  - v1 never fed `skel_ratio`/`skel_jw` to the model; they were missing from the feature list.
  - Concatenated fold files duplicated rival pairs; they are now deduplicated with the in-group row preferred.

## v3 decisions (26 Sep 2026; evidence in research_notes/v3/, plan docs/PLAN_V3.md)

### D-021 Distractor-signature features
The generator nudges one address number UP by an offset drawn uniformly from {+1..+5, +7, +9, +11, +13, +21} and edits
the name from small vocabularies (legal-form swap, inserted word); true copies show typos, abbreviations, dropped words
and digit truncation (F_generator_signature.md). `ber.signature` diffs each pair and emits 46 features (offset-set
membership, which number changed, edit-token log-odds). Vocabularies are learned from labelled FIT-group candidate
pairs, out-of-fold for fitA/fitB. Measured +0.23 holdout points as an add-on to v2 (FP 1,090 → 731).

### D-022 Blocking v3: compound keys + learned cut
74% of blocking misses share no single rare token; combinations are rare. Five compound-key IDF passes (px, hs, n2,
na, phx) on top of the v2 candidates, then "legacy-linear top-1 per right OR cut-GBDT p ≥ 1e-3" (GBDT trained on the
FIT group). Measured on the full train split: holdout recall 94.54% → 98.20% at 7.6 candidates per S1 (v2: 14.1)
(E_proof_end_to_end.md). The cut runs per right bucket to bound spill.

### D-023 Per-country EM prior shift; learning rate 0.2
EM prior estimates on v2 test differ by country (India −0.56, US −0.46, France −0.32 logit vs the pooled −0.49), so the
pooled shift over-corrects France (G_public_gap_france.md); the shift is now estimated per test country. LightGBM
learning rate 0.1 → 0.2: 27% faster and lower logloss on a 1.2M-row benchmark (C_speed.md).

### D-024 Transductive vocabulary adaptation for France (submission #5)
Rule-labelled France test pairs (number signature + model agreement; rules 99.5-99.99% precise on train) give a French
signature vocabulary and token log-odds; France feature rows are recomputed with them (`ber adapt`, `predict --adapted`).
Public 0.977648 → 0.981323 (France ≈0.921 → 0.946). Superseded by D-025.

### D-025 Self-labelled per-country vocabularies everywhere; no EM shift for France (submission #6, best)
In a train-India / test-US simulation, learning ALL vocabularies from self-labels (distractor = sole generator offset;
true = all numbers equal and core-name ratio ≥ 0.6) and applying each pair's own-country table to training rows too
recovered 77% of the unseen-country gap with no in-domain cost; the EM prior shift never helped an unseen country
(research_notes/v4/I_transfer.md). `ber learn-selfvocab`; `features.self_vocab: true`; `decision.shift_skip_countries:
[France]`. Local holdout 0.9879; public 0.982191 (France ≈0.950).

### D-026 France pseudo-label self-training — rejected
Adding rule/model-labelled France test pairs to stage-1 training (`ber pseudo`, `train --extra-fit`) scored public
0.981916 (#7), below #6; not used.

### D-027 Stage-2 entity features, initialism features, admin features dropped (v5)
Entity-level context from stage-1 scores (rank within the S1 entity, gaps, the right record's second choice) measured +0.0003
holdout (research_notes/v5/K_india_us.md); initialism flags for French abbreviated names; admin_eq/admin_diff dropped because
French region vs departement values make them misleading (measured neutral in-domain). Local holdout 0.9882.

### D-028 Cross-encoder stage 3
Organizers confirmed open-weight MIT/Apache models <= 8B, fine-tuned only on provided data, run offline, are allowed.
intfloat/multilingual-e5-small (MIT, 117.7M) fine-tuned on 725k fit-group pairs (Colab T4) re-scores pairs with
0.003 < p < 0.997; a LightGBM stage 3 on [logit p2, ce] cross-fitted on tune: holdout 0.98815 -> 0.98918 (+0.0010).
Larger models (BAAI/bge-reranker-v2-m3 568M Apache-2.0; Qwen2.5-1.5B-Instruct LoRA Apache-2.0) prepared for Kaggle.

### D-029 Competition context in stage 3, wider e5 band, per-country model choice (submission #14)
Holdout loss decomposition of #12 (0.98937): true pairs never generated by blocking 0.0069 (ceiling 0.99306),
rejected true pairs 0.0032 (77% have an empty right address; the true S1 is the record's top candidate but shares its
name with other S1), false matches 0.0005. Measured and adopted:
- `stage3.py --ctx`: every score (logit p2 and each cross-encoder) also gets its rank, gap to the best and gap to the
  best other candidate within the S1 and within the right record, candidate counts, and empty-address / source flags.
  Holdout 0.98937 -> 0.98965; unseen-country proxy (stage 3 fitted on one country, scored on the other): India->US
  +0.00028, US->India +0.00060.
- e5 re-scores the pairs just outside its first band (1e-4 < p <= 0.003, 0.997 <= p < 0.9999; `export_ext.py`,
  `ce_colab_e5_ext.ipynb`, reusing the saved fine-tuned model): with --ctx holdout 0.98971.
- Stage-3 training rows exclude rival rows whose S1 is in the holdout group; frame rows are ordered (reproducible).
- France keeps the #12 stage 3 (no --ctx, first-band scores): on French test pairs the new models ADD pairs whose house
  numbers carry the distractor generator's offsets (6.9% of additions vs 0.14% of accepted pairs) and remove few of
  them, the opposite of the #11 -> #12 change that raised the public score. Decisions never cross countries, so the
  two outputs are merged per country (`merge_by_country.py`).
Measured and rejected: stage-2 input features in stage 3 (+0.00008 in-domain, -0.00016 US->India), stage-3 tree size /
learning rate / min_data variants (all within +-0.00003), doubling stage-1 training data (15% -> 30% of S1: +0.00012
stage-1 holdout, so the 60% spare buckets are not worth a rebuild), blocking re-run (earlier research: ceiling
~98.5% recall vs 98.2%). The Qwen adapter was not saved by the first Kaggle run, so Qwen could not be extended.
