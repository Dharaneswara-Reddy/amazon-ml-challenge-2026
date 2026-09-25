# D — Methods to push 0.961 → ~0.99, with France generalization

Research only. Written incrementally (append after each finding, per instructions — if this run is cut off, keep
everything already appended). Companion to `docs/DESIGN_V2.md`, `docs/DECISIONS.md`, `docs/data_profile.md`,
`docs/normalization_findings.md`, and `research_notes/Business entity resolution methods/*.md`.

Current state: local holdout 0.9715 (v2, submission 02), public 0.961. Top teams ≥0.982, leaders ~0.988.

Status: COMPLETE.

---

## Q1 — France / unseen-domain generalization

### What we already know (measured, from `docs/normalization_findings.md` and `docs/data_profile.md`)

- France is test-only, 15% of test, never seen in train. `docs/normalization_findings.md` already sampled French test
  rows and found: S2 addresses are upper-case with abbreviations `R`/`R.`→rue, `AV`, `All.`, `Ch.`, `St-`, and
  `N°`/`Nº`/`#`; legal forms SARL/SAS/SASU/EURL/SCI/SA/EI/SNC (sometimes dotted, `S.A.R.L.`); `(France)` is injected
  noise; `Cie`/`Ets` are abbreviations; région and département are interchanged on the two sides of a pair.
- `docs/DECISIONS.md` D-011 (proposed) already splits the admin component (state/région/département) out of the
  address and down-weights it, explicitly rejecting a hand-built département↔région table as unnecessary and
  gray-zone.
- French addresses per the French postal service (La Poste) and general references: number + street type + street
  name, then postal code + commune, then country; CEDEX ("Courrier d'Entreprise à Distribution EXceptionnelle") is a
  business-mail routing suffix, not part of the physical address; in Paris/Lyon/Marseille the last two digits of the
  postal code encode the arrondissement (e.g. `75011` = Paris 11e, sometimes written "Paris 11e" instead of a
  postal code). — [Postal codes in France, Wikipedia](https://en.wikipedia.org/wiki/Postal_codes_in_France),
  [France address format, Address Guard](https://addressguard.io/address-format/france/),
  [Addresses in France, Endereco](https://www.endereco.de/en/general_news/endereco-know-addresses-in-france-the-golden-rules-for-your-parcel-shipping-how-to-and-fun-facts/)

### What in the current pipeline plausibly breaks on France

1. **Accent stripping** — low risk. French relies on accents (é/è/à/ç/î/ô) but accent-folding for comparison (already
   done for Indic transliteration per D-016) is a safe, well-tested normalization for cross-source matching (typos
   commonly drop/add accents); no evidence it should be treated differently for French.
2. **Elision/apostrophes** — real risk, not yet addressed. French uses `l'`, `d'`, `qu'`, `j'`, `n'`, `s'`, `c'`, `m'`,
   `t'` fused to the next word ("l'Atelier", "Café de la Gare" vs "Café d'la Gare"-style noise). If `normalize.py`'s
   tokenizer splits on apostrophes without folding these, "l'Atelier" and "Atelier" tokenize differently (extra
   single-letter token `l`), diluting token-overlap and Jaccard features. This is symmetric with contractions in
   English ("O'Brien") but the *elision* pattern (leading, not embedded) is specifically French/Italian.
3. **Function-word stopwords "de/du/la/le/les/des"** — same risk class as English "the/of": if `name_core`/`street`
   token sets keep these, they inflate token-overlap similarity for unrelated businesses that happen to share
   "de la" (precision risk, and F0.5 punishes false positives 2×) and dilute it for genuine matches where only one
   side keeps the article. Hand-coding a French stopword list is allowed (hand-written rule) but is exactly the kind
   of "country-specific hard-coding" the rules warn against generalizing badly — see proposal below for a
   language-agnostic alternative.
4. **House-number suffix "bis"/"ter"/"quater"** — real risk. These denote a *distinct* building at the same numeric
   address (12, 12 bis, 12 ter are three different doors). If the current house-number extractor either (a) strips
   trailing letters and treats "12 bis" == "12" == "12 ter", it will manufacture false positives between different
   addresses (precision risk); if it (b) fails to parse "12BIS" as number+suffix at all, it will fail to recognize
   "12 bis" and "12bis" as the same address (recall risk). Note English/Indic addresses have an analogous pattern
   ("123", "123A", "123B" unit letters), so this is a generalizable extraction rule, not French-only.
5. **`cedex` and trailing routing digits** — real risk if unhandled. `"75008 PARIS CEDEX 08"` — the `08` after CEDEX
   is a distribution-office code, not a second postal code or house number. If a number-comparison feature picks up
   both `75008` and `08` as address numbers, it can create spurious numeric mismatches against a matched record that
   omits CEDEX, or spurious "close numeric distance" against an unrelated record. `cedex` is a single, closed-class
   token (not a gazetteer) and is safe to hand-code as junk, same class as existing `PMB`/`PO Box` handling.
6. **Arrondissement ordinals ("11e", "Paris 11e")** — the trailing `e`/`ème` ordinal marker on a bare number can
   confuse a digit-only extractor. Generalizes to English "11th", already a candidate for a generic ordinal-suffix
   regex rather than a French-specific one.
7. **Postal code detection** — `docs/normalization_findings.md` notes postal codes are already largely absent /
   unreliable (`extract_postal` only fires on a keyword like "PIN"); France's plain 5-digit postal code attached to
   the commune name won't be keyword-marked either, so it is already being conservatively ignored — consistent
   behavior across countries, not a new gap.
8. **Country-conditioned learned artifacts are the biggest hidden risk, not the string rules**:
   - **Learned token log-odds** (D-013) are estimated out-of-fold on the *fit group*, which is India+US only. French
     tokens ("boulangerie", "sarl", "rue") were never seen during that estimation, so the feature is simply
     unpopulated/default for essentially all French pairs — not actively misleading, but it silently removes one of
     the model's stronger distractor-detection signals exactly where the model is least confident otherwise (France
     has no labeled fit signal for the classifier to correct on either). Recommend adding a companion
     `token_logodds_coverage` (share of the pair's tokens for which log-odds are known) feature so the model can
     learn to lean less on log-odds and more on generic string similarity for low-coverage (French) pairs; this is
     a 1–2 hour feature addition, not a redesign.
   - **State/admin abbreviation maps** (MH↔Maharashtra, TX↔Texas) are learned from India/US pairs and will not fire
     for French régions/départements — this is already the intended behavior under D-011 (admin down-weighted, no
     French admin map), so it is a known and accepted gap, not a bug.
   - **Absolute term/name frequency** — any feature using a *raw* count of how often a token/name appears in the
     corpus (rather than a per-country relative rate) will be dominated by India/US volumes (2.2M/1.7M S1 entities
     each) and will misjudge rarity for France's smaller, differently-distributed 15% slice. This is the general
     version of the log-odds risk above.

### Proposed normalizations (generic/rule-based, not France-only, hand-written or derived from provided data)

- **Elision folding**: a small hand-written closed list of French elided articles/pronouns (`l'`, `d'`, `qu'`, `j'`,
  `n'`, `s'`, `c'`, `m'`, `t'`) stripped when they are a token prefix before an apostrophe — same status as the
  existing alias-marker list (D-012 precedent: small, hand-written, closed-class tokens are already accepted practice
  in this pipeline).
- **Frequency-based stopwording instead of a hand-built per-language list**: compute token document-frequency
  *per country* over the pool of that country's `name_core`/address tokens across **train+test combined** (using
  test's own unlabeled records is allowed — it's provided data, not external data), and drop tokens whose relative
  frequency exceeds a threshold (e.g. top 0.5% by DF) **and** length ≤ 4 characters from token-overlap-style features
  (keep them in edit-distance-style features, which are less sensitive to function words). This automatically
  surfaces "de/la/le/du/des" for France, "the/of" for US, common Hindi/Gujarati function words for India — all from
  the same generic rule, with no per-language table to maintain, and it directly satisfies the brief's ask for
  "country-agnostic" normalization.
- **Generic house-number-suffix extraction**: extend the house-number parser to always split a token like `12BIS`,
  `12 bis`, `123A`, `123-B` into `(number, suffix)`, and add a low-weight **suffix-agreement feature** (equal /
  one-side-missing / different) alongside the existing number-relation feature — instead of silently stripping or
  keeping the suffix fused to the number. This is one parser change, testable immediately on the existing hard-negative
  and true-pair samples in `docs/normalization_findings.md`'s harness.
- **Generic ordinal-suffix stripping**: a regex `\d+(st|nd|rd|th|er|ère|e|ème)\b` applied before numeric comparison,
  covering English and French ordinals with one rule.
- **`cedex` as a junk/routing token**: strip `cedex` and an immediately following 1–2 digit code from the address
  before house-number/postal extraction — same treatment class as existing PMB/PO Box handling, not a gazetteer.
- **French legal-form and abbreviation entries**: add SARL, SAS, SASU, EURL, SA, SCI, SNC, EI, Sté, Ets, Cie
  (dotted and undotted forms) to the existing hand-written legal-form/`type_words` canonicalization map, and `r`/`r.`,
  `av`/`av.`, `bd`, `pl`, `all`, `ch`, `st-` to the street-abbreviation map — same mechanism as the existing India/US
  entries, just new rows in an existing hand-written table (a few hours; low risk, D-011's admin work already implies
  this is in scope).

### Country-agnostic features to prefer / risky features to avoid

- **Prefer**: relative token rarity (IDF) recomputed **per country** from the pool of that country's own records
  (train+test), not looked up from a single global or train-only table — this gives "SARL"/"rue" a correctly low IDF
  weight for France even with zero labeled French pairs.
- **Prefer**: structural/positional features that don't depend on lexical content at all — house-number relation
  class, relative numeric distance, token-count/length agreement, script mix, alignment-based counts (exact/typo/
  abbreviation/transliteration/legal-form/unexplained) — these transfer by construction because they don't encode
  what a French word "means," only how two strings relate.
- **Avoid / down-weight for cross-country transfer**: absolute frequency or count features; token log-odds treated
  as always-informative (add the coverage-indicator companion feature above instead of removing log-odds outright,
  since it's still useful signal for the 85% of test that is India/US); any state/admin abbreviation equality feature
  used at full weight (D-011 already down-weights this); any regex tuned to one country's postal-code shape (India's
  6-digit PIN vs France's 5-digit code) — keep postal detection keyword-gated and conservative as it already is,
  rather than adding a digit-count heuristic that would misfire on France.

---

## Q2 — Transductive / self-training on test

All of the following are explicitly allowed under the competition rules: the test TSVs are provided data, so
unsupervised/transductive use of *unlabeled* test records (not any external source) is fair game. General background:
self-training/pseudo-labeling generates labels for unlabeled target-domain data from a source-trained model, then
retrains on the confident subset; the well-documented failure mode across the domain-adaptation literature is that
pseudo-labels are noisy and errors compound if fed back uncritically, so every serious method adds a confidence
filter. — [Kong et al. 2022, energy-based pseudo-label constraint](https://arxiv.org/pdf/2208.12885),
[general UDA pseudo-labeling framing](https://arxiv.org/pdf/2105.13502)

Ranked by cost/risk for a ~1-day CPU budget:

1. **Per-country EM prior shift (cheapest, safest, do this).** D-010 already implements Saerens-Latinne-Decaestecker
   EM globally for test. The brief specifically asks whether to split it per country — yes: run the same EM procedure
   independently on the India / US / France subsets of test predictions instead of pooling them. Rationale: the
   measured noisy-records-per-entity shift is already country-dependent in the forensics (India 5.82, US 5.76, France
   5.53 vs train's blended 4.68), and France additionally has worse blocking quality (never-seen domain), which
   plausibly shifts the *candidate* positive rate differently than India/US even if true match-count-per-entity is
   similar. Implementation is a `groupby(country)` around the existing EM call — hours, not days, and the pipeline
   already logs "why" (D-020), so add "EM prior per country" as a logged value and sanity-check France's estimate
   against India/US rather than assuming it's identical. Low risk because EM only rescales already-calibrated
   probabilities; it cannot introduce new false positives beyond what miscalibration already risked.
2. **Bias-corrected calibration as an alternative/complement to plain EM.** Alexandari et al. (2020) show
   Maximum-Likelihood label-shift estimation with **Bias-Corrected Temperature Scaling (BCTS)** — temperature scaling
   extended with per-class bias terms — is competitive with or beats plain EM for label-shift adaptation, especially
   when the base classifier's calibration is imperfect (which GBDT probabilities often are without care). Since we
   have no France labels to *fit* a temperature/bias per country, the practical use is: fit BCTS on tune (India/US),
   apply the same transform to France test predictions, and compare against EM-only as a robustness check, not a
   replacement. — [Alexandari, Kundaje, Shrikumar, "Maximum Likelihood with Bias-Corrected Calibration is
   Hard-to-Beat at Label Shift Adaptation," arXiv:1901.06852](https://arxiv.org/pdf/1901.06852)
3. **Conservative pseudo-labeling of high-confidence test pairs (esp. France), fed back as extra weak training
   signal.** Practical instance for this pipeline: take test pairs where stage-1 *and* stage-2 agree with very high
   probability (e.g. both > 0.97), the pair is the unique top candidate for its right-hand record with a large margin
   over the runner-up (mirrors the existing rank/margin features, D-008), and ideally the entity's other matched
   record (other source) also independently supports the same entity (cross-source agreement, already a stage-2
   feature) — i.e. reuse existing high-precision signals as the pseudo-label filter rather than inventing a new one.
   Use these pseudo-positives (and, symmetrically, very-confident-negative rival candidates as pseudo-negatives) to:
   - retrain/augment the **token log-odds table restricted to French tokens** (directly fixes the Q1 coverage gap —
     "SARL," "rue," "boulangerie" get real log-odds once even a few thousand confident French pseudo-pairs exist);
   - optionally add pseudo-labeled France rows to a *held-out-only* retraining pass of stage 1, kept strictly
     separate from the fit/tune/holdout buckets used for the honest metric, so the headline holdout number is never
     computed on data the model was partly fit on.
   Cost: roughly half a day (need a careful pipeline: score test → filter → dedupe/cap per entity → retrain →
   re-validate on holdout to confirm no regression). Risk: real, per the self-training literature above — a
   confidently wrong distractor that gets pseudo-labeled positive is *worse* than a missed true positive under
   F0.5 (precision weighted 2×), so filtering must be strict and should be spot-checked manually on a sample (e.g.
   200 pseudo-labeled France pairs eyeballed) before trusting it. Verdict: worth a careful pilot given France is the
   only slice with literally zero labeled signal, but budget it after the cheaper wins (Q1 rules, per-country EM,
   Q6 features) and cap the time spent if the manual spot-check looks noisy.
4. **Importance weighting of India/US training rows toward "France-like" characteristics** (sparser/shorter address,
   rarer tokens, more junk-free names) approximates covariate-shift correction cheaply (just a `sample_weight` in the
   existing LightGBM `fit()`). Cost is low, but the weighting function is a heuristic proxy for "France-ness" with no
   labels to validate it against, and it risks just re-weighting toward "generally noisier/sparser records" rather
   than anything genuinely France-specific. Worth a quick experiment (compare holdout F0.5 with/without) but rank
   below 1–3.
5. **Domain-adversarial training / learned domain-invariant embeddings** — the standard deep-UDA tool, but it
   requires a differentiable model architecture (not GBDT) and meaningful implementation + tuning time; not practical
   to stand up new from scratch in the remaining hackathon window on CPU alongside everything else. **Not
   recommended** for this pipeline in the time available; if a neural reranker is built at all (see Q3), a
   lighter-weight substitute is simply *including* France-like held-out examples (there are none) or, more
   realistically, relying on the multilingual pretrained encoder's own cross-lingual transfer instead of training
   a domain-adversarial head.

---

## Q3 — Stronger pair models (small cross-encoders / embeddings), CPU feasibility, Ditto-family results

This question is already substantially answered in `research_notes/Business entity resolution methods/
competitions_and_matchers.md` — summarizing and adding the France-specific angle rather than re-deriving from
scratch.

**License/size check (already done, confirmed reusable):** all of multilingual-e5-small (MIT, ~118M params),
paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, ~118M), BAAI/bge-m3 (MIT, ~560M), microsoft/mdeberta-v3-base (MIT,
~280M) clear the ≤8B/MIT-or-Apache bar and are multilingual (relevant for France), vs. all-MiniLM-L6-v2 which is
English-only. For blocking/ANN, **faiss is MIT-licensed** (Meta) —
[faiss.ai](https://faiss.ai/index.html) — and **hnswlib is commonly cited as Apache-2.0** (its own repo states this;
re-verify the LICENSE file directly before adoption since this session's search snippet did not show the license
text itself) — [hnswlib repo](https://github.com/nmslib/hnswlib). `sparse_dot_topn` (already used for the char-3-gram
TF-IDF blocking pass per D-006) is Apache-2.0.

**CPU throughput estimate for ~30M test pairs, reasoned from the numbers already gathered:**
- A cross-encoder (joint pair scoring) at a rough secondary-source estimate of ~100 pairs per 300–500 ms on one CPU
  core (~200–330 pairs/sec/core) would need, at 12 threads with near-linear batch scaling, roughly 2,400–4,000
  pairs/sec aggregate → **30M pairs ≈ 2–3.5 hours**. That is a full afternoon just for inference, on top of any
  fine-tuning time, and the throughput numbers behind it are secondary-source estimates, not a primary benchmark —
  risk of the estimate being optimistic is real.
- Restricting a cross-encoder to only the **uncertain band** (e.g. 0.05 < p < 0.95 after stage-2) is much more
  tractable: this band is plausibly a small fraction of all candidate pairs (most pairs are confidently negative —
  the blocking union already narrows to ~10 candidates/record, and stage-1+stage-2 push most of those toward 0 or 1).
  If the uncertain band is on the order of 5–15% of candidates (not yet measured — **first step should be to actually
  measure this share on the tune group before committing time**, it's a 10-minute query), that's 1.5–4.5M pairs ≈
  10–35 minutes at the same throughput — comfortably CPU-feasible within a day.
- A **bi-encoder** (embed each record once, not each pair) is far cheaper: encoding ~11.6M test records (S1+S2+S3) at
  ~237 sentences/sec single-thread (measured for all-MiniLM-L6-v2, English-only; assume similar order of magnitude
  for a multilingual sibling of similar size) → ~13.6 hours single-threaded, or **~70–80 minutes at 12 threads**
  (near-linear for independent, batchable encode calls). Then ANN search (faiss/hnswlib) over the resulting vectors
  is fast (sub-minute to low minutes at this scale). This is a genuinely cheap addition **as a blocking pass**
  (D-006 already lists a char-3-gram TF-IDF pass and a name-only pass; a bi-encoder ANN pass would be a natural 6th
  pass, specifically targeted at closing the "never a candidate" 6.0% miss rate and giving France — which has no
  learned token log-odds — a semantically-aware signal it currently lacks) or **as a stage-2 feature** (cosine
  similarity between record embeddings, computed once, joined in — much cheaper than a cross-encoder feature since it
  doesn't require per-pair inference).
- Verdict for this task: a **bi-encoder embedding pass (blocking addition and/or stage-2 cosine-similarity feature)**
  is the higher-confidence, lower-cost win; a **cross-encoder reranker restricted to the measured uncertain band** is
  a reasonable stretch goal if time remains after the cheaper items in Q1/Q2/Q6, but should not be started without
  first measuring the uncertain-band size and doing a small (10k-pair) timing pilot, exactly as D-014 already
  requires before reviving this line of work.

**Ditto-family results (already in `competitions_and_matchers.md`, restated for the prioritized list in Q7):** Ditto
beats GBDT-style matchers (Magellan) by the largest margins specifically on "dirty"/near-duplicate benchmark data —
Dirty Walmart-Amazon: Magellan 37.4 F1 vs Ditto 85.69 F1 — which is the closest published analogue to our
near-duplicate-distractor problem (26.5k of 29.5k rule-baseline false positives). This is evidence *for* a reranker
on the hard band, not evidence to replace LightGBM wholesale (D-007/D-014 already reflect this). No new
Sudowoodo/HierGAT numbers were pulled beyond what's already recorded (HierGAT up to +8.7 F1 over Ditto on attention-
weighted token importance — already cited in `blocking_features_decision.md` and directly supports the "learned
token log-odds" feature family we already have, D-013). — [Ditto paper, VLDB 2021](https://ar5iv.labs.arxiv.org/html/2004.00584)

---

## Q4 — Collective / graph-based ER

**How the classic literature frames this:** connected components (transitive closure of pairwise matches) is cheap
but noise-sensitive; **correlation clustering** partitions a similarity graph to jointly maximize within-cluster links
and between-cluster non-links, without needing a fixed cluster count, and is described as more robust than raw
connected components. **Collective entity resolution** (Bhattacharya & Getoor, TKDD 2007) resolves *cooccurring*
references jointly (e.g. co-authors on the same paper) and reports accuracy gains specifically when such relational
cooccurrence exists. — [Bhattacharya & Getoor, "Collective Entity Resolution in Relational Data,"
TKDD 2007](https://linqs.org/assets/resources/bhattacharya-tkdd07.pdf),
[general framing, "(Almost) All of Entity Resolution"](https://arxiv.org/pdf/2008.04443)

**Why this transfers only partially to our problem shape:** the classic collective-ER gain comes from *relational*
cooccurrence between references (shared paper, shared event) that our data doesn't have — S2 and S3 records don't
reference each other directly, they only share a candidate S1 hub. Our graph is close to a bipartite star (S1 centers,
S2/S3 leaves), and the ground truth already gives us the hardest constraint for free: each S2/S3 record belongs to
**at most one** S1 (0 exceptions in 7.64M pairs, `docs/data_profile.md`), which D-003 already enforces via
best-per-right-record. Full correlation clustering assumes a general similarity graph among peer entities; here we
don't have S2↔S2 or S2↔S3 pairwise scores at all (the pipeline only ever scores S1↔S2/S3 pairs), so there's no direct
edge set to correlation-cluster over without first adding a whole new S2↔S3 pairwise-scoring stage — a significant,
likely not-worth-it, engineering lift for a benefit that's unproven on this bipartite-star structure.

**What *does* transfer, and is already partly implemented:** the useful generalization of "collective" reasoning here
is the **cross-source agreement** feature already in D-008 stage-2 (max similarity between this candidate and the
entity's most-confident candidate from the *other* source) — this is the right-scoped analogue of collective ER for
a bipartite-star graph: it uses the fact that a real business usually has both an S2 and an S3 record that agree,
without needing general graph clustering machinery.

**One concrete, cheap extension worth adding:** a hard **"mutual-best-match" indicator** — is this S1↔S2/S3 pair each
side's independent top candidate (S1's highest-probability S2 candidate is also that S2 record's highest-probability
S1 candidate), and does the entity's mutually-best S3 record (if any) also textually agree with this S2 record above
some threshold (a same-business triangle)? This is a few hours of feature engineering on top of data the pipeline
already computes (per-record best-candidate rank is needed anyway for D-003), not a new clustering algorithm, and it
gives the "graph consistency" benefit the literature points to without the precision risk that global weight-
maximization (Hungarian/bipartite) is documented to carry in ER settings (a real large-scale ER system saw naive
global weight maximization *add* false positives at low thresholds — already noted in
`blocking_features_decision.md`, consistent with D-003's rejection of Hungarian assignment). **Net: prefer this
cheap, soft, learned "mutual-best" signal over standing up general graph/correlation clustering**, whose main
precondition (a real peer-to-peer similarity graph) doesn't exist in this pipeline yet and isn't worth building for
an unproven gain given the deadline.

---

## Q5 — Winning solutions of similar public competitions

Already researched in depth in `research_notes/Business entity resolution methods/competitions_and_matchers.md`;
summarized here for completeness, with the specific fit to our setting called out.

| Competition | Winning recipe | Fit to our pipeline |
|---|---|---|
| Kaggle Foursquare — Location Matching (2022) | geo+text candidate generation → cheap LightGBM cut → richer LightGBM (Levenshtein/JW, ratios, SVD name embeddings) → optional transformer/GNN rerank → graph post-process. 7th place reached **0.9935 maxIoU with LightGBM only, no transformer**. Rank/margin-vs-neighbor features called out as a specific hard-negative discriminator. | Matches our architecture shape almost exactly (D-006/D-007/D-008 already mirror this funnel). The rank/margin features are already planned (D-008); the biggest unclaimed piece is the optional GNN/transformer rerank layer used by 1st place — see Q3/Q4 for the CPU-feasible substitutes (bi-encoder feature, mutual-best indicator) instead of a full GNN. |
| SIGMOD 2022 blocking contest | winning recipe = **contrastively pre-trained transformer embeddings + FAISS ANN + symbolic re-rank** ("embed + ANN + symbolic re-rank"), pure blocking/recall task. | Directly supports the Q3 bi-encoder-as-blocking-pass proposal; FAISS is MIT, safe to adopt. |
| SIGMOD 2021 full-ER contest | simple: clean/normalize key fields → near-exact key matching, scored by average F-measure. | Confirms that on *cleaner* data simple key-normalization goes a long way, but our task explicitly injects near-duplicate distractors to defeat exactly this — so this is a floor/sanity-check pattern (our current 0.9715 already clears it), not a ceiling strategy. |
| Kaggle Shopee — Price Match Guarantee (2021) | multimodal (image+text) embeddings trained with ArcFace metric-learning loss + TF-IDF, combined via per-modality thresholds; text-only slice is the relevant piece. | No images here, but reinforces the "embedding + union-of-signals" pattern; ArcFace-style metric-learning embeddings would require labeled fine-tuning time we likely don't have — lower priority than a stock pretrained multilingual bi-encoder. |

**Consistent theme across all of these**: candidate generation via embeddings+ANN, GBDT scoring, and graph/consistency
post-processing are the three moves that recur; no winning solution skipped GBDT scoring entirely, and only 1st-place
tier solutions added the transformer/GNN layer on top rather than instead of it. This is consistent with the
prioritized list below: harden blocking + features + decision first, treat a neural component as an additive
reranker/feature, not a replacement.

---

## Q6 — Decision-level hacks given the metric (macro F0.5, per-S1, singleton = 1.0 for empty prediction)

Grounded in `docs/data_profile.md` (already measured, no new research needed for the base facts) plus reasoning:

- **Matches-per-entity is not flat.** Train distribution: 1 match → 5.7%, 2 → 18%, 3 → 25%, 4 → 23%, 5 → 15%,
  6 → 8%, 7+ → 4% (max 11); singleton rate 5.6%. D-010's EM prior-shift already uses an aggregate version of this
  (assumes matches-per-entity ≈ stable ≈ 3.4 across countries, flags France if its predicted rate deviates). The
  expected-F0.5 decision rule (D-009, Jansche/Dembczyński) is already the theoretically-correct way to fold a size
  prior in via calibrated probabilities rather than a hard count target — **no need for a separate explicit
  "predict-the-count-then-take-top-k" model**, that would just be a less-principled version of what D-009 already
  does, so this is not a new recommendation, just a confirmation that the current design choice is the right one.
- **Per-source composition is a genuine additional signal not yet exploited.** `docs/data_profile.md`: "typical
  composition: 1–3 from S2 and 1–4 from S3; ~11% of S1s match only one source." Combined with
  `docs/normalization_findings.md`'s finding that S2 and S3 have *systematically different* noise profiles
  (all-caps name/address: S2 27.9%/63.2% vs S3 7.6%/0%), pooling S2 and S3 candidates into one ranked list per entity
  for calibration/threshold purposes risks a subtly wrong calibration curve for whichever source's score distribution
  differs from the pooled average. **Proposal: fit isotonic calibration (and/or the expected-F0.5 decision) per
  source** (S2 vs S3) in addition to per country, not just globally. This is a small change to an existing stage
  (D-009's isotonic calibration step already exists; this just adds a `source` grouping key alongside or instead of
  a single global fit) — cheap (a few hours) and directly targeted at a measured asymmetry, so it should rank as a
  quick, low-risk item.
- **Abstaining on ambiguous chain-store names.** 50% of S1 names are shared with another S1 entity
  (`docs/DESIGN_V2.md` §1), which is exactly where a single pair's raw similarity is least informative (a chain
  store's name matches many candidates equally well; only the address disambiguates, and France chains would be
  hit hardest since address features are also the weakest signal there per Q1). The expected-F0.5 rule already
  abstains naturally *if* probabilities are well-calibrated and genuinely lower for ambiguous chain cases — but
  today's feature set has no direct signal for "how many other S1 entities in this country share this literal
  name," only pairwise/local features. **Proposal: add a `name_core` document-frequency-within-country feature**
  (count of other S1 entities sharing the same normalized `name_core` in that country) as a stage-1 or stage-2
  input, so the model can learn extra caution specifically for high-frequency chain names instead of relying on
  address features alone to carry that signal indirectly. Cheap (one groupby count), directly targets the design
  doc's own stated top false-positive driver (near-duplicate distractors on repetitive names), and is
  country-agnostic by construction (computed fresh per country, so it works unchanged for France).

---

## Q7 — Prioritized top 8, with expected gain / cost / risk

Ordered roughly by gain-per-hour, with low-risk small items first since they're safe to stack; the two larger bets
(#7, #8) should only be started once the cheaper items are done and — per D-014's own rule — after measuring the
inputs that determine whether they're worth it (uncertain-band size for a reranker; pseudo-label precision on a
manual spot-check for self-training).

| # | Idea | Targets | Cost | Expected gain | Risk |
|---|---|---|---|---|---|
| 1 | **Split D-010's EM prior-shift by country** (India/US/France separately, not pooled) | France's measured 20.6% empty-prediction rate (rule baseline; check if v2 still shows this) | ~2h | Small–medium, concentrated in France's 15% | Low — EM only rescales already-calibrated probabilities |
| 2 | **French normalization pass**: SARL/SAS/SASU/EURL/SA/SNC/EI/Sté/Ets/Cie legal-form map, street abbreviations (r/av/bd/pl/all/ch), elision folding (l'/d'/qu'...), `cedex`+routing-digit stripping, generic ordinal-suffix regex | Q1 gaps 2, 3, 5, 6, directly measured in `normalization_findings.md`'s France sample | ~4–6h | Medium — France is 15% of test and plausibly the weakest-normalized slice today | Low — generic, hand-written, testable against the existing true-pair/hard-negative harness before shipping |
| 3 | **Generic house-number-suffix extraction** (bis/ter/quater, unit letters) as its own low-weight agreement feature instead of fusing into or stripping from the number | Q1 gap 4 — precision risk (false merges) and recall risk (missed matches) on French and English/Indic addresses alike | ~3h | Small–medium, mostly precision protection | Low |
| 4 | **`name_core` document-frequency-within-country feature** (how many other S1 entities in this country share this normalized name) | The design doc's own stated #1 false-positive driver: 26.5k near-duplicate distractors on repetitive/chain names; 50% of S1 names are shared | ~2–3h | Medium — directly targets the largest measured FP class | Low |
| 5 | **Per-source (S2 vs S3) isotonic calibration / expected-F0.5 decision**, alongside the existing per-country grouping | Measured S2/S3 noise asymmetry (all-caps name/address 27.9%/63.2% on S2 vs 7.6%/0% on S3) that a pooled calibration curve can't see | ~3–4h | Small–medium | Low |
| 6 | **`token_logodds_coverage` feature** (share of a pair's tokens with known log-odds) + **frequency-based generic stopwording** (per-country DF threshold instead of hand lists) | Q1 gap 8 (log-odds silently unpopulated for France) and function-word noise (French de/la/le, generalizes to any language) | ~4h | Small–medium, mostly protective for France | Low–medium (stopwording needs a recall check before shipping) |
| 7 | **Multilingual bi-encoder embedding pass** (multilingual-e5-small, MIT) as a 6th blocking pass + cosine-similarity stage-2 feature, ANN via faiss (MIT) | Blocking's "never a candidate" 6.0% miss rate, especially transliterated/French names where string-only blocking is weakest; matches the SIGMOD-2022-winning "embed+ANN+symbolic rerank" recipe | ~1–1.5 days (encode ~11.6M test records ≈70–80 min at 12 threads, plus index build + integration + re-validation) | Medium–high if blocking is still the score ceiling; effectively free extra signal for France since it needs no French labels | Medium — new dependency, engineering integration risk, must re-verify hnswlib/faiss license text and re-time on this laptop before committing |
| 8 | **Conservative pseudo-labeling of high-confidence France test pairs**, used to (a) populate French-specific token log-odds and (b) optionally augment a holdout-safe retraining pass | France is the only slice with zero labeled signal at all; this is the one lever that can give it real learned signal rather than just better generic features | ~0.5–1 day (filter design, retrain, re-validate holdout, manual spot-check of ~200 pseudo-pairs) | Medium–high specifically for France, but high variance | Medium–high — self-training literature is explicit that errors compound if the confidence filter is too loose, and F0.5's 2× precision weighting makes a wrong pseudo-positive costlier than a missed true positive; must be piloted and spot-checked, not shipped blind |

**Honorable mentions, not in the top 8 given cost/benefit at this deadline:** the "mutual-best-match" cross-source
triangle indicator (Q4) — cheap (~3h) but lower expected gain since D-008's cross-source agreement feature already
covers most of the same signal softly; a cross-encoder reranker restricted to the measured uncertain band (Q3) —
potentially high ceiling but explicitly gated by D-014 on first measuring the uncertain-band size and running a
10k-pair timing pilot, so it's a *possible* #9 only after that measurement comes back favorable; per-record
importance weighting toward "France-like" characteristics (Q2.4) — cheap but the weighting heuristic has no labels
to validate against, so it's a quick experiment to try opportunistically rather than a planned line item; full
correlation clustering / domain-adversarial training — not recommended, see Q4 and Q2.5 respectively.

**Net read:** none of the top 8 requires new external data or violates the model-license/size rule. Items 1–6 are
all low-risk, half-day-or-less, and stack independently — they should be done first and in parallel where possible
(a fork per item is reasonable if the coordinator wants to parallelize implementation, though this research task
itself did not implement anything). Items 7–8 are the two genuine bets for closing more of the gap to the ~0.988
leader ballpark; both are explicitly allowed by the rules, both have a measurement step that should happen before
committing full implementation time, and both are more valuable for France specifically than for India/US, which is
consistent with the brief's emphasis on the unseen-country generalization problem.

Status: COMPLETE.

