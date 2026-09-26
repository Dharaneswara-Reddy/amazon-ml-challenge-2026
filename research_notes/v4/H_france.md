# H — France: self-labelling French test pairs, French vocabularies, cut diagnosis (v4 research)

Research/prototype only; `code/business_entity_resolution/` untouched. Scripts: `notebooks/v4_france/h*.py`
(helper `h_common.py`, logs in `notebooks/v4_france/out/`). DuckDB temp `artifacts/v4_tmp/spill`, 3 GB, 5 threads.
v3 run dir: `artifacts/runs/20260926T032546_e2e7494_678411c5_v3/` (pred2 p = raw stage-2 score; the decision is
"best candidate per right AND calibrated, per-country-shifted p >= calibrated 0.69").
Findings are appended as they are made (status line at the end of each section).

## Context numbers (predict.json of v3)
- Test pairs by S1 country: France 5,346,389 for 259,452 S1 = **20.6 candidates per S1**; India 7,742,742 / 809,986
  = 9.6; US 5,835,047 / 663,106 = 8.8. So the France candidate density is 2.2x India/US after the v3 cut (v2 had
  16.6 vs 17.5/17.3, i.e. the learned cut barely prunes France).
- Matches per S1: France 3.43, India 3.33, US 3.38; EM prior shift France -0.32, India -0.62, US -0.55 logit.

## Finding 1 — self-label rules: precision measured on TRAIN (h01_pairs.py, h02_rules.py; log out/h02.log)
Pairs: v3 train holdout+tune candidate pairs (4.55M, labels) and all France test pairs (5,346,389; 1,434,993 rights).
`p` = v3 stage-2 score (pred2), `rid-best` = highest-p candidate of the right record, margin = p - second p of the right.
"nums all equal" = sg_num_changed = 0, no unpaired number on either side, >= 1 equal number (signature diff of
name_norm/addr_norm, `ber.signature.pair_ops`). Precision = share of rule pairs with the rule's label; coverage = share
of all pairs of that label in the candidate set.

| rule | y | India n | India prec | India cov | US n | US prec | US cov | France n (share) |
|---|---|---|---|---|---|---|---|---|
| D1 sole offset in S(>=3) | 0 | 112,608 | 0.9946 | 0.085 | 156,962 | 0.9953 | 0.081 | 500,840 (0.094) |
| D1b D1 & rid-best | 0 | 60,721 | 0.9905 | 0.046 | 118,293 | 0.9940 | 0.061 | 263,252 (0.049) |
| D1c D1 & name edit (ins/repl) | 0 | 105,690 | 0.9978 | 0.080 | 152,688 | 0.9987 | 0.079 | 451,653 (0.084) |
| D1d D1 & p<0.5 | 0 | 112,047 | 0.9993 | 0.085 | 156,334 | 0.9983 | 0.081 | 486,649 (0.091) |
| D2 all pairs of a right whose best pair is D1 | 0 | 111,167 | 0.9945 | 0.084 | 210,489 | 0.9965 | 0.109 | 781,069 (0.146) |
| E p<0.01 | 0 | 1,252,863 | 0.9996 | 0.953 | 1,797,086 | 0.9991 | 0.931 | 4,051,596 (0.758) |
| E2 p<0.05 & not rid-best | 0 | 1,131,746 | 0.9958 | 0.857 | 1,652,649 | 0.9933 | 0.851 | 3,845,687 (0.719) |
| T1 nums all equal & no name ins/repl | 1 | 197,525 | 0.9757 | 0.378 | 380,551 | 0.9811 | 0.469 | 717,772 (0.134) |
| T2 T1 & rid-best | 1 | 186,288 | 0.9954 | 0.364 | 361,365 | 0.9972 | 0.453 | 639,268 (0.120) |
| T3 T2 & p>=0.5 | 1 | 185,454 | 0.9995 | 0.364 | 360,398 | 0.9996 | 0.453 | 632,774 (0.118) |
| T4 nums all equal & rid-best & p>=0.5 | 1 | 295,945 | 0.9987 | 0.580 | 476,097 | 0.9989 | 0.597 | 796,627 (0.149) |
| T5 T4 & margin>=0.5 | 1 | 284,283 | 0.9991 | 0.557 | 458,829 | 0.9992 | 0.576 | 795,185 (0.149) |
| Tp rid-best & p>=0.95 & margin>=0.5 (model only) | 1 | 447,064 | 0.9994 | 0.876 | 695,021 | 0.9995 | 0.873 | 830,200 (0.155) |
| Tn nums all equal & name exact/case-only & rid-best | 1 | 89,095 | 0.9991 | 0.175 | 188,587 | 0.9998 | 0.237 | 397,868 (0.074) |

- The number signature alone (D1, D2, T2 — no model score, no language) is 0.990-0.998 precise on both train
  countries; adding the model score (D1d, T3/T4) pushes it to >= 0.998. These transfer to France as long as the
  generator is the same (it is: finding 2).
- On France, "model only" Tp covers 830k pairs but its precision there is NOT the train number (finding 2 shows ~25k
  distractors inside France's p >= 0.69 rid-best set), so the pseudo labels use signature-anchored rules.

## Finding 2 — France's accepted pairs contain ~20-25k number-nudged distractors (h03_offsets.py; out/h03.log)
Rid-best pairs with exactly one changed number (offset class, no truncation), count by signed offset R-L and p band
(train cells = n/true):

| country | p band | -2 | -1 | +1 | +2 | +3 | +4 | +5 | +6 | +7 | +9 | +11 | +13 | +21 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| France | p>=.69 | 277 | 262 | **9,995** | **10,377** | 1,400 | 1,384 | 1,021 | 77 | 806 | 655 | 342 | 377 | 36 |
| France | .2-.69 | 190 | 161 | 8,534 | 8,811 | 6,121 | 6,363 | 5,700 | 57 | 5,484 | 4,790 | 3,483 | 3,510 | 838 |
| India | p>=.69 | 1,102/1,099 | 1,119/1,112 | 1,178/1,165 | 1,225/1,212 | 131/126 | 146/146 | 101/99 | 100/100 | 60/60 | 21/20 | 9/7 | 15/12 | 10/10 |
| US | p>=.69 | 1,558/1,557 | 1,631/1,621 | 1,475/1,386 | 1,491/1,396 | 71/57 | 67/59 | 56/48 | 86/81 | 66/56 | 53/48 | 47/39 | 58/47 | 0 |

- On train, high-p +1/+2 pairs are true typos: +1/+2 counts equal the -1/-2 counts (true copies make symmetric
  +-1/+-2 typos) and ~99% are true. **In France, +1/+2 at p >= .69 are 37x the -1/-2 counts**: ~9.7k excess at +1 and
  ~10.1k at +2, i.e. ~19.8k distractors (~97% of those pairs) accepted by the model, plus ~5.3k with an offset in
  S>=3 above the off-S baseline (~75 per offset). Total **~25k French distractor pairs among 896k rid-best p>=.69**
  (India 13 of 475k, US 64 of 736k in the train groups). The model learned "+1/+2 with a clean-looking name = typo";
  on French pairs it cannot see the French name edit, so a nudged +1/+2 distractor looks clean.
- Rough cost: ~25k FP pairs over 259k French S1 (~9% of French S1 touched) ~ 2 F0.5 points of the ~6.6-point gap.
  Name-only distractors (numbers equal, ~7% of distractors on train) are invisible to this check and add more.

## Finding 3 — French pseudo-labelled pair set (h04_pseudo_vocab.py; out/h04.log) -> artifacts/v4_tmp/fr_pseudo.parquet
Columns lid, rid, y, confidence (= min train precision of the rule over India/US), rule, p. Priority T > D1 > D2 > E.

| rule | definition | y | train precision (India+US pooled) | France pairs | France S1 | France rights |
|---|---|---|---|---|---|---|
| T3 | nums all equal & rid-best & p>=0.5 & no name ins/repl | 1 | 0.99956 | 632,774 | 230,181 | 632,774 |
| T4 | nums all equal & rid-best & p>=0.5, with name ins/repl | 1 | 0.99706 | 163,853 | 117,457 | 163,853 |
| D1 | exactly one changed number, offset in {+3,+4,+5,+7,+9,+11,+13,+21} | 0 | 0.99500 | 500,840 | 207,911 | 420,811 |
| D2 | other pairs of a right whose best pair is D1 | 0 | 0.99998 | 484,134 | 174,862 | 172,956 |
| E | p < 0.01 | 0 | 0.99920 | 3,220,176 | 252,435 | 899,500 |

Total 5,001,777 of 5,346,389 French pairs labelled (93.6%); 796,627 positives (0.55 of the 1.43M French rights).
Unlabelled: 344,612 pairs = the hard ones (incl. the +1/+2 distractors of finding 2, which carry no rule).
Caveat: T4's train precision (0.997) is an upper bound for France — French name-only distractors (numbers equal,
French word inserted) get p >= 0.5 more often than on train because the model is blind to the French words.

## Finding 4 — the French distractor name vocabulary, isolated (h05_generator_vocab.py; out/h05.log)
Rid-best D1 pairs (number-anchored distractor vs its parent S1; labels independent of the name) vs T3+T4 true copies,
150k sampled pairs each; per-pair presence of an inserted / replaced token; lo = log(share in D1b / share in T).
- **Inserted words (share of French D1b pairs / share of T pairs):** groupe 7.7% / 0.41%, holding 7.7% / 0.016%,
  developpement 7.6% / 0.40%, participations 7.4% / 0.029%, france 7.4% / 0.42%, distribution 7.3% / 0.007%,
  international 4.9% / 0.003%, snc 1.1% / 0.012%. This is the French counterpart of train's group / holdings /
  industries / enterprises ... list (each ~1-3% of train D1b). groupe / developpement / france show ~0.4% in T:
  either French name-only distractors that passed T4's p >= 0.5 (likely; the train analogue is ~0) or true-copy
  noise — do not hard-code them as certain.
- **Legal-form swaps** (French llc>inc analogue): sarl>snc 1.15%, sas>sci 0.85%, sas>snc 0.85%, eurl>sasu, eurl>sas,
  ei>sci, sa>snc, and legal -> generator word (sarl>international, sarl>holding, sas>developpement, sas>groupe).
  Plain insertion of a legal form (sarl, sas, sa, sci, sasu, eurl ~1.8-2.3% each) is neutral (same rate in D1b and T),
  exactly like train's inserted llc/inc.
- No French location-qualifier list (train's east/uptown/... ) shows up; the French generator seems to use only the
  org-suffix + legal-swap vocabularies. Coverage: a French D1b pair carries a generator word (lo >= 3) in 45% of cases
  (train: 70%) — the rest are number-only edits or edits we do not pair up.
- **Where these words sit in France's accepted pairs** (rid-best, p >= .69): the +1/+2 accepted pairs (P12, 20,372)
  carry a generator word in **39.2%** (D1b: 45.4%) -> P12 is ~85-90% distractors, independently confirming finding 2
  with the name side; other accepted pairs: T4 0.87% (1,308 of 150k sample), the rest 1.26%; on train the same groups
  are 0.03-0.3%.
- Train check of the procedure: the same D1b-vs-T lists on train recover exactly the known generator vocabulary
  (holdings, industries, enterprises, infratech, group, the 21 location words, llc>holdings/partners/group/inc/...).

## Finding 5 — vocabularies learned from pseudo labels (h04_pseudo_vocab.py) -> sigvocab_fr.json, logodds_fr.json
- Sample: 350k pairs per rule family (T / D / E) -> 350k positives, 700k negatives; same composition on train.
- **Validation on train (India+US):** a sigvocab learned from the SAME pseudo rules on train vs the production
  `learned/sigvocab_fit.json` (true labels, all candidates): Pearson on shared tokens ins 0.951 (438 tokens),
  repl 0.973 (1,162), rtgt 0.965 (2,510), ains 0.790, del 0.33 (del values are all near 0 in both); slope ~1.15-1.2
  (pseudo values slightly larger because the negative mix differs), every token with true lo >= 3 has pseudo lo >= 2.
  Token log-odds (textdiff) pseudo vs true on the same pairs: r = 0.996. So the self-labelling recovers the vocab.
- **France vocab** (sigvocab_fr.json): ins 305, del 520, repl 6,277, rtgt 4,571, ains 1,584, adel 3,105 tokens; 272 /
  6,258 / 4,130 ins/repl/rtgt tokens are absent from the train vocab (train vs France shared ins tokens: 33 only).
  Top ins: international 7.4, distribution 6.4, holding 5.9, participations 5.1, snc 5.0, freres, amis, club (the last
  ones come from the E negatives = unrelated businesses, like the train vocab). Top repl: sarl>international 6.7,
  sas>distribution 6.5, sarl>holding 6.3, eurl>sas 5.9, sci>sas 5.7, sas>groupe 5.5, sarl>france 5.4; du>des/de>du
  ~6.5 (article swaps: unrelated-business negatives). True-copy noise (negative lo): ins nee -3.1, e, c; repl
  abbreviations (etablissements>ec, union>us, bordeaux>bs, lycee>lj — initialisms), address del of region names
  (aquitaine, hauts, pays, loire, france: dropped by copy noise in both classes, lo ~ -0.4).
- logodds_fr.json: 10,121 tokens (train table: 15,321; overlap small).

## Finding 6 — why France keeps 3.7 candidates per right (h06_cut.py, h06b_cut.py, h06c_sim.py; out/h06*.log)
**Cause: French S1 records are far more mutually similar, not a broken cut feature.**
- Geography: 77.5% of French S1 sit in the top-10 cities (Bordeaux 40k, Nantes 32k, Lille 30k, Tourcoing, Dunkerque,
  Calais, Roubaix, Pessac, Saint-Nazaire, La Teste-de-Buch; US 6.7%, India 5.1%); only 169 distinct postal codes (US
  635, India 836); distinct street / S1 = 0.068 (US 0.30, India 0.29) — "rue jules guesde" 538 S1, "rue jean bart" 532.
- Names are templated: 49% of French S1 share their exact name_core with another S1 (US 40%, India 53% but India's
  shared names are spread over 590k cities); most repeated: "bordeaux club" 530, "nantes club" 486, "lille club" 423,
  "bordeaux amicale" 342; top tokens club 8.7%, france 8.0%, de 6.7%, ecole 6.0%, amicale 5.5%, comite 5.3%.
  So a French right has many same-city, same-street, same-template S1 neighbours.
- Consequence in the cut (test, all candidates): France rights with 1 candidate 27% (India 67%, US 69%), >= 4: 39%
  (India 7%, US 4.5%). Non-best kept pairs (lrank > 1) look MORE similar in France: median jw_core 0.887 (India 0.849,
  US 0.692), jw_phn 0.834 (0.734 / 0.685), jw_addr 0.758, jacc_addr 0.50; they enter mainly via the phonetic pass phx
  (21.9% vs 12-13%) and name-word x address-word na (23.3% vs 13-15%), not via v2 (41% vs 57-61%). Per-feature the cut
  model behaves sensibly — the pairs really are similar. Rid-best (lrank = 1) pairs look like India/US.
- A stricter cut on lrank > 1 pairs is almost free in recall:

| cut threshold t (lrank>1 & cut_p<t dropped) | train India true lost | train US true lost | France pairs dropped | France currently accepted pairs dropped (of 896,071) |
|---|---|---|---|---|
| 0.003 | 0.026% | 0.010% | 1.05M (20%) | 48 |
| 0.01 | 0.083% | 0.043% | 2.07M (39%) | 127 |
| 0.03 | 0.21% | 0.13% | 2.81M (52%) | 1,320 (233 are D1 distractors) |
| 0.1 | 0.47% | 0.36% | 3.40M (64%) | 3,955 (665 D1) |

  Pseudo-label view (lrank > 1 kept French pairs): E 3.05M, D2 416k, D1 289k, T3+T4 only 3k, unlabelled 150k.
- **Recommendation (cut):** do NOT retrain the cut for France — its job is recall and France recall is fine; the
  extra candidates are mostly easy/negatives that the stage-1 model already scores < 0.01 (E = 76% of French pairs).
  Cheap optional change: France-only (or global) threshold 0.003-0.01 on lrank > 1 (costs <= 0.08% recall on train,
  127 French accepted pairs) to cut French pairs by 20-39%, bringing the per-right competition features (r_n_cand,
  l_n_cand, s2_l_n50 ...) closer to the train distribution. The F0.5 gap is a discrimination problem (findings 2, 4).

## Finding 7 — French normalization bugs (h07_norm.py, h07b_norm_bugs.py; out/h07*.log; 1,694,445 French records)
Legal forms, accents, street types and "bis" are handled well: SARL/SAS/SASU/EURL/SA/SCI/EI -> `legal_form`
(S.A.S. and Sàrl too), accents folded (Allée -> allee, Semprùn -> semprun), R./Ave/Bd/Rte/St.- -> rue/avenue/
boulevard/route/saint, "N°5"/"NO 19" -> 5/19, "Cie" -> compagnie, "trading as" DBA handled, "30 bis" -> addr_house
"30 bis". Concrete bugs (all in `ber/normalize.py`):
1. **Region names are not admin units, and region <-> département is not equated.** addr_admin values: hauts de
   france, nouvelle aquitaine, nord, gironde, loire atlantique, pas de calais — but **"pays de la loire" is not
   recognised** and leaks into addr_city ("nantes pays de la loire" 88,788 records, "saint nazaire pays de la loire"
   33,558, "pays de la loire nantes" 9,818 ... vs plain "nantes" 121,222), so city_eq fails between copies. Copy noise
   swaps region <-> département (Nouvelle-Aquitaine <-> Gironde, Hauts-de-France <-> Nord / Pas-de-Calais, Pays de la
   Loire <-> Loire-Atlantique) like US state name <-> code: **admin_eq on French pseudo-true pairs (T3) = 0.25 vs
   0.90-0.91 on train positives** (city_eq 0.76). Fix: add "pays de la loire" to the admin list and a hand-written
   département -> région map for the 4 départements seen (gironde -> nouvelle aquitaine, nord / pas de calais ->
   hauts de france, loire atlantique -> pays de la loire), comparing admin at région level (and generally strip
   admin words out of addr_city before city_eq).
2. **"ter" -> "terrace"**: the US street-abbreviation map (normalize.py:520 `"ter": "terrace"`) fires on the French
   house-number suffix: raw "56 Ter Rue de Coulmiers" -> "56 terrace rue de coulmiers", addr_house "56" (bis is kept
   as "30 bis" because `_NUM_SUFFIX` catches it first). 8,888 raw "<n> ter" records, 7,894 normalized to "terrace".
   Fix: apply the `_NUM_SUFFIX` rule (number followed by bis/ter/quater) before the street-type map.
3. **Elision is inconsistent**: ASCII apostrophe before a consonant splits ("d'Halluin" -> "d halluin", "L'YSER" ->
   "l yser") but before a vowel it is glued ("l'Egalité" -> "legalite", "l'Océan" -> "locean", "L'ÉPEULE" ->
   "lepeule"); the curly apostrophe ’ (12,980 records) is always deleted ("D’ARCACHON" -> "darcachon", "D’AFN" ->
   "dafn"). Copies of one address can therefore produce "d arcachon" vs "darcachon". Fix: map ’ and ' identically and
   drop the elided article: `\b[ldjmnst][’']` -> "" followed by a space, i.e. "l'eglise" -> "eglise",
   "d'arcachon" -> "arcachon" (also "qu'").
4. **US legal forms / abbreviations eat French tokens**: "LP Club SARL" -> legal_form "lp sarl"; "Company Studio" ->
   "co"; "PA Sportive" -> "pa" (co 520, pa 233, lp 218, "co sarl" 171 ... ~1.5k records); "Centre" -> "center" (US
   map, harmless because consistent). Low impact; restrict co/lp/pa/pc to the trailing position or non-French names.
5. Postal codes: French addresses carry none (postal non-empty 0.14%), so postal features are blank for France — not
   a bug, but France relies entirely on city/street/number.

## Finding 8 — impact: re-scoring France with the French vocab (h08_rescore.py, h09_eval.py; out/h08.log, out/h09.log)
Faithful v3 path on the France subset: feat -> model1 -> stage-2 competition features -> model2 -> v3 decision
(is_best & calibrated p shifted by France's delta >= calibrated 0.69). 'base' = unchanged France features: reproduces
the submitted pred2 exactly (max |dp| = 0.0), so the France subset is self-contained (blocking is within-country).
'fr' = sg_* (46 signature features) and nd_*_logodds_* recomputed with vocab = train vocab overridden by the French
pseudo vocab (sig values scaled by 1/1.18, h04 slope) for the 1,406,694 French pairs with p2 >= 0.003. Models unchanged.
Runtime ~10 min (recompute 3 min with 4 workers, model1+stage2+model2 ~3 min per variant).

| France, v3 decision | v3 submitted | fr vocab (fixed shift -0.318) | fr vocab (EM shift -0.409) |
|---|---|---|---|
| accepted pairs | 890,378 | 886,536 | 884,915 |
| matches per S1 / empty S1 share | 3.432 / 5.30% | 3.417 / 5.46% | 3.411 / 5.48% |
| accepted with sole offset +1/+2 vs -1/-2 | 18,922 vs 521 | **808 vs 479** | 755 vs 471 |
| -> est. +1/+2 distractors accepted (excess) | **18,401** | **329** | 284 |
| accepted D1 (offset in S>=3) | 4,885 | **390** | 362 |
| rid-best pairs with 0.05<p<0.95 | 17.65% | **7.44%** | 7.44% |
| all pairs with 0.05<p<0.95 | 5.97% | **2.89%** | 2.89% |
| pseudo-true T3 / T4 accepted | 99.76% / 92.38% | 99.81% / 94.32% | 99.79% / 93.96% |

Flips vs submitted (fixed shift): +27,141 added, -30,983 removed; removed = 18,302 +1/+2-offset pairs, 4,602 D1,
343 T3, 4,615 T4, 21,423 unlabelled (overlapping categories).

## Finding 9 — what the flips are (h10_flips.py, h11_flip_words.py; out/h10.log, out/h11.log)
- Flips are almost never re-assignments (18 rights switch S1); 27,123 rights newly accepted, 30,965 rights dropped.
- **Removed, numbers changed (25,591)**: 59% sole +1/+2 offset; 39.3% carry a French generator word (number-anchored
  distractors: 45.4%) -> **~87% distractors (~22k FPs removed, ~3.4k TPs lost)**. Examples: "Departementale Club SCI,
  100 Rue du Vauxhall" -> "DEPARTEMENTALE CLUB PARTICIPATIONS, 107 RUE DU VAUXHALL"; "Fees Sport SARL, 30 Rue Fénelon"
  -> "Fees Sport France SARL, 37 R. FÉENLON"; "Avant Loisirs SA, 6 Rue Reine des Prés" -> "Avant Loisirs [EURL], 11 ...".
- **Removed, numbers equal (5,392)**: generator-word share 17.3% -> ~38% distractors (~2k FP removed, ~3.3k TP lost,
  e.g. acronyms "Lycée Saint Tous" -> "LST", "Ets Action" -> "Ets Action SAS" with no house number on the S1).
- **Added (27,141)**: 85.5% numbers all equal, 87% with a name insert/replace (true-copy noise now recognised: "Et
  Fils", "& Associés", hyphenation, acronyms "MA"/"LC", "-S.A.S"); old p 0.2-0.67. Only 188 (0.7%) have a +1/+2
  offset; since distractors show a changed number 93% of the time and offsets are uniform over 10 values, the added
  set holds at most ~1k distractors (<= ~4%). Caveat: generator-word rate 0.15% here is partly circular (the vocab
  drove the rescoring); some content-word-replacement siblings slip in ("Dunkerque Sportive SAS" -> "Groupe
  Dunkerque SAS", "Locale Atelier SCI" -> "Locale [Culturelle]").
- Kept pairs (859k): generator-word share 0.01% (train-like).
- **Estimated France effect**: FP pairs -23k, TP pairs +19k over 259k French S1. With ~3.4 matches per S1, one FP costs
  an S1 ~0.2 F0.5 and one missed TP ~0.08 -> **France F0.5 ~ +0.02 to +0.025 (0.92 -> ~0.945), public ~ +0.003 to
  +0.004 (0.9776 -> ~0.981)**. Model's own uncertainty on French rid-best pairs falls 17.7% -> 7.4% (India 2.3%,
  US 2.7%), so France is still ~3x less decisive: remaining levers are normalization (finding 7) and a stage-2 model
  that has seen French pairs (not prototyped here — out of time budget).

## Recommended implementation (in order of value / effort)
1. **French vocab from transductive pseudo labels (biggest, measured above).** New step after test features exist
   (or before, using a v3 test prediction pass): in `ber/pipeline.py learn_logodds_stage`, add test pairs labelled by
   the signature rules (h04 `LABEL_SQL`: T4 = nums all equal & rid-best & p2 >= 0.5 -> 1; D1 sole offset in
   {+3,+4,+5,+7,+9,+11,+13,+21} or D2 other pairs of such a right -> 0; E p2 < 0.01 -> 0) and learn
   `sigvocab_test.json` / `logodds_test.json` with the existing `learn_sigvocab` / `textdiff.learn_token_logodds`
   (sample ~350k per rule family). For test feature computation use vocab = {**train_fit, **test_pseudo * 1/1.18} and
   logodds = {**train_fit, **test_pseudo}. Language-independent (applies to all test countries; for India/US tokens
   the train values are kept). Requires a two-pass predict: pass 1 = current v3 (for p2, rid-best), pass 2 = recompute
   sg_* and nd_*_logodds_* (only pairs with p2 >= 0.003 need it; ~3 min for France with 4 workers) -> model1 ->
   stage-2 -> model2 -> decision. No retraining needed. Prototype: `notebooks/v4_france/h08_rescore.py`.
   Rules only use the offset set / number equality learnt from train, so no external data and no country code.
2. **normalize.py fixes (finding 7):** "pays de la loire" as admin + département->région map (gironde, nord,
   pas de calais, loire atlantique); number suffix bis/ter/quater before the street-type map (`"ter": "terrace"` bug);
   one elision rule for ' and ’ that drops the article (l', d', qu' ...). These change features for train too (they
   are hand-written, language rules) -> needs a full rerun; admin_eq on French true pairs is 0.25 vs 0.90 on train.
3. **Optional stage-2 adaptation:** append French pseudo-labelled s2 rows (T3/T4/D1/D2 + a sample of E, weight ~0.3)
   to model2 training; evaluate India/US on the untouched holdout. Not prototyped.
4. **Cut:** leave as is, or raise the lrank>1 threshold to 0.003-0.01 (France pairs -20..-39%, train recall cost
   <= 0.08%) mainly to keep stage-2 competition features in-distribution (finding 6).

Artifacts: `artifacts/v4_tmp/{fr_pseudo.parquet, sigvocab_fr.json, logodds_fr.json, sigvocab_merged.json,
logodds_merged.json, pred2_fr_fr.parquet (France pairs re-scored), pairs_fr.parquet, pairs_train.parquet}`.
Status: tasks 1-5 done except the LightGBM retrain prototype (recommendation 3).
