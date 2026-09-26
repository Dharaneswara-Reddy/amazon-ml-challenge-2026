# E — End-to-end proof: v3 blocking + features vs v2 (measured, holdout macro F0.5)

Scripts: `notebooks/v3_proof/` (p01..p0N). Scratch data: `artifacts/v3_proof_tmp/` (DuckDB db + spill, git-ignored).
Frozen code `code/business_entity_resolution/` is imported, never edited. Findings appended as they are measured.

## Protocol (identical for every arm)
- Left for candidate generation = ALL 2,206,821 train S1 (realistic rivals and IDF); right = all train S2 ∪ S3.
- Entity subsets (md5(S1 id) % 100 buckets, `ber.split`): FIT-SUB = 0-4 (5% of S1), TUNE = 30-34, HOLDOUT = 35-39.
- Every group's feature file = pairs whose lid is in the group (in_group = 1) + all other candidates of the same right
  records (in_group = 0, rival context), exactly as `pipeline.features_stage`.
- Model: one LightGBM stage (ber DEFAULT_PARAMS + learning_rate 0.2, seed 42, 8 threads), trained on FIT-SUB
  in_group rows, early-stopped (50 rounds) on TUNE in_group rows.
- Decision: `ber.decision.choose_decision(tune, holdout)` on `_group_arrays`-equivalent arrays (rights whose best lid
  over all rows is outside the group are dropped, all group entities present with n_true); rule chosen on TUNE halves
  only; HOLDOUT scored once per arm.
- Text-difference log-odds: FIT-SUB rows use `logodds_fitB` (out-of-fold, as v2 does for fitA), TUNE/HOLDOUT use
  `logodds_fit`; name IDF from `ber.features.name_idf` over train L ∪ R. Same for all arms.

### Arm `base_v2` — HOLDOUT macro F0.5 **0.96734** (rule expected_f; 83 features, 358 trees, 1,545,064 FIT-SUB rows)
- candidate recall 94.5396%, candidate pairs per holdout S1 14.09
- per country: india 0.96167 (n=44,353; oracle 0.97500), us 0.97114 (n=66,096; oracle 0.98354)
- all rules on holdout: threshold 0.96702, expected_f_pure 0.96671, expected_f_model 0.96734, expected_f 0.96734
- true pairs 382,075: TP 352,201, blocking miss 20,863, one-to-one loss 1,666, decision FN 7,345;
  FP 1,678 on matched entities + 187 on singletons (171 singletons zeroed)
- oracle / one-at-a-time fixes: oracle: perfect decisions on candidates (TP = all in-candidate true pairs, no FP): 0.98011 (+1.277 pts); oracle: perfect decisions after rival drop: 0.97854 (+1.120 pts); (a) add all blocking misses as TP: 0.98801 (+2.067 pts); (b) recover one-to-one losses: 0.96899 (+0.165 pts); (c) select all decision FNs: 0.97369 (+0.635 pts); (d1) remove FP on entities with matches: 0.97065 (+0.332 pts); (d2) remove FP on singletons: 0.96889 (+0.155 pts)
- top features by gain: block_score, r_gap, r_rank, nd_r_logodds_max, nr_class, addr_token_set, nd_r_logodds_sum, l_name_freq, name_ratio, name_jw, nr_min_rel_diff, nd_r_unexp_idf

## Step 1-2: candidate generation (measured on all train rights, all 2.2M train S1 on the left)
New passes px/hs/n2/na/phx exactly as B §Q5 (caps 30/50/30/30/30, top-5, phx all <= 30), run over all 10.32M train
rights (p02). Union restricted to the 8,639,989 right records that have >= 1 union candidate in FIT-SUB/TUNE/HOLDOUT
(all their candidates kept). Cut model: LightGBM (300 trees) on 21 cheap features, trained on union rows of S1
buckets 5-29 only (out-of-sample for all three subsets). True pairs: HOLDOUT 382,075, TUNE 381,948,
FIT-SUB 381,102; entities HOLDOUT 110,449.

| cut rule | HOLDOUT recall | TUNE recall | FIT-SUB recall | holdout pairs / S1 | pairs / right (relevant rights) |
|---|---|---|---|---|---|
| v2 (cand_train K=3) | 94.5396% | 94.4705% | 94.4348% | 14.09 | 3.00 |
| union (v2 + 5 passes, uncut) | 98.3111% | 98.2613% | 98.2679% | 60.24 | 13.49 |
| linear top-3 | 97.8578% | 97.8136% | 97.8150% | 14.05 | 3.00 |
| top-1 OR p>=1e-3 | 98.2001% | 98.1584% | 98.1580% | 7.59 | 1.60 |
| top-2 OR p>=1e-3 | 98.2522% | 98.2024% | 98.1997% | 11.06 | 2.35 |
| top-1 OR p>=1e-4 | 98.2127% | 98.1676% | 98.1674% | 11.24 | 2.41 |
| top-3 OR p>=1e-3 | 98.2760% | 98.2202% | 98.2262% | 15.06 | 3.22 |
