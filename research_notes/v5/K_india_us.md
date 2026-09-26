# K — India+US holdout: error analysis and stage-2/decision experiments (v5)

Baseline run: artifacts/runs/20260926T063247_1a0d20f_99304c0b_v4 (v4). All numbers are HOLDOUT macro F0.5 (buckets 35-39),
decision fitted on TUNE only unless stated. Scripts: notebooks/v5_iu/. Findings appended incrementally.

## T1. v4 reproduction and loss decomposition (holdout)

```
{
 "holdout_f05": 0.9879,
 "entities": 110449,
 "pairs": 538864,
 "true_pairs": 382075,
 "TP": 369537,
 "FP": 674,
 "FP_singletons": 52,
 "decision_FN": 4651,
 "blocking_miss": 6814,
 "one_to_one_loss": 1073,
 "oracle_perfect_decisions": 0.99336,
 "fix_decision_FN": 0.99163,
 "fix_all_FP": 0.98964,
 "fix_FP_singletons": 0.98834,
 "fix_blocking+o2o": 0.99452,
 "f05_india": 0.98603,
 "n_india": 44353,
 "f05_us": 0.98915,
 "n_us": 66096,
 "f05_singleton": 0.99215,
 "share_singleton": 0.05535,
 "loss_pts_singleton": 0.04346,
 "f05_n=1": 0.9578,
 "share_n=1": 0.05391,
 "loss_pts_n=1": 0.22746,
 "f05_n=2-3": 0.98814,
 "share_n=2-3": 0.41149,
 "loss_pts_n=2-3": 0.48786,
 "f05_n>=4": 0.99059,
 "share_n>=4": 0.47925,
 "loss_pts_n>=4": 0.45116
}
```

## T2. Error analysis (v4 holdout) — raw query output

```
FN_split: [(False, False, 158), (False, True, 7), (True, False, 4401), (True, True, 85)]
FN_notbest_winner: [(False, False, 157, 0.071), (False, True, 8, 0.686)]
FN_best_by_src_k0: [(False, False, 2112), (False, True, 45), (True, False, 2289), (True, True, 40)]
TP_by_src: [(False, 178678), (True, 190859)]
FN_best_by_nt: [(1.0, 65), (2.0, 402), (3.0, 892), (4.0, 1126), (5.0, 976), (6.0, 1025)]
FN_best_raw_p_bins: [('<.01', 59, 3), ('<.1', 303, 10), ('<.3', 1145, 44), ('<.5', 1477, 28), ('<.7', 1172, 0), ('<.9', 330, 0)]
FN_best_country: [('india', 1589), ('us', 2897)]
FP: [(False, True, 622, 0.893, 96), (True, True, 52, 0.742, 4)]
FP_by_src: [(False, 346), (True, 328)]
entity_loss_by_kind: [('only_blocking/o2o_missing', 6582, 0.4744), ('partial_fn_in_cands', 4263, 0.3029), ('no_true_in_cands_emptypred', 189, 0.1711), ('tp_and_fp', 610, 0.1375), ('empty_pred_true_in_cands', 88, 0.0797), ('fp_on_singleton', 48, 0.0435), ('all_wrong', 1, 0.0009), ('perfect', 98668, 0.0)]
n1_entities_by_kind: [('true_not_in_cands', 147, 0.1331), ('true_in_cands_empty_pred', 71, 0.0643), ('tp_plus_fp', 74, 0.0301), ('perfect', 5662, 0.0)]
loss_by_nt_country: [('india', 0.0, 2470, 0.9939, 0.0136), ('india', 1.0, 2364, 0.9525, 0.1017), ('india', 2.0, 7585, 0.9842, 0.1088), ('india', 3.0, 10721, 0.9873, 0.1234), ('india', 4.0, 9747, 0.9887, 0.0993), ('india', 5.0, 11466, 0.989, 0.1141), ('us', 0.0, 3643, 0.9909, 0.0299), ('us', 1.0, 3590, 0.9613, 0.1258), ('us', 2.0, 11352, 0.9881, 0.1227), ('us', 3.0, 15791, 0.9907, 0.133), ('us', 4.0, 14505, 0.9915, 0.1115), ('us', 5.0, 17215, 0.9919, 0.1262)]
```

## T3a. Decision-layer variants on v4 pred2 (fit tune -> holdout)

```
{
 "base": {
  "exp_f": 0.9879,
  "thr": 0.98769,
  "exp_f_model": 0.98791,
  "tune_exp_f": 0.98746
 },
 "max_k=8": {
  "exp_f": 0.98786,
  "thr": 0.98769,
  "exp_f_model": 0.98787,
  "tune_exp_f": 0.98742
 },
 "max_k=24": {
  "exp_f": 0.9879,
  "thr": 0.98769,
  "exp_f_model": 0.98791,
  "tune_exp_f": 0.98746
 },
 "missing_bins=40": {
  "exp_f": 0.9879,
  "thr": 0.98769,
  "exp_f_model": 0.98791,
  "tune_exp_f": 0.98745
 },
 "per_source_precal": {
  "exp_f": 0.98793,
  "thr": 0.9877,
  "exp_f_model": 0.9879,
  "tune_exp_f": 0.98746
 },
 "second_choice": {
  "exp_f": 0.9879,
  "first": 0.9879,
  "added_pairs": 1,
  "added_true": 0,
  "removed": 30,
  "removed_true": 21
 }
}
```

**T3a verdict:** the decision layer is saturated on v4 probabilities. max_k 8/24, 40 missing-bins, per-source
(S2/S3 x best/rival) isotonic pre-calibration, second-choice reassignment of rejected R-best pairs: all within
±0.00003 of 0.98790 (noise). Only 165 in-group one-to-one FNs exist (157 of them lost to a rival that was itself
not selected, gap 0.07) — second-choice reassignment recovers none net (+1 pair added, 30 removed incl. 21 TP, because
dropping rejected pairs changes the entity's sum-p). Gains must come from better probabilities or recall.

## T2b. Cross-group one-to-one losses (true holdout pair, right's best candidate is an out-of-group S1)

`{'n': 1073, 'winner_true': 0, 'rival_lt05': 775, 'rival_lt05_true_ge05': 0, 'r03': 0, 'avg_wp': 0.376, 'avg_p': 0.229, 'close_01': 519, 's3': 563}`

(winner_true: the rival is ALSO labelled true — impossible under one-to-one, i.e. GT duplicates; rival_lt05: rival pair p<0.5 so the rival entity would likely reject R -> second-choice could recover it)

**T2b verdict:** all 1,073 cross-group one-to-one losses are low-confidence: the true pair has mean p 0.23 and never
p>=0.5 when the rival is <0.5 (0 cases); the rival is never labelled true. A second-choice / global assignment cannot
recover them — they would be rejected by the entity's decision anyway. One-to-one is not a lever.

## T2c. Examples (random samples, holdout v4)

### Decision FN (R-best, entity has picks, p 0.3-0.7)


### FP with p > 0.99 (label noise?)


### FP on matched entities, p 0.7-0.95


### n=1 entity, true pair in cands, empty prediction

## T2c. Examples (random samples, holdout v4)

### Decision FN (R-best, entity has picks, p 0.3-0.7)
- `S1-47854308` n_true=7 k=6 p=0.573
  - L: Lester Limited Inc | IL, 402 Gershwin Lane, Machesney Park
  - R `S3-197614095`: Lester Limited | 
  - entity's R-best cands (name|p|label, *=selected): Lester Limited North Inc |0.0|F ;; Sollumjax |1.0|T* ;; Lester Límited Inc |1.0|T* ;; Lester Limited |1.0|T* ;; Korcira |1.0|T* ;; Lester Limited Inc |1.0|T* ;; Lester Inc Limited |1.0|T* ;; Lester Limited |0.57|T
- `S1-120518019` n_true=3 k=2 p=0.404
  - L: Ika Aircon Corp | F-24, Uttar Pradesh, Gautam Buddha Nagar, 3Rd Floor, Habitech, Plot No-09, Knowledge Park-Iii, Bishrakh
  - R `S2-743593213`: lka Corp Aircon | F-24, 3RD FLOOR, HABITECH, PLOT NO-09, KNOWLEDGE PARK-III, BISHRAKH, GAUTAM BUDDHA NAGAR, Uttar Pradesh
  - entity's R-best cands (name|p|label, *=selected): Ika  Aircon Corp |1.0|T* ;; Ika Aircon Córp |1.0|T* ;; lka Corp Aircon |0.4|T ;; Mr Ika Aircon Overseas (Corporation) |0.0|F
- `S1-694549875` n_true=4 k=3 p=0.486
  - L: Womens Health Summit Specialists | 103 Fir Avenue, Deer Park, WA
  - R `S3-400321009`: Womens Health Summit Specialists Inc. | 
  - entity's R-best cands (name|p|label, *=selected): Womens Health Summit Specialists LLC #76614 |1.0|T* ;; Womens Health Summit Specialists Co |0.0|F ;; Womens Health Summit Specialists Council |1.0|T* ;; Womens Health Summit Specialists Inc. |0.49|T ;; Womens Hrfht Summit  Specialists |1.0|T*
- `S1-129206761` n_true=6 k=5 p=0.525
  - L: Brenda S. Philbeck, Esq. | 3350 Stimson Road, Barberton, OH
  - R `S3-169884123`: Brenda S. Philbeck,-Esq. | 
  - entity's R-best cands (name|p|label, *=selected): BSPHILBECK.COM |1.0|T* ;; *** Brenda Philbeck, S. Esq. |1.0|T* ;; brenda s. philbeck, esq. |1.0|T* ;; Brenda S. Philbeck,-Esq. |0.52|T ;; Brenda S. Philbeck,-Esq. Co |1.0|T* ;; Brenda Esq. Phileck, S. |1.0|T* ;; Brenda S. Philbeck, Esq. Group |0.0|F
- `S1-722632181` n_true=5 k=4 p=0.542
  - L: Halal Institute | Sikar, Rajasthan, Sikar, Vidhyadhar Charn Singh Gate Ke Pass, Navalgarh Road
  - R `S3-852761337`: Halal Institute LLP | H.no 735 Vidhyadhar Charn Singh Gate Ke Pass, Navalgarh Road, Sikar, RJ
  - entity's R-best cands (name|p|label, *=selected): Halal Institute LLP |0.54|T ;; Halal Institute Limited |0.9|T* ;; HALAL COMMERCIAL |0.0|F ;; halal institute |1.0|T* ;; Halal Institute |1.0|T* ;; Halal Institute Limited |0.99|T*
- `S1-122438116` n_true=2 k=1 p=0.399
  - L: Admire Gems | Pune City, Sr No 4516, Shop No-07 Vasant Center, Haveli Pune Pimpri Chinchwad Mcorp, Maharashtra, Pune
  - R `S2-607787163`: Admire Gr | Maharashtra, NO. 2-359 SR NO 4516, SHOP NO-07 VASANT CENTER, HAVELI PUNE PIMPRI CHINCHWAD MCORP, PUNE CITY
  - entity's R-best cands (name|p|label, *=selected): Admire Gr |0.4|T ;; >> ADMIRE GS EXPORTS |0.0|F ;; ADMIRE GEMS |1.0|T*
- `S1-295945643` n_true=3 k=2 p=0.611
  - L: Johnson & Locke Cadenza P.C. | 36 Rochester Street, Chicopee, MA
  - R `S3-170712050`: Inc Johnson & Locke Cadenza | 46 Rochester St, Chicopee, Massachusetts
  - entity's R-best cands (name|p|label, *=selected): Johnson & Locke Cadenza Ltd. | www.johnson.com |0.0|F ;; Johnson & Locke Cadenza |1.0|T* ;; Johnson  & Lcobe Cadenza P.C. |1.0|T* ;; Inc Johnson & Locke Cadenza |0.61|T
- `S1-291211748` n_true=4 k=2 p=0.644
  - L: Silver Tri-State Keystone | 10418 Osprey Drive, Pineville, NC
  - R `S2-659808129`: Silver Keystone Tbr-igdte | 10418 OSPREY DRIVE, PINEVILLE, NC
  - entity's R-best cands (name|p|label, *=selected): SILVER TRI-STATE CENTER |1.0|T* ;; Silver Trm-cctete Keystone |0.54|T ;; Silver Keystone Tbr-igdte |0.64|T ;; Silver Tri-State Keysetorne Ltd |0.0|F ;; silvertristate.com |1.0|T*

### FP with p > 0.99 (label noise?)
- `S1-342226581` n_true=6 k=7 p=0.993
  - L: Fleurette Hawthorne, DPM, M.D., P.C. LP | 514 Stanley Avenue, Clarksburg, WV
  - R `S2-848900128`: Fleurette Hatwornge, DP, M.D., P.C. LP | 514 STANLEY AVE, CLARKSBURG, PO BOX 4395, WV
  - entity's R-best cands (name|p|label, *=selected): Fleurette Hawthorne, DPM, M.D., P.C. |1.0|T* ;; Hawthorne,, Fleurette DPM, P.C. M.D., LP |1.0|T* ;; Fleurette Háwthorne, Háwthorne, DPM, M.D., |1.0|T* ;; Fleurette Hawthorne, DPM, M.D., P.C. |1.0|T* ;; Fleurette Hatwornge, DP, M.D., P.C. LP |0.99|F* ;; FLEURETTE HAWTHORNE, DPM, M.D., P.C. LP |1.0|T* ;; Fleurette Hawthorne, DPM, M.D., (P.C.) |1.0|T*
- `S1-990501127` n_true=0 k=1 p=1.0
  - L: Fort Hosiery Care | Powvala Building, 251, Hornby Road, Fort, Mumbai, Maharashtra
  - R `S2-276592775`: Service Fort Care | NO 11/7 POWVALA BUILDING, 251, HORNBY ROAD, GREATER BOMBAY, MUMBAI, Maharashtra
  - entity's R-best cands (name|p|label, *=selected): Fort Hosiery Care Private Limited |0.0|F ;; Service Fort Care |1.0|F*
- `S1-762619772` n_true=3 k=4 p=0.991
  - L: Meeks Platinum LLC | 6001 Southern Avenue, Unit 52, Mesa, AZ
  - R `S3-619451041`: abreliablelloyds.com | 6001 Southern Ave, Unit 52, Mesa, Arizona
  - entity's R-best cands (name|p|label, *=selected): Meeks Platinum Platinum L.L.C. |1.0|T* ;; Meeks Plotinum Plotinum LLC |1.0|T* ;; abreliablelloyds.com |0.99|F* ;; Meeks Platinum |1.0|T* ;; Meeks  Partners Platinum |0.0|F
- `S1-602125129` n_true=4 k=5 p=0.997
  - L: Caex Companies LLC | 812 16th Street, Decatur, IL
  - R `S3-177842540`: Veraquo Companies Llc | 812- 16th St, Decatur, Illinois
  - entity's R-best cands (name|p|label, *=selected): [LLC] Caex Companies |1.0|T* ;; Caex Cemopnies LLC |1.0|T* ;; Veraquo Companies Llc |1.0|F* ;; caexcompanies.com |1.0|T* ;; Caex Companies |1.0|T*
- `S1-103282487` n_true=3 k=4 p=0.998
  - L: Golden Massage Corporation | OH, 1440 Beecham Street, Toledo
  - R `S2-363869138`: GOLDEN MMFSNAE CORP | 8443 BEECHAM ST, TOLEDO, OH
  - entity's R-best cands (name|p|label, *=selected): Golden Mássage Corporation |1.0|T* ;; GOLDEN MMFSNAE CORP |1.0|F* ;; Golden-Masmsageg Corp |0.09|F ;; Muriel H. Oconnell, MD Group |0.0|F ;; GOLDEN MASSAGE |1.0|T* ;; GOLDEN MASSAGE CORPORATION |1.0|T*
- `S1-357639148` n_true=4 k=5 p=1.0
  - L: West Chemical | 312 Pheasant Drive, Fl 1, Lexington, NC
  - R `S2-846895859`: West Chemical Corp | 312 PHEASANT DR, LEXINGTON, NC
  - entity's R-best cands (name|p|label, *=selected): WEST CHEMICAL CO |0.98|T* ;; West Chémical |1.0|T* ;; West Services | www.westservi.com |1.0|T* ;; ILA Pmha Inc |0.0|F ;; Westchemical.Com |1.0|T* ;; West Chemical Corp |1.0|F*

### FP on matched entities, p 0.7-0.95
- `S1-539979860` n_true=2 k=3 p=0.756
  - L: Vio Capital Partners PC | 403 Commerce Street, Gainesville, TX
  - R `S3-360234297`: Vio Capital Partners | 
  - entity's R-best cands (name|p|label, *=selected): Vio Capital Pártners PC |1.0|T* ;; Vio Capital |1.0|T* ;; Vio Capital Partners Westgate |0.0|F ;; Vio Capital Partners |0.76|F* ;; Cooper Capital, LLC |0.0|F
- `S1-963204357` n_true=3 k=4 p=0.941
  - L: Golden Tavern | 229 Superior Avenue, WI, City Of Tomah
  - R `S3-43799096`: Vantagekelo | 229 Superior Avenue, Tomah, Wisconsin
  - entity's R-best cands (name|p|label, *=selected): 6OLDEN TÁVÉRN |1.0|T* ;; GOLDEN TAVERN CORP |0.01|F ;; Vantagekelo |0.94|F* ;; GOLDEN TAVERN TAVERN |0.01|F ;; Golden Tcaiedenr LLC |0.0|F ;; Golden Tavern |1.0|T* ;; Golden Tavern |0.01|F ;; Golden Tavern |1.0|T* ;; Golden Tavern [[Ltd]] |0.01|F ;; GOLDEN TAVERN! CORPORATION |0.01|F ;; Golden Távern! |0.01|F
- `S1-504528813` n_true=5 k=4 p=0.787
  - L: Sree Foundation Private Limited | #104, 10Th Main, 6Th, Sector, Hsr Layout, Bangalore South, Bangalore, Karnataka
  - R `S3-839874429`: Private Sree Foundation Ltd | 003, Bengaluru, Bangalore, KA
  - entity's R-best cands (name|p|label, *=selected): ಶ್ರೀ ಫೌಂಡೇಶನ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್ |1.0|T* ;; foundationprivate.com |1.0|T* ;; Private Sree Foundation Ltd |0.79|F* ;; Evogild |1.0|T*
- `S1-108901001` n_true=3 k=5 p=0.918
  - L: Smart Hospitality Services L.L.C. | 18 Granada Park, Boston, MA
  - R `S3-407668202`: Smart Hospitality Services LLC | 20 Granada Park, Roxbury, Massachusetts
  - entity's R-best cands (name|p|label, *=selected): Smart Hospitality-Services L.L.C. |1.0|T* ;; Smart  Hospitality Services |0.97|F* ;; Smart Hospitality Services LLC |0.92|F* ;; L.L.C. Smart Hospitality Srivies |1.0|T* ;; taylor skin care care có |0.0|F ;; Smart Services Hospitality L.L.C. (ID: 80207) |1.0|T*
- `S1-148111960` n_true=4 k=5 p=0.816
  - L: Dental Royal Care | 3944 Easy Street, Dickinson, TX
  - R `S2-222346954`: Dental Royal Care | 
  - entity's R-best cands (name|p|label, *=selected): Dental Royal |1.0|T* ;; Dental Róyal Care North |0.0|F ;; SERVICES ROYAL DENTAL |1.0|T* ;; Dental Royal Care |0.82|F* ;; Dental Róyal Cáre |1.0|T* ;; Gildhalodelta d/b/a Dental Royal Care |1.0|T*
- `S1-929661298` n_true=2 k=3 p=0.93
  - L: Shuster Partners Co | 17150 Dixie Hwy, KY, Louisville
  - R `S3-752207310`: SHUSTER PARTNERS PARTNERS | 17155d Dixie Hwy, Louisville, Kentucky
  - entity's R-best cands (name|p|label, *=selected): SHUSTER PARTNERS PARTNERS |0.93|F* ;; Shuster Partners |1.0|T* ;; Partners Shuster Co |1.0|T*

### n=1 entity, true pair in cands, empty prediction
- `S1-906695217` n_true=1 k=0 p=0.27
  - L: Ek Welfare Society | C/O Jitendra Bhati, Vill. Yakubpur Sec-85, Noida, Gautam Buddha Nagar, Uttar Pradesh
  - R `S2-557086231`: Ek Welfare (Society) | 
  - entity's R-best cands (name|p|label, *=selected): Ek Welfare-Society Welfare-Society Infratech |0.01|F
- `S1-127932081` n_true=1 k=0 p=0.326
  - L: Classic Lending Inc | 15918 Harney Street, Omaha, NE
  - R `S2-512704908`: Classic Lending Inc. | 
  - entity's R-best cands (name|p|label, *=selected): Classic Lending Inc. |0.33|T ;; Classic Lending Westgate Inc |0.0|F
- `S1-847193744` n_true=1 k=0 p=0.179
  - L: Pediatric Dental Grand Partners | 1124 Chambers Road, Unit 221, Columbus, OH
  - R `S2-262857219`: PEDIATRIC DNETLA GRAND PARTNERS | 
  - entity's R-best cands (name|p|label, *=selected): PEDIATRIC DNETLA GRAND PARTNERS |0.18|T ;; Pediatric Dental Grand Partners Inc |0.0|F
- `S1-16997316` n_true=1 k=0 p=0.182
  - L: Midwest Sony | 1119 Pembrook Drive, Peoria, IL
  - R `S3-602683534`: Midwest Sóny | 
  - entity's R-best cands (name|p|label, *=selected): MIDWEST SONY  SUMMIT |0.0|F ;; Midwest Sóny |0.18|T
- `S1-521618154` n_true=1 k=0 p=0.324
  - L: Chennai City Corporation Producer Private Limited | No.1, Ground Floor, Throwpathy Amman Koil, 2Nd Street, Velachery East, Chennai City Corporation, Chennai, Tamil Nadu
  - R `S3-993199785`: Chennai  City Corporation Producer Private | Dor No 1, Chennai, Rajalakshmi Nagar, TN
  - entity's R-best cands (name|p|label, *=selected): Private Chennai City Corporation Products Limited |0.03|F ;; Chennai  City Corporation Producer Private |0.32|T
- `S1-331099616` n_true=1 k=0 p=0.161
  - L: Heartland Summit Deep LLC | 18925 Lansbrook Lane, Mokena, IL
  - R `S3-101294884`: Heartland Summit Deep | 
  - entity's R-best cands (name|p|label, *=selected): Heartland Summit Deep |0.16|T

## T2d. The dominant decision-loss population: right records with an EMPTY address

Holdout R-best pairs split by the right record's normalized address being empty:

| R address | R-best pairs | true | TP | FN (R-best) | FP |
|---|---|---|---|---|---|
| present | 494,937 | 361,882 | 360,299 | 1,583 | 508 |
| **empty** | 24,090 (4.6%) | 12,141 | 9,238 (76% recall) | **2,903 (65% of all R-best FNs)** | 166 |

For empty-address R-best pairs that are FALSE (11,949): 518 R have no GT match at all; **6,113 R have a true S1 that is
NOT among R's candidates** (blocking miss of R's own pair; the R-best L is a similar-name S1, mostly rejected, p 0.05);
5,318 R have their true S1 among the candidates but ranked below this L (same-name S1 ambiguity, p 0.26).
The generator never blanks a distractor's address (v3/F), so an empty-address R is ~98% a true copy of SOME S1 — the
problem is WHICH S1 among same-name entities, which the text cannot resolve. Examples (T2c) confirm: FNs are
"Lester Limited | (empty)", "Brenda S. Philbeck,-Esq. | (empty)", n=1 misses are "Classic Lending Inc. | (empty)" p 0.33.
FPs with p>0.99 are "same address, different/random name" records (label noise or generator aliases like "Sollumjax" = T,
"Veraquo Companies Llc" = F) — irreducible.

## T4. Blocking misses (holdout) and a sibling-propagation candidate rule

```
{
 "misses": 6814,
 "miss_R_noaddr": 3734,
 "miss_by_src": [
  [
   "S2",
   3096
  ],
  [
   "S3",
   3718
  ]
 ],
 "miss_name_core_equal": 1936,
 "miss_jw_name_bins": [
  [
   ".7-.85",
   713
  ],
  [
   ".85-.95",
   2099
  ],
  [
   "<.7",
   1424
  ],
  [
   ">=.95",
   2578
  ]
 ],
 "prop_new_pairs": [
  7708772,
  59610
 ],
 "prop_new_true": 2219,
 "prop_new_by_rdf": [
  [
   2,
   6199,
   38
  ],
  [
   3,
   7272,
   34
  ],
  [
   4,
   9420,
   28
  ],
  [
   5,
   11809,
   18
  ],
  [
   6,
   13853,
   28
  ],
  [
   7,
   15596,
   27
  ],
  [
   8,
   16012,
   21
  ],
  [
   9,
   16527,
   20
  ],
  [
   10,
   7612084,
   2005
  ]
 ],
 "holdout_S1": 110449
}
```

**T4 verdict:** 6,814 holdout blocking misses; **3,734 (55%) have an EMPTY right address** (vs 4.6% of candidate pairs),
1,936 have name_core exactly equal to the S1's, 2,578 JW(name_core) >= 0.95. So most misses are name-only copies of
crowded names: the blocking passes cap them out (df caps) AND, if generated, the classifier faces the same-name ambiguity
as T2d (empty-address R-best recall 76%, precision ~50% base rate). Sibling propagation (exact name_core of an already
selected sibling right, same country): 7.7M new pairs for 2,219 true (df<=9: ~100k pairs for ~230 true) — useless.
Cheap cut changes (v3/E measured on this same holdout): top-2 OR p>=1e-3 = +0.052 pp recall (~200 pairs) for +46%
candidate pairs (7.59 -> 11.06 / S1); top-1 OR p>=1e-4 = +0.013 pp for +48% pairs. At the current ~0.5 acceptance rate
for recovered hard pairs that is < +0.01 F-pt: not worth the pair cost. Blocking is NOT a cheap lever any more.

- stage-2 retrain `top42` feats=top seed=42 params={'learning_rate': 0.2}: holdout expF **0.98783** (thr 0.98759), tune expF 0.98747, valid logloss 0.016257, iters 189, 695 s. top gain: [('s2_p1', np.float64(52.72)), ('s2_r_margin', np.float64(39.15)), ('s2_l_gap', np.float64(6.27)), ('s2_l_n90', np.float64(0.54)), ('s2_l_max', np.float64(0.4)), ('s2_r_share', np.float64(0.18)), ('s2_l_sum', np.float64(0.07)), ('cut_p', np.float64(0.06))]

- stage-2 retrain `topE42` feats=topE seed=42 params={'learning_rate': 0.2}: holdout expF **0.98818** (thr 0.98801), tune expF 0.98779, valid logloss 0.015131, iters 133, 247 s. top gain: [('s2_p1', np.float64(76.84)), ('s2_r_margin', np.float64(22.08)), ('e_p_over_lmax', np.float64(0.12)), ('e_l_n_best', np.float64(0.09)), ('s2_r_share', np.float64(0.09)), ('s2_l_gap', np.float64(0.08)), ('e_ls_max_other', np.float64(0.05)), ('e_l_sum_best', np.float64(0.05))]

- stage-3 on tune (5-fold OOF) `s3_v4` from pred2_holdout.parquet: holdout expF **0.98808** (thr 0.98789), tune-OOF expF 0.98761, iters [295, 267, 216, 257, 222]. top gain [('s3_p1', np.float64(76.68)), ('s3_r_margin', np.float64(21.65)), ('s2_r_margin', np.float64(0.41)), ('s2_p1', np.float64(0.34)), ('e_l_next_gap_3', np.float64(0.13)), ('e_p_over_lmax_3', np.float64(0.07))]

- stage-2 retrain `topE7` feats=topE seed=7 params={'learning_rate': 0.2}: holdout expF **0.98814** (thr 0.98802), tune expF 0.98775, valid logloss 0.01519, iters 116, 255 s. top gain: [('s2_r_margin', np.float64(69.28)), ('s2_p1', np.float64(29.09)), ('e_l_next_gap', np.float64(0.66)), ('e_p_over_lmax', np.float64(0.13)), ('e_l_n_best', np.float64(0.09)), ('s2_r_share', np.float64(0.07)), ('e_r_second', np.float64(0.06)), ('e_ls_max_other', np.float64(0.06))]

- stage-3 on tune (5-fold OOF) `s3_topE42` from pred_topE42_holdout.parquet: holdout expF **0.98812** (thr 0.98805), tune-OOF expF 0.98773, iters [240, 194, 211, 192, 196]. top gain [('s3_p1', np.float64(78.09)), ('s3_r_margin', np.float64(20.65)), ('e_l_next_gap_3', np.float64(0.29)), ('s2_r_margin', np.float64(0.18)), ('s2_p1', np.float64(0.17)), ('s3_l_gap', np.float64(0.09))]

### Paired bootstrap vs `v4` (0.98790); decision expected_f fitted on tune; delta in F-points, 95% CI over entities

| variant | holdout F0.5 | delta pts | 95% CI |
|---|---|---|---|
| top42 | 0.98783 | -0.008 | [-0.021, +0.007] |
| topE42 | 0.98818 | +0.028 | [+0.012, +0.042] |
| topE7 | 0.98814 | +0.024 | [+0.008, +0.041] |
| topE42+topE7 | 0.98830 | +0.040 | [+0.026, +0.056] |
| s3_v4 | 0.98808 | +0.018 | [+0.008, +0.028] |
| s3_topE42 | 0.98812 | +0.022 | [+0.004, +0.039] |
| v4+topE42+topE7 | 0.98820 | +0.030 | [+0.014, +0.044] |

### Per-country / entity-size holdout F0.5

| spec | all | india | us | n=0 | n=1 | n=2-3 | n>=4 |
|---|---|---|---|---|---|---|---|
| v4 | 0.9879 | 0.98603 | 0.98915 | 0.99215 | 0.9578 | 0.98814 | 0.99059 |
| topE42+topE7 | 0.9883 | 0.98619 | 0.98972 | 0.99117 | 0.96011 | 0.9885 | 0.99098 |

## T3 summary so far (stage-2 / stacking; all measured on holdout, decision fitted on tune)

Memory forced a reduced stage 2: top-60 v4 stage-2 features by gain (99.85% of gain) = `top`; this reproduces v4
(0.98783 vs 0.98790, -0.008 pt, CI [-0.021,+0.007]). `topE` = top + 15 entity/rival features from stage-1 p
(t3_feats.py `esql`: e_l_n_best, e_l_sum_best, e_l_rank_best, e_l_next_gap, e_l_prev_gap, e_l_n10, e_l_n30,
e_ls_rank / e_ls_n50 / e_ls_max_other (within S1 x source), e_r_n, e_r_second, e_r_other_lgap (rival S1's max p minus
the rival pair's p), e_r_other_ln50, e_p_over_lmax). Stage-2 valid logloss 0.01626 -> 0.01513 (-7%).
Stage 3 (tune-only 5-fold stacking on p2 competition features) helps v4 (+0.018) but adds nothing on top of topE
(it re-discovers the same entity context). Decision-layer changes: nothing.

- stage-2 retrain `topE13` feats=topE seed=13 params={'learning_rate': 0.2}: holdout expF **0.98828** (thr 0.98806), tune expF 0.9878, valid logloss 0.015154, iters 119, 279 s. top gain: [('s2_p1', np.float64(52.1)), ('s2_r_margin', np.float64(46.51)), ('e_l_next_gap', np.float64(0.28)), ('s2_r_share', np.float64(0.19)), ('e_p_over_lmax', np.float64(0.13)), ('e_l_n_best', np.float64(0.1)), ('e_ls_max_other', np.float64(0.05)), ('s2_l_gap', np.float64(0.05))]

- stage-2 retrain `topE42lr1` feats=topE seed=42 params={'learning_rate': 0.1}: holdout expF **0.98836** (thr 0.9881), tune expF 0.98783, valid logloss 0.015003, iters 369, 286 s. top gain: [('s2_p1', np.float64(62.07)), ('s2_r_margin', np.float64(36.26)), ('e_p_over_lmax', np.float64(0.45)), ('e_l_next_gap', np.float64(0.14)), ('e_l_n_best', np.float64(0.1)), ('s2_r_share', np.float64(0.1)), ('e_ls_max_other', np.float64(0.06)), ('e_l_sum_best', np.float64(0.05))]

### Paired bootstrap vs `v4` (0.98790); decision expected_f fitted on tune; delta in F-points, 95% CI over entities

| variant | holdout F0.5 | delta pts | 95% CI |
|---|---|---|---|
| topE13 | 0.98828 | +0.038 | [+0.023, +0.055] |
| topE42+topE7+topE13 | 0.98821 | +0.031 | [+0.015, +0.048] |
| topE42lr1 | 0.98836 | +0.046 | [+0.031, +0.062] |
| topE42lr1+topE13+topE7 | 0.98818 | +0.028 | [+0.011, +0.044] |

- Seeds (topE, lr 0.2): 42 -> 0.98818, 7 -> 0.98814, 13 -> 0.98828 (seed SD ~0.007 pt). Bagging (logit mean):
  2 seeds 0.98830, 3 seeds 0.98821 -> **bagging is noise-level, not a lever**. Mixing lr 0.1 and lr 0.2 models: 0.98818.
- **Stage-2 learning rate 0.1** (topE, seed 42): 0.98836 (+0.046 pt vs v4, CI [+0.031,+0.062]; +0.018 vs same seed at
  lr 0.2), valid logloss 0.01500, 369 iters (~5 min, 5 threads).

## T2e. How much of the loss is irreducible same-name ambiguity?

True pairs whose right record has an EMPTY address (all 49.4k in pred2_holdout incl. rival-context rows), by the number
of OTHER candidate S1s of the same right whose name_core is IDENTICAL to the true S1's ("twins"):

| identical-name twin S1s among R's candidates | true pairs | mean p2 | accepted-ish (p>=0.5) |
|---|---|---|---|
| 0 | 27,791 | 0.92 | 94% |
| 1 | 6,443 | 0.44 | 42% |
| 2 | 4,193 | 0.30 | 23% |
| >=3 | 10,991 | 0.18 | 11% |

44% of empty-address true pairs have >=1 identical-name twin: the text cannot tell which S1 the name-only copy belongs to
(the model still separates somewhat: twins' mean p 0.16-0.18 vs 0.18-0.44 for the true S1, via legal forms / noise
details). With ~55% of blocking misses and ~65% of decision FNs being empty-address rights, roughly **0.4-0.5 of the
1.21 lost F-points on India+US are this ambiguity** and cannot be recovered by better features or decisions.
Realistic India+US ceiling with the current candidate set is therefore ~0.990-0.991, not 0.992+.

Check for an entity-level tie-breaker: over 97,982 (true S1, identical-name twin S1) pairs of empty-address rights, twins
have matches themselves (mean n_true 3.45 vs 4.30; only 5.8% of twins are singletons) and the true S1 has more
confident (p>=0.9) candidates in 21.4% of cases vs the twin in 19.7% -> **no usable tie-breaker; irreducible**.

- stage-2 retrain `topE42half` feats=topE seed=42 params={'learning_rate': 0.1}: holdout expF **0.98823** (thr 0.98807), tune expF 0.98773, valid logloss 0.015234, iters 269, 132 s. top gain: [('s2_r_margin', np.float64(59.38)), ('s2_p1', np.float64(38.85)), ('e_p_over_lmax', np.float64(0.46)), ('e_l_next_gap', np.float64(0.13)), ('e_l_n_best', np.float64(0.09)), ('s2_r_share', np.float64(0.09)), ('e_ls_max_other', np.float64(0.07)), ('e_l_sum_best', np.float64(0.06))]

- **Training-data learning curve (stage 2, topE lr 0.1 seed 42):** fitA only (2.70M rows, half) 0.98823 vs full fit
  (5.39M) 0.98836 -> doubling stage-2 data = +0.013 pt (logloss 0.01523 -> 0.01500, -1.5%). Stage 1 (v4 logs): half-fit
  fold models tune logloss 0.01982 vs full 0.01885 (-4.9% per doubling). Buckets 40-99 (60% of train) are UNUSED; using
  them (3x fit) is estimated at +0.03-0.06 pt (log-linear extrapolation), but stage-1 training at 16M rows x 147 features
  does not fit this laptop (needs easy-negative subsampling with weights, or a bigger box).

- stage-2 retrain `topE7lr1` feats=topE seed=7 params={'learning_rate': 0.1}: holdout expF **0.98819** (thr 0.98806), tune expF 0.98779, valid logloss 0.015048, iters 352, 232 s. top gain: [('s2_r_margin', np.float64(66.02)), ('s2_p1', np.float64(32.1)), ('e_l_next_gap', np.float64(0.45)), ('e_p_over_lmax', np.float64(0.38)), ('e_l_n_best', np.float64(0.1)), ('s2_r_share', np.float64(0.08)), ('e_ls_max_other', np.float64(0.06)), ('e_r_second', np.float64(0.06))]

### Paired bootstrap vs `v4` (0.98790); decision expected_f fitted on tune; delta in F-points, 95% CI over entities

| variant | holdout F0.5 | delta pts | 95% CI |
|---|---|---|---|
| topE7lr1 | 0.98819 | +0.029 | [+0.012, +0.045] |
| topE42lr1+topE7lr1 | 0.98843 | +0.053 | [+0.037, +0.069] |
| topE42half | 0.98823 | +0.033 | [+0.018, +0.048] |

### Per-country / entity-size holdout F0.5

| spec | all | india | us | n=0 | n=1 | n=2-3 | n>=4 |
|---|---|---|---|---|---|---|---|
| v4 | 0.9879 | 0.98603 | 0.98915 | 0.99215 | 0.9578 | 0.98814 | 0.99059 |
| topE42lr1 | 0.98836 | 0.98629 | 0.98974 | 0.99231 | 0.9592 | 0.98858 | 0.99099 |
| topE42lr1+topE7lr1 | 0.98843 | 0.98637 | 0.98981 | 0.99149 | 0.96087 | 0.98859 | 0.99103 |

- stage-2 retrain `topE42lr05` feats=topE seed=42 params={'learning_rate': 0.05}: holdout expF **0.98837** (thr 0.98819), tune expF 0.9879, valid logloss 0.014907, iters 1017, 601 s. top gain: [('s2_p1', np.float64(63.08)), ('s2_r_margin', np.float64(33.63)), ('e_p_over_lmax', np.float64(1.69)), ('e_l_next_gap', np.float64(0.24)), ('s2_r_share', np.float64(0.14)), ('e_l_n_best', np.float64(0.11)), ('s2_l_max', np.float64(0.07)), ('s2_l_gap', np.float64(0.07))]

### Paired bootstrap vs `v4` (0.98790); decision expected_f fitted on tune; delta in F-points, 95% CI over entities

| variant | holdout F0.5 | delta pts | 95% CI |
|---|---|---|---|
| topE42lr05 | 0.98837 | +0.047 | [+0.030, +0.060] |
| topE42lr05+topE42lr1+topE7lr1 | 0.98822 | +0.032 | [+0.016, +0.048] |

- decomposition `v4`: {'spec': 'v4', 'F': np.float64(0.9879), 'TP': 369537, 'FP': 674, 'FN_in_cands': 4651, 'oracle_on_cands': np.float64(0.99336), 'fix_FN': np.float64(0.99163), 'fix_FP': np.float64(0.98964)}

- decomposition `topE42lr1+topE7lr1`: {'spec': 'topE42lr1+topE7lr1', 'F': np.float64(0.98843), 'TP': 369939, 'FP': 582, 'FN_in_cands': 3865, 'oracle_on_cands': np.float64(0.99303), 'fix_FN': np.float64(0.99146), 'fix_FP': np.float64(0.99001)}

- lr 0.05 (topE, seed 42): 0.98837 (+0.047, CI [+0.030,+0.060]), logloss 0.01491, 1017 iters, 10 min -> same as lr 0.1.
- lr 0.1 seed 7: 0.98819; 2-seed lr-0.1 bag 0.98843 (+0.053) but 3-model bag (lr .05/.1/.1) 0.98822 -> bags are noise.
- Decomposition of the lr-0.1 2-seed bag vs v4: FP 674 -> 582, in-candidate FN 4,651 -> 3,865, TP +402;
  India 0.98603 -> 0.98637, US 0.98915 -> 0.98981; n=1 entities 0.9578 -> 0.9609; singletons 0.99215 -> 0.99149.

## T5. Ranked changes (measured on holdout, decision fitted on tune; v4 = 0.98790)

| # | change | holdout F0.5 | delta (F-pts) | status |
|---|---|---|---|---|
| 1 | stage-2 entity/rival features `E` from stage-1 p (15 cols, t3_feats.py `esql`) | 0.98814-0.98828 (3 seeds, mean 0.98820) | **+0.030** (CI excl. 0 for every seed) | measured, ship |
| 2 | + stage-2 learning_rate 0.1 (or 0.05) instead of 0.2 | 0.98819-0.98837 (mean 0.98831) | +0.011 on top of #1 (noisy, 2 of 3 runs clearly up) | measured, cheap |
| 3 | use unused train buckets 40-99 for fit (3x data) | not run | est. +0.03-0.06 (learning curve: 2x stage-2 data = +0.013; stage-1 logloss -4.9%/doubling) | needs RAM (subsample easy negatives) |
| 4 | stage 3 (tune-only stacking on p2) | v4: 0.98808 (+0.018); on topE: -0.006 | 0 once #1 is in | drop |
| 5 | seed bagging of stage 2 (2-3 seeds, logit mean) | 0.98821-0.98843 | 0 +- 0.01 (inconsistent) | drop |
| 6 | decision: max_k, missing bins, per-source calibration, second-choice reassignment | 0.98786-0.98793 | 0 | drop |
| 7 | blocking: top-2 cut / p>=1e-4 / sibling propagation | (v3/E recall numbers) | < +0.01, +46% pairs | drop |

**Combined expected India+US holdout: ~0.9883-0.9884 with #1+#2 (measured), ~0.9886-0.9889 if #3 also lands.**
0.992 is not reachable on India+US with this candidate set: ~0.4-0.5 of the 1.2 lost points are name-only (empty-address)
right records whose S1 has an identical-name twin among the candidates (T2e), and the oracle with perfect decisions on
current candidates is 0.9934.

### Implementation notes for ber (not applied; code/business_entity_resolution untouched)
- `ber/stage2.py`: add the 15 E columns to `competition_sql` (window/aggregate SQL is in notebooks/v5_iu/t3_feats.py
  `esql`: per-rid top-2 lids/p, per-lid max/n50, R-best counts/sums, lead/lag gaps, (lid, source) windows) and to
  `STAGE2_FEATURES` so they flow into s2_{fit,tune,holdout,test}. Computed on the same (lid, rid, p) relation incl. rival
  context rows, so train/test fidelity is unchanged. Cost: +~90 s on fit (5.4M rows), trivial on tune/holdout.
- `ber/pipeline.train_stage`: stage-2 params `learning_rate: 0.1` (config `model.params_stage2`), rounds cap 5000.
  Stage-2 training rose from ~130 to ~370 iterations (~5 min at 5 threads).
- The measured runs used the top-60 v4 stage-2 features (memory limit here); the full 164-feature stage 2 + E should
  be at least as good (top-60 reproduced v4 within -0.008 pt).
- Everything reusable is in artifacts/v5_iu/ (s2x_* matrices, pred_* files, t3_results.jsonl).

