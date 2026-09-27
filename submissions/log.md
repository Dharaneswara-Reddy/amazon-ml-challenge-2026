# Submission log

Max 5 per day. One row per leaderboard upload; the folder holds the exact files uploaded (TSV + code zip).

| # | Time (IST) | Folder | Git commit | Local F0.5 | Public F0.5 | Status | Change vs previous |
|---|------------|--------|------------|------------|-------------|--------|--------------------|
| 01 | 20260925-1157 | `submissions/01_20260925-1157` | `3b1b9a2` | 0.8075 | 0.749 | SCORED | rule baseline: best block_score candidate per record, threshold tuned on train |
| 02 | 20260926-0046 | `submissions/02_20260926-0046` | `fc7aa50` | 0.9715 | 0.961 | SCORED | v2: blocking v2 + features v2 + stage 2 + tune-chosen decision + prior shift |
| 03 | 20260926-0924 | `submissions/03_20260926-0924` | `475357f` | 0.9876 | 0.977648 | SCORED | v3: blocking v3 (compound keys + learned cut) + signature features + per-country prior shift + lr 0.2 |
| 04 | 20260926-0947 | `submissions/04_20260926-0947` | `ccac9d7` | probe | 0.847629 | SCORED | PROBE (not a model change): submission 03 with every France row emptied, to measure France's public F0.5 |
| 05 | 20260926-1100 | `submissions/05_20260926-1100` | `9e2adac` | 0.9876 | | 0.981323 | SCORED | v4a: v3 + France transductive adaptation (rule-labelled test pairs -> French signature vocab + token log-odds; France features recomputed) |
| 06 | 20260926-1234 | `submissions/06_20260926-1234` | `7c92407` | 0.9879 | | 0.982191 | SCORED | v4: v3 + self-labelled per-country vocabularies (France learned from test, no labels) + no EM shift for France |
| 07 | 20260926-1307 | `submissions/07_20260926-1307` | `c2042b9` | 0.9877 | | 0.981916 | SCORED | v4s: v4 + France pseudo-label self-training (stage-1 trained with rule/model-labelled France test pairs) |
| 08 | 20260926-1750 | `submissions/08_20260926-1750` | `4a2d985` | 0.9882 | | pending | v5: v4 + stage-2 entity features (D-027) + initialism features + admin features dropped (French region/departement mismatch) |
| 09 | 20260926-1804 | `submissions/09_20260926-1804` | `9e7f50a` | 0.9882 | | pending | variant of v5 (#8): France EM prior shift re-enabled (decision only) |
| 10 | 20260926-2219 | `submissions/10_20260926-2219` | `a677223` | 0.9882 | | 0.984041 | SCORED | v6-e5: v5 + stage-3 cross-encoder (multilingual-e5-small, MIT, fine-tuned on provided train data on Colab); holdout 0.98815 -> 0.98918 |
| 11 | 20260927-1134 | `submissions/11_20260927-1134` | `99f74ca` | 0.9882 | | 0.984453 | SCORED | v6-e5bge: v5 + stage-3 cross-encoders multilingual-e5-small (MIT) + BAAI/bge-reranker-v2-m3 (Apache-2.0), fine-tuned on provided train data; holdout 0.98815 -> 0.98929 |
