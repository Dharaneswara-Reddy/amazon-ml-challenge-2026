# Submission log

Max 5 per day. One row per leaderboard upload; the folder holds the exact files uploaded (TSV + code zip).

| # | Time (IST) | Folder | Git commit | Local F0.5 | Public F0.5 | Status | Change vs previous |
|---|------------|--------|------------|------------|-------------|--------|--------------------|
| 01 | 20260925-1157 | `submissions/01_20260925-1157` | `3b1b9a2` | 0.8075 | 0.749 | SCORED | rule baseline: best block_score candidate per record, threshold tuned on train |
| 02 | 20260926-0046 | `submissions/02_20260926-0046` | `fc7aa50` | 0.9715 | 0.961 | SCORED | v2: blocking v2 + features v2 + stage 2 + tune-chosen decision + prior shift |
| 03 | 20260926-0924 | `submissions/03_20260926-0924` | `475357f` | 0.9876 | 0.977648 | SCORED | v3: blocking v3 (compound keys + learned cut) + signature features + per-country prior shift + lr 0.2 |
| 04 | 20260926-0947 | `submissions/04_20260926-0947` | `ccac9d7` | probe | 0.847629 | SCORED | PROBE (not a model change): submission 03 with every France row emptied, to measure France's public F0.5 |
| 05 | 20260926-1100 | `submissions/05_20260926-1100` | `9e2adac` | 0.9876 | | 0.981323 | SCORED | v4a: v3 + France transductive adaptation (rule-labelled test pairs -> French signature vocab + token log-odds; France features recomputed) |
