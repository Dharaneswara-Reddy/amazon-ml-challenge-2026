# Submission 06 (20260926-1234 IST)

- Git commit: `7c92407` (code zip: `code_7c92407.zip`)
- Run dir: `/home/toji339/Documents/Amazon_ML_Hackathon/artifacts/runs/20260926T063247_1a0d20f_99304c0b_v4`
- Local holdout macro F0.5: **0.9879** (tune nan, threshold None)
- Holdout candidate recall: 0.9794
- Change vs previous: v4: v3 + self-labelled per-country vocabularies (France learned from test, no labels) + no EM shift for France
- Output stats: {"rule": "expected_f", "train_prior": 0.7204130690832403, "matches_per_entity_by_country": [["India", 809986, 3.3235944818799337], ["US", 663106, 3.3709391861934592], ["France", 259452, 3.407744014307078]], "matching_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/matching_results.tsv", "candidate_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/candidate_pairs.tsv", "n_rows": 1732544, "n_candidate_pairs": 18924178, "n_match_pairs": 5811501, "n_empty_candidate_rows": 205, "n_empty_match_rows": 100084}
- Public leaderboard F0.5: _fill in after upload_

Upload: `matching_results.tsv` as the matching file and the code zip as the code file.
