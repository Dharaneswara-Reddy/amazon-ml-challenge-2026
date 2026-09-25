# Submission 02 (20260926-0046 IST)

- Git commit: `fc7aa50` (code zip: `code_fc7aa50.zip`)
- Run dir: `artifacts/runs/20260925T161436_b538130_1242e7cd_v2`
- Local holdout macro F0.5: **0.9715** (tune nan, threshold None)
- Holdout candidate recall: 0.9420
- Change vs previous: v2: blocking v2 + features v2 + stage 2 + tune-chosen decision + prior shift
- Output stats: {"rule": "expected_f", "train_prior": 0.6846643998056976, "test_prior_em": 0.5699994751824676, "logit_delta": -0.49344231087999096, "matches_per_entity_by_country": [["India", 809986, 3.183132794887813], ["US", 663106, 3.277765545780011], ["France", 259452, 3.187202257064891]], "matching_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/matching_results.tsv", "candidate_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/candidate_pairs.tsv", "n_rows": 1732544, "n_candidate_pairs": 29908767, "n_match_pairs": 5578725, "n_empty_candidate_rows": 209, "n_empty_match_rows": 108850}
- Public leaderboard F0.5: _fill in after upload_

Upload: `matching_results.tsv` as the matching file and the code zip as the code file.
