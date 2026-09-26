# Submission 08 (20260926-1750 IST)

- Git commit: `4a2d985` (code zip: `code_4a2d985.zip`)
- Run dir: `artifacts/runs/20260926T114709_10acc83_99304c0b_v5`
- Local holdout macro F0.5: **0.9882** (tune nan, threshold None)
- Holdout candidate recall: 0.9783
- Change vs previous: v5: v4 + stage-2 entity features (D-027) + initialism features + admin features dropped (French region/departement mismatch)
- Output stats: {"rule": "expected_f", "train_prior": 0.7268714919564443, "matches_per_entity_by_country": [["India", 809986, 3.329964962357374], ["US", 663106, 3.3811125219798948], ["France", 259452, 3.420902517614048]], "matching_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/matching_results.tsv", "candidate_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/candidate_pairs.tsv", "n_rows": 1732544, "n_candidate_pairs": 18924178, "n_match_pairs": 5826821, "n_empty_candidate_rows": 205, "n_empty_match_rows": 99733}
- Public leaderboard F0.5: _fill in after upload_

Upload: `matching_results.tsv` as the matching file and the code zip as the code file.
