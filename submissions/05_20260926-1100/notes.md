# Submission 05 (20260926-1100 IST)

- Git commit: `9e2adac` (code zip: `code_9e2adac.zip`)
- Run dir: `artifacts/runs/20260926T032546_e2e7494_678411c5_v3`
- Local holdout macro F0.5: **0.9876** (tune nan, threshold None)
- Holdout candidate recall: 0.9793
- Change vs previous: v4a: v3 + France transductive adaptation (rule-labelled test pairs -> French signature vocab + token log-odds; France features recomputed)
- Output stats: {"rule": "threshold", "train_prior": 0.714396049560202, "matches_per_entity_by_country": [["India", 809986, 3.3312551081129795], ["US", 663106, 3.378589546769295], ["France", 259452, 3.43176387154464]], "matching_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/matching_results.tsv", "candidate_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/candidate_pairs.tsv", "n_rows": 1732544, "n_candidate_pairs": 18924178, "n_match_pairs": 5829011, "n_empty_candidate_rows": 205, "n_empty_match_rows": 100871}
- Public leaderboard F0.5: _fill in after upload_

Upload: `matching_results.tsv` as the matching file and the code zip as the code file.
