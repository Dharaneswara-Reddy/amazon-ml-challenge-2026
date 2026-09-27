# Submission 16 (20260927-2027 IST)

- Git commit: `774d279` (code zip: `code_774d279.zip`)
- Run dir: `artifacts/runs/20260927T140904_774d279_e6b1b643_v5x`
- Local holdout macro F0.5: **0.9886** (tune nan, threshold None)
- Holdout candidate recall: 0.9793
- Change vs previous: v7 rebuild (D-030): blocking hs pass also ranks crowded number x word keys by name similarity (holdout recall 98.22 -> 98.31%); full retrain (stage 2 holdout 0.9882 -> 0.9886); stage 3 India/US --ctx + wide e5 holdout 0.99008, France no-ctx holdout 0.98977; merged per country. Build: scripts/run_x.sh
- Output stats: {"rule": "expected_f", "train_prior": 0.7340872444600691, "matches_per_entity_by_country": [["India", 809986, 3.337779912245397], ["US", 663106, 3.382646213425908], ["France", 259452, 3.4219547353653086]], "matching_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/matching_results.tsv", "candidate_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/candidate_pairs.tsv", "n_rows": 1732544, "n_candidate_pairs": 19405266, "n_match_pairs": 5834441, "n_empty_candidate_rows": 180, "n_empty_match_rows": 99468}
- Public leaderboard F0.5: _fill in after upload_

Upload: `matching_results.tsv` as the matching file and the code zip as the code file.

- Code zip rebuilt from `023cab4` (pipeline code identical to run commit 774d279; README S9 + runbook `run_x.sh` added to the package).
