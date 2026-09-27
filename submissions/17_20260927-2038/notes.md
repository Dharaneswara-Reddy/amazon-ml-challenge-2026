# Submission 17 (20260927-2038 IST)

- Git commit: `57b7151` (code zip: `code_57b7151.zip`)
- Run dir: `artifacts/runs/20260927T140904_774d279_e6b1b643_v5x`
- Local holdout macro F0.5: **0.9886** (tune nan, threshold None)
- Holdout candidate recall: 0.9793
- Change vs previous: 17a: #16 with the context stage 3 (stage3_x_iu) for France too; France-like holdout slices (shared names, crowded candidates) all favour it (+0.0002..+0.0006)
- Output stats: {"rule": "expected_f", "train_prior": 0.7340872444600691, "matches_per_entity_by_country": [["India", 809986, 3.337779912245397], ["US", 663106, 3.382646213425908], ["France", 259452, 3.4219547353653086]], "matching_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/matching_results.tsv", "candidate_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/candidate_pairs.tsv", "n_rows": 1732544, "n_candidate_pairs": 19405266, "n_match_pairs": 5834441, "n_empty_candidate_rows": 180, "n_empty_match_rows": 99468}
- Public leaderboard F0.5: _fill in after upload_

Upload: `matching_results.tsv` as the matching file and the code zip as the code file.
