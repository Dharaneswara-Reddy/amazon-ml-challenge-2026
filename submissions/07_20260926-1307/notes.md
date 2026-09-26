# Submission 07 (20260926-1307 IST)

- Git commit: `c2042b9` (code zip: `code_c2042b9.zip`)
- Run dir: `/home/toji339/Documents/Amazon_ML_Hackathon/artifacts/runs/20260926T070539_1f9ad5d_99304c0b_v4s`
- Local holdout macro F0.5: **0.9877** (tune nan, threshold None)
- Holdout candidate recall: 0.9793
- Change vs previous: v4s: v4 + France pseudo-label self-training (stage-1 trained with rule/model-labelled France test pairs)
- Output stats: {"rule": "expected_f", "train_prior": 0.726776615193123, "matches_per_entity_by_country": [["India", 809986, 3.322312978249007], ["US", 663106, 3.3711352332809534], ["France", 259452, 3.450765459506961]], "matching_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/matching_results.tsv", "candidate_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/candidate_pairs.tsv", "n_rows": 1732544, "n_candidate_pairs": 18924178, "n_match_pairs": 5821755, "n_empty_candidate_rows": 205, "n_empty_match_rows": 100338}
- Public leaderboard F0.5: _fill in after upload_

Upload: `matching_results.tsv` as the matching file and the code zip as the code file.
