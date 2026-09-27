# Submission 14 (20260927-1713 IST)

- Git commit: `f72eb1b` (code zip: `code_f72eb1b.zip`)
- Run dir: `artifacts/runs/20260926T114709_10acc83_99304c0b_v5`
- Local holdout macro F0.5: **0.9882** (tune nan, threshold None)
- Holdout candidate recall: 0.9783
- Change vs previous: v6-final: India/US from stage 3 with competition context (--ctx) + e5 wider band (holdout 0.98971); France from the #12-method stage 3 (holdout 0.98934 same model family); merged per country (D-029)
- Output stats: {"rule": "expected_f", "train_prior": 0.7268714919564443, "matches_per_entity_by_country": [["India", 809986, 3.329964962357374], ["US", 663106, 3.3811125219798948], ["France", 259452, 3.420902517614048]], "matching_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/matching_results.tsv", "candidate_path": "/home/toji339/Documents/Amazon_ML_Hackathon/output/candidate_pairs.tsv", "n_rows": 1732544, "n_candidate_pairs": 18924178, "n_match_pairs": 5826821, "n_empty_candidate_rows": 205, "n_empty_match_rows": 99733}
- Public leaderboard F0.5: _fill in after upload_

Upload: `matching_results.tsv` as the matching file and the code zip as the code file.

## WARNING: src/ differs between the run commit 10acc833678a05756feec6f56577d3af04f9b583 and HEAD

 code/business_entity_resolution/src/ber/cli.py     | 11 +++++-----
 .../business_entity_resolution/src/ber/pipeline.py | 24 ++++++++++++++++------
 2 files changed, 24 insertions(+), 11 deletions(-)

## Final build (D-029)
- India/US rows: `artifacts/stage3_final_iu` (stage 3 --ctx, e5 first + extended band, bge, qwen) — holdout **0.98971**
- France rows: `artifacts/stage3_final_fr` (stage 3 without ctx, first-band e5/bge/qwen = #12 method) — holdout 0.98934
- Merged with `ce/merge_by_country.py artifacts/out_final_iu artifacts/out_final_fr output France`; exact commands in README S8.
- vs #12 (public 0.985004): India +5073/-1215 pairs, US +2957/-1194, France +1602/-1379 (retrain noise).
