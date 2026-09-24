# Hackathon notes

Distilled from `problem_statement.pdf`, `guidelines.pdf`, and the AWS Builder Center prep guide (Sep 2026).

## Event
- Window: **25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST**. Dataset released day 1.
- **Max 5 leaderboard submissions per day** (15 total); the submit button is disabled after day 3.
- Keep a version history of every submission — shortlisting is based on submitted solutions, and final source code may be requested later.
- Shortlisting uses both public and private leaderboards → top 100 teams announced → they submit the methodology details.
- Portal: desktop/laptop only; **one login device per participant** (simultaneous logins can terminate the attempt).
- Multiple IDs / plagiarism / cheating → instant disqualification.
- Queries: the organizers' Google Form (link in `guidelines.pdf`); tech issues: support@unstop.com with a screenshot + registered email.
- 72-hour hackathon, teams of 2–4 (cross-college allowed), Indian engineering students graduating 2027/2028.
- Prizes: INR 1L / 75K / 50K for top 3; PPIs for top 50 teams.
- AWS credits: $200 per participant; extra $100 for the top 500 teams at the 48-hour mark.
- Public leaderboard = subset of test; final ranking = private leaderboard (rest of test). Don't overfit the public board.

## Task
Business Entity Resolution. Source 1 is the deduplicated reference. For each S1 entity, list all matching S2/S3 records
(zero, one, or many).

Columns in every source TSV: `entity_id` (S1-/S2-/S3- prefix), `business_name`, `business_address`, `country`.
Ground truth (`train_ground_truth.tsv`): `source1_entity_id`, `matched_entity_ids` (comma-separated, empty if none).

Noise to expect:
- Names: Corp/Corporation, Pvt/Private, Ltd/Limited, legal-suffix drift, DBA names, `&` vs "and", word reordering, typos, transliteration.
- Addresses: Rd/Road, St/Street, transliteration, missing PIN/state, landmark references ("Near SBI ATM"), municipal numbering, reordering.
- Train countries: US, India. Test adds France.

## Deliverables
1. Leaderboard: upload `matching_results.tsv`.
2. Final zip `<team_name>_submission.zip`:
   ```
   output/matching_results.tsv
   output/candidate_pairs.tsv
   code/business_entity_resolution/{src/, README.md, requirements.txt}
   Documentation_template.md   (filled: methodology, blocking, model + features, anything else)
   ```
   `utils/validate_submission.py` and `Documentation_template.md` come in the organizer's `student_resource/`.
3. For the best solution: a **1–2 page** document (ML approach, models, experiments, conclusion) and source code for
   experiments, training, and inference **with comments describing each function**.

## Metric
F0.5 = 1.25·P·R / (0.25·P + R), per S1 entity, macro-averaged. Singletons: empty prediction → 1.0, any match → 0.0.

## Organizer tips
Strong blocking sets the recall ceiling; string similarity (Jaccard, Levenshtein, TF-IDF cosine); country-specific address
patterns; tune for precision; don't neglect singletons; validate format before submitting.

## AWS (from the prep guide)
- Use a SageMaker **Notebook Instance** (not Studio — Studio domains can be slow to provision on new accounts),
  `ml.t3.medium`, `conda_python3` kernel. Train locally in the notebook.
- Free tier: Notebooks 250 h `ml.t3.medium`, Training 50 h `ml.m5.xlarge`, Inference 125 h `ml.m5.xlarge` (2 months); S3 5 GB.
- Region `us-east-1`. Set billing alerts first. Stop the notebook when done (files persist). Never leave endpoints running.
- Notebook IAM role needs read access to the bucket holding the dataset.
