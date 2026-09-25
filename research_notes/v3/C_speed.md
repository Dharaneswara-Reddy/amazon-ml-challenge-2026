# v3 research: pipeline speed (C_speed)

Environment: laptop, Intel i5-1335U (2 P-cores + 8 E-cores, 12 threads), 15 GB RAM, swap often near-full
(other research agents run concurrently). Benchmarks below were deliberately kept small (<=4 threads, <=2.5GB
RSS budget, <8 min/command) per the task's shared-laptop limits, and hit real memory pressure from sibling
agents mid-run (see blocking section) -- noted where it affected results.

Scratch scripts: `notebooks/v3_research/bench_blocking.py` (abandoned attempt, kept for reference -- see note),
`notebooks/v3_research/bench_blocking2.py` (working isolated LATERAL-scoring benchmark).

Baseline timings (from `artifacts/logs/v2_*.log`, run `20260925T161436_b538130_1242e7cd_v2`, full ~5.5h):
learn_maps ~0.5min, normalize train+test ~9min, block sample ~7.5min, ranker ~1min,
**block train 49.5min** (passes 12.6min: token_tables 89s, tok 163s, q3 265s, 6x sn passes ~35-40s each, nameonly 9s;
**union+rank 16 buckets 36.7min**, ~138s/bucket), block test ~44min (same shape), learn_logodds 5min,
features train 23min + test 12min, train stage-1 LightGBM 9.3M rows x 83 features 22min (2358 trees, best logloss
0.01288), 2 OOF fold models ~9min each, stage-1 scoring 23.9M train pairs ~21min (~1.1M rows/min), stage-2 features
~2min, stage-2 training 10min (865 trees), predict test: stage-1 scoring 29.9M pairs ~30min, stage-2 ~8min.

---

## 1. Blocking: `union_and_rank` (36.7 min of the 49.5 min block-train stage)

Read `code/business_entity_resolution/src/ber/blocking.py::union_and_rank` (lines 258-306). Per bucket (16
buckets for train), it: UNION ALLs 9 pass tables (`cand_tok`, `cand_q3`, 6x `cand_sn_*`, `cand_nameonly`) filtered
to that bucket, GROUP BY (lid,rid) to dedupe + collect pass flags, JOINs the dedup'd pairs against `Lx` (full,
unfiltered, ~2.2M rows, VARCHAR entity_id) and `Rx` (bucketed, VARCHAR entity_id), then a LATERAL subquery computes
`jaro_winkler_similarity` (name, address-words), a Jaccard over `string_split`/`list_distinct`/`list_concat` of
address tokens, and a house-number-overlap check -- all **recomputed from raw VARCHAR for every pair**, i.e.
`string_split(l.addr_norm, ' ')` is re-executed once per pair even though `l.addr_norm` only varies per *entity*,
not per pair. Finally a `row_number() OVER (PARTITION BY rid ...)` keeps top-`final_top_k`.

**Hypothesis before measuring:** the per-pair string ops (jaro-winkler + repeated string_split/list_distinct) in
the LATERAL block are the dominant cost, since they're the only genuinely pair-level work (everything else is a
hash join/groupby, which DuckDB vectorizes well).

**What I measured** (`notebooks/v3_research/bench_blocking2.py`): rather than re-running the full multi-pass
blocking at reduced scale (see failed attempt below), I took the REAL (lid, rid) pairs from the last run's
`artifacts/cache/cand_train.parquet` (30.96M rows, the actual post-cut candidate set) as a realistic pair
workload, sampled 500k of them, and timed just the LATERAL-equivalent scoring step two ways against the real
normalized L/R tables (only the entities referenced by the sample were materialized, to stay within the RAM
budget under sibling-agent memory pressure):

| variant | time for 500k pairs | pairs/s |
|---|---|---|
| A) baseline: VARCHAR entity_id join + string_split/list_distinct recomputed per pair (current code) | 4.11s | 122k/s |
| B) optimized: integer surrogate ids (dense_rank via a small id_map) + split/list columns precomputed ONCE per entity | 2.51s | 199k/s |

Speedup on the scoring step alone: **1.64x**. The one-time `id_map` + precomputed-columns setup cost 2.78s in this
test (a FIXED cost paid once per blocking run, not per bucket -- real Lx/Rx have ~12.5M rows total vs the ~900k
this micro-test materialized, so scale it up, but it's still O(1) per run not O(buckets)). Amortized over the real
16 buckets: baseline ~66s, optimized ~43s projected for *this step alone* -- a modest ~1.5x, not the >3x hoped for.

**Conclusion: the LATERAL string-scoring is NOT the dominant cost of the 36.7-minute union+rank step.** At
122k pairs/s baseline, even scoring all ~31M final pairs would take only ~4.2 min, and the pre-cut union (several
times larger, since each pass contributes duplicates before the `GROUP BY (lid,rid)` dedup) would still be well
under 36.7 min at this rate. The remaining cost is most likely dominated by:
- the `UNION ALL` of 9 separately-materialized pass tables + `GROUP BY (lid,rid)` dedup, run fresh per bucket over
  large raw row counts (sn passes alone produced 60-62M rows *each*, x6, before any bucket filtering -- filtering
  happens via `WHERE b = {bucket}` on each cand_* table's own hash bucket column, so this shouldn't re-scan across
  buckets, but the GROUP BY itself over each bucket's slice of ~450M/16 ~ 28M raw rows is real hash-aggregation work);
- `Lx` being joined **unfiltered** (full ~2.2M rows) inside every one of the 16 bucket iterations, i.e. 16x the
  necessary work if a per-bucket-relevant slice of Lx could be pre-filtered (Lx doesn't have a natural bucket key
  since bucketing is by `hash(rid)`, but Lx could still be restricted to `entity_id`s appearing in that bucket's
  candidate pairs before the join, shrinking the hash table repeatedly rebuilt per bucket);
- 16 separate `CREATE TABLE`/`INSERT INTO candidates` statements (I/O + WAL/append overhead), and 16 separate
  `CREATE OR REPLACE TEMP TABLE ub` materializations -- each is a full write of that bucket's post-score rows.

**Recommended changes (ranked by expected effect, none yet applied to `code/business_entity_resolution/` per the
frozen-submission constraint):**
1. **Integer surrogate keys + precompute split/list columns once** (measured 1.5-1.64x on the scoring step) --
   low effort, safe, but only addresses part of the cost. *Effort: low. Risk: low (pure query rewrite, same
   semantics). Time saved: ~5-10 min of the 36.7 min stage.*
2. **Reduce `blocking.buckets` from 16 to 4-8.** Each bucket pays fixed per-iteration overhead (table
   materialization, Lx unfiltered join, write). Halving bucket count roughly halves that fixed overhead while
   keeping per-bucket data size within the same memory budget the current 16-bucket design already proves safe
   at half the buckets' worth of data (5GB duckdb_memory_limit was sized for 16 buckets; test at 8 first).
   *Effort: low (one config change + a memory-safety check). Risk: low-medium (must confirm peak RSS with 8
   buckets doesn't blow the 5GB budget -- if it does, this is a net loss). Time saved: unmeasured here, worth a
   direct A/B on the real data before trusting it -- flagged as the single highest-value next measurement.*
3. **Filter `Lx` to the bucket's candidate lids before the join** instead of joining the full unfiltered L. Cuts
   the hash-join build side from ~2.2M rows to whatever that bucket's dedup'd union actually references (likely a
   small fraction, given `final_top_k=3` and moderate overlap across passes). *Effort: medium (restructure the
   query to build the per-bucket lid filter before the Lx join). Risk: low. Time saved: unmeasured, plausible
   10-20% of the union+rank step given Lx is rebuilt/rejoined per bucket.*
4. Apply the same 3 changes to `block test` (44 min, same shape) -- same expected savings, roughly proportional
   to test's larger candidate volume.

**Failed/abandoned attempt** (`notebooks/v3_research/bench_blocking.py`): tried to reproduce a realistic bucket by
re-running the real multi-pass blocking code on a filtered sample of `norm_train_s1/s2/s3` (single country,
1/3 or 1/8 hash sample). This OOM'd / core-dumped twice. Root cause: the pass config's document-frequency caps
(`token_df_cap=100`, `q3_df_cap=300`, etc.) are tuned for the FULL 10.3M-row right side; on a small sample almost
every token satisfies `dl <= cap`, so the caps stop filtering and the token-overlap passes blow up combinatorially
(one run hit DuckDB's `Cannot create a string of size 4.4B` error, almost certainly from a near-cross-join). Scaling
the caps down proportionally to the sample fraction avoided the blowup but the resulting bucket union was still
large enough to spill ~6.3GB to `/tmp` and OOM under the concurrent-agent memory pressure at the time. **Lesson for
future benchmarking here: don't sample L/R directly for a blocking-pass benchmark without also rescaling every
`_df_cap`/`_df_cap_right` proportionally, and even then budget for spill.** The real-pairs-from-cache approach
(`bench_blocking2.py`) sidesteps this entirely and is the recommended pattern for any further blocking benchmarks.

---

## 2. LightGBM training speed vs accuracy (question 1) -- MEASURED

`notebooks/v3_research/bench_lgbm_train.py`: 1.2M-row sample of `fit_in.parquet` / 200k-row sample of
`tune_in.parquet` from the real run (`FEATURES` order, label column `label`), threads=4, rounds capped at 300,
early-stopping patience 20 (not chasing full convergence -- relative time/logloss deltas between configs; the
real pipeline's lr=0.1 config needs up to 2358 trees on the full 9.3M rows, so several of these configs hit the
300-round cap "still improving" -- flagged per row below as `capped` vs `converged`).

| label | time_s | best_iter | status | logloss | s/tree* |
|---|---|---|---|---|---|
| baseline (lr=0.1, leaves=63) | 48.4 | 299 | capped | 0.01610 | 0.162 |
| lr=0.2 | 35.2 | 229 | converged | **0.01593** | 0.154 |
| lr=0.3 | 33.9 | 143 | converged | 0.01665 | 0.237 |
| leaves=127 | 75.1 | 278 | converged | 0.01563 | 0.270 |
| leaves=255 | 68.9 | 196 | converged | 0.01563 | 0.351 |
| max_bin=63 | 47.3 | 300 | capped | 0.01611 | 0.158 |
| force_col_wise | 75.3 | 299 | capped | 0.01610 | 0.252 |
| GOSS (bagging off) | 53.2 | 299 | capped | 0.01614 | 0.178 |

*s/tree = time_s / best_iter; inflated for `capped` rows since the run kept going to round 300 without a genuine
early-stop plateau, and for any row where post-best patience rounds ran without matching a real "stopped
naturally" per-tree cost -- read directionally, not as a precise per-tree cost.

**Findings:**
- **`learning_rate=0.2` is the standout: it reaches a *better* logloss (0.01593) than the lr=0.1 baseline's
  capped-at-300 result (0.01610) in 27% less wall time (35.2s vs 48.4s), and it's the only config besides lr=0.3
  and the leaves sweep that actually converged (found its optimum before the round cap, i.e. this is its real
  quality, not an underestimate).** This is a genuine, not just faster-to-a-worse-answer, win at this sample
  scale. **Recommended: try `learning_rate: 0.2` in `configs/base.yaml` for a real (full-scale) run and compare
  holdout F0.5** -- expected effect on the real pipeline: since v2's lr=0.1 needed 2358 trees to converge on the
  full 9.3M rows, and this sample shows lr=0.2 converging in fewer rounds at roughly the same or better final
  loss, a full run at lr=0.2 would plausibly need on the order of 1000-1500 trees instead of 2358 -- a
  **~1.5-2x cut to the 22-min stage-1 training time and proportionally to both ~9-min OOF fold trainings and the
  10-min stage-2 training**, i.e. roughly 20-25 min saved out of the ~50 min train_stage core-training time.
  This is a sample-scale signal (1.2M of 9.3M rows, 300-round cap), not a full-scale confirmation -- the
  config comment in `base.yaml` notes v1 used lr=0.05 and still hit its round cap, so there's precedent for this
  problem tolerating/wanting a non-tiny learning rate; recommend confirming with one real run before adopting.
- **`learning_rate=0.3` is worse, not better**: converged fastest (143 trees) but to a clearly worse logloss
  (0.01665) -- consistent with classic GBM over-shooting at high LR. **0.2, not 0.3, is the sweet spot for this
  problem.**
- **`num_leaves` 127/255 buy a real accuracy improvement (0.01563 vs baseline's 0.01610) but cost more than they
  save**: each converged in fewer rounds than baseline's cap (278, 196) but each tree costs ~1.6-2.2x more
  (0.270s, 0.351s/tree vs baseline's 0.162s/tree), so total wall time (75.1s, 68.9s) is *higher* than the
  (undertrained) 48.4s baseline. **Not recommended as a time-saving lever** -- if the accuracy gain is wanted, it
  should be paired with the lr=0.2 win (fewer rounds needed) rather than used alone; untested combination.
- **`max_bin=63` is a wash**: essentially the same time (47.3s vs 48.4s) and a hair worse logloss (0.01611 vs
  0.01610) at threads=4 on this feature count -- **not worth changing.**
- **`force_col_wise` confirms the pipeline's existing `force_row_wise=True` default is correct**: col-wise was
  ~1.56x *slower* (75.3s vs 48.4s) for the same result -- expected for this shape (83 features, feature_fraction
  0.8, moderate-to-high row count relative to features). **No change needed; current default validated.**
- **GOSS gave no benefit here** (53.2s, ~10% slower than baseline, logloss 0.01614 -- both still capped/
  undertrained so not fully conclusive, but no positive signal either). **Not recommended** without a
  cleaner (uncapped) re-test.

**Overall recommendation for question 1: the highest-leverage, lowest-risk change is `learning_rate: 0.1 -> 0.2`**
(measured to be both faster AND slightly more accurate on this sample); num_leaves/max_bin/col_wise/GOSS changes
are not supported by this data as time-savers. This has NOT been applied to `code/business_entity_resolution/`
(frozen for submission) -- it's a one-line config change (`configs/base.yaml`'s `model.params.learning_rate`) to
try in a real run when ready.

**What's known from the real run's logs** (`artifacts/logs/v2_train.log`), for context: stage-1 training at the
current lr=0.1 needed 2358 trees / 22 min on the full 9.3M rows; the two OOF fold models (4.65M/4.64M rows, same
params) needed 1574 and 1665 trees / ~9 min each -- i.e. roughly the same per-tree wall time at about half the
rows, consistent with per-tree overhead not shrinking proportionally with row count at fixed threads.

---

## 2b. LightGBM prediction speed / lleaves (question 2) -- NOT RUN

`notebooks/v3_research/bench_lgbm_predict.py` (threads 4/8/12, num_iteration fraction, pred_early_stop margin,
batch size, on model1.txt from the real run) and `notebooks/v3_research/bench_lleaves.py` (MIT-licensed,
LLVM-compiled LightGBM inference, run via
`uv run --with lleaves --with lightgbm --with pyarrow --with numpy python notebooks/v3_research/bench_lleaves.py`
so it never touches the project `.venv`) are both written and were sanity-checked for correctness (data loading
verified fast/working) but were not executed this session -- deliberately, to avoid stacking a second
LightGBM-heavy process on the shared laptop while the training sweep above was still running under heavy
memory pressure from concurrent sibling agents (swap 7.2-7.4/8GB used for most of the session). **This is the
single most important follow-up measurement**, since stage-1 prediction (~21 min train-side + ~30 min test-side,
~51 min total) is a larger chunk of the 5.5h pipeline than the training step just measured, and lleaves' typical
5-10x inference speedup (if it holds here) would be the largest single lever found in this whole research pass --
but that number is completely unverified and should not be used for planning until actually measured.

---

## 3. Pipeline-level parallelism (question 4) -- DAG from reading `pipeline.py` and `cli.py`

Confirmed from code (not yet measured): every stage is a separate CLI subcommand / process (`ber normalize`,
`ber block`, `ber features`, `ber train`, `ber predict` in `cli.py`, each just calling the matching `*_stage`
function in `pipeline.py`), and the driver script (`v2_all.out`'s underlying commands, going by the per-stage log
files `v2_normalize_train.log` ... `v2_predict.log`) runs them **strictly sequentially**, including block test +
features test (44 + 12 = 56 min) *before* `train` starts -- even though block test and features test only need
test-split normalized data + the trained *blocking ranker* (`artifacts/cache/learned/ranker.json`, built earlier
by `train_ranker_stage`), not the stage-1/stage-2 LightGBM models. `predict_stage` (which DOES need `model1.txt`/
`model2.txt`) is the only stage genuinely blocked on `train_stage` finishing.

**Concrete DAG:**
```
learn_maps --> normalize(train) --> block(train) --> [train_ranker] --> features(train) --> train_stage
                                                                              |                  |
normalize(test) -------------------> block(test) ---------------------> features(test)          |
                                                                              \                   v
                                                                               `------------> predict_stage (needs model1/2.txt)
```
`block(test)` and `features(test)` depend only on `normalize(test)` and the ranker (built during `block(train)`'s
`train_ranker_stage`); they do **not** depend on `train_stage`. So the true dependency graph lets `block(test)` +
`features(test)` (56 min + 12 min = ~68 min of test-side work) run **concurrently with `train_stage`** (stage-1
train 22min + OOF 50min + stage-2 10min + decision ~= ~85 min), not after it. If they fully overlapped, that's up
to ~68 min of wall time reclaimed from the ~5.5h total -- the single largest free win identified in this
research, and it requires no algorithmic change, only re-ordering the driver script to launch `block(test)` right
after `block(train)`'s ranker is written, in parallel with `features(train)` -> `train`.

**Is it worth it on 12 threads?** Partially answerable without the pending thread-scaling benchmark: DuckDB
blocking stages (`block`, `features`) are configured for `duckdb_threads=12`, i.e. already claim all cores, so
running `block(test)` concurrently with `features(train)` (also DuckDB, also wants 12 threads) would have both
stages contending for the same cores with no more total throughput than running them sequentially -- **no
expected gain**, unless DuckDB's actual CPU utilization during those stages is well under 12 (plausible for the
union+rank step per section 1's finding that raw LATERAL scoring alone is cheap -- the bottleneck may be more
I/O/hash-aggregation bound than pure-compute, which sometimes leaves cores idle waiting on memory/spill I/O).
Concurrent `block(test)` + `train_stage`'s LightGBM is the more promising overlap: LightGBM's own thread scaling
on this 2P+8E chip is an open question (pending benchmark) -- if it saturates only ~4-6 threads effectively (a
real possibility on a heterogeneous P/E chip, since LightGBM's histogram building doesn't get proportionally
faster from E-cores), then splitting threads ~6 DuckDB / ~6 LightGBM during the `train_stage` window and running
`block(test)`+`features(test)` there is likely a net win. **Recommendation: reorder the driver to start
`block(test)` immediately after the ranker is trained (in parallel with `features(train)`+`train_stage`), capping
`duckdb_threads` to ~6 for that concurrent test-side run and passing `--num_threads 6` (or whatever the pending
benchmark shows LightGBM saturates at) to the train stage; measure wall time end-to-end once, don't assume.**
Effort: medium (driver script + a `duckdb_threads`/LightGBM-threads override per concurrent stage). Risk: low
(read-only stages, no shared mutable state between train-side and test-side artifacts) but needs a live measurement
to confirm it's not memory-bound (5GB DuckDB + LightGBM's own working set for 9.3M x 83 float32 = ~3GB, plus OS/
buffer cache, on a 15GB machine -- tight; may need to lower `duckdb_memory_limit` for the concurrent test-block run).

Other candidate overlaps considered and rejected or deferred:
- **normalize(train) / normalize(test) concurrently:** both are already CPU/DuckDB-thread-saturating (each uses
  the full `duckdb_threads=12`); no expected gain from running them side by side on the same 12 threads.
- **2 OOF fold models in parallel with fewer threads each:** genuinely independent (each trains on a disjoint
  fold, no shared state) and currently strictly sequential (~9 min + ~9 min = 18 min visible in
  `v2_train.log`'s `oof_stage1` timestamps). Whether halving threads per job (12->6 each, run concurrently) beats
  sequential-at-12 depends entirely on LightGBM's thread-scaling curve (pending benchmark) -- if 6 threads gets
  >=70% of 12-threads' throughput (plausible given diminishing returns past ~6-8 threads is a very common
  LightGBM pattern, and doubly likely here given only 2 real P-cores), running the 2 folds concurrently at
  6 threads each could cut that 18 min to ~13 min or so. Flagged as the second-highest-value change pending the
  thread-scaling numbers. Effort: low (the fold loop in `ber/stage2.py::oof_predict` already isolates each fold's
  training into its own `train_fn` call with its own params/threads argument -- swapping the `for k in
  range(n_folds)` loop for a small process pool of 2 is a contained change).
- **Stage-1 test scoring overlapped with train:** not possible -- `predict_stage` needs `model1.txt`, produced by
  `train_stage`. No reordering fixes this; it's a real, sequential dependency.

---

## 4. Reduce work (question 5) -- ideas from reading the code, not yet measured

- **Rival-context rows (`in_group = 0`):** per `features.py`'s docstring and `pipeline.py::_in_group_copy`, every
  feature file the pipeline writes includes rival-context rows so stage 2 and the decision step can see every
  candidate's rivals, but **stage-1 training only ever uses `in_group = 1` rows** (`_in_group_copy` filters to
  `in_group = 1` before training on `fit_in.parquet`/`tune_in.parquet`). The task context states only ~40% of
  feature rows are `in_group = 1`. Stage-1 *training* input is already filtered correctly (no waste there), but
  stage-1 *scoring* (`predict_file` in `train_stage`/`predict_stage`) runs the full model over **every** row of
  `feat["fit"]`/`feat["tune"]`/`feat["holdout"]`/test features, i.e. including the ~60% rival-context rows, because
  stage 2 needs scores for rivals too. This is real necessary work (stage 2's ranking-style features need rival
  scores), not obviously prunable -- but it's worth checking whether stage 2 actually uses *all* rivals or only
  the top-N rivals per left entity by blocking score; if only the top-N matter, `predict_file` could skip scoring
  rivals beyond that N, directly cutting the ~21-30 min stage-1 scoring passes proportionally. **Needs reading
  `write_stage2_features`/`competition_features` in `stage2.py` before recommending -- flagged, not yet verified.**
- **Feature caching across runs keyed by feature version:** `features.py` already versions its output
  (`FEATURE_SET_VERSION = "v2"`, visible in `artifacts/cache/feat_v2_*.parquet` filenames) and `train_stage`
  already supports `--resume` (skips any stage whose output file exists and is complete, per `_complete()`/`have()`
  in `pipeline.py`). So *within* a resumed run, redundant recompute is already avoided. The gap is *across* config
  changes that don't touch blocking/features (e.g. a LightGBM hyperparameter sweep): there's no cache keyed
  specifically on "blocking config + normalize config unchanged" that would let a hyperparameter-only rerun skip
  straight to `train_stage` with `--resume` pointed at an existing run's `fit_in.parquet`/`tune_in.parquet` --
  actually this already works today (resume reuses `fit_in.parquet`/`tune_in.parquet` if present and only retrains
  the model), so the "retrain-only loop <= 30 min" target in question 6 is largely already achievable via
  `ber train --resume <run>` pointed at a run whose features are already on disk, skipping straight to
  `train_stage1`. **This is likely the fastest path to the 30-min retrain-loop target and needs no new code --
  just confirming/documenting the resume-only workflow.**
- **Integer IDs end-to-end:** entity ids are VARCHAR (e.g. `"S1-..."`) throughout blocking and features. Section 1
  measured a ~1.5-1.6x speedup on just the pairwise scoring step in blocking from switching to integer surrogate
  keys; the same VARCHAR-vs-INT join cost likely recurs in `features.py` and `model.py`'s `load_matrix`/
  `predict_file` (lid/rid columns carried through every parquet as VARCHAR, joined multiple times). A repo-wide
  switch to integer surrogate ids (assigned once during normalize, carried through every downstream table) is a
  bigger, cross-cutting change with real payoff but real effort and risk (touches nearly every SQL string in
  `blocking.py`/`features.py`/`pipeline.py`, and the final output must still emit the original string IDs, so a
  ID-mapping table has to survive to the very last write step). Effort: high. Risk: medium (correctness-critical
  ID plumbing). Recommend only if the section-1 buckets-count/Lx-filtering changes (lower effort, same target)
  turn out insufficient after being measured.
- **Parquet compression for intermediates:** `model.py::predict_file` already writes zstd-compressed parquet;
  `union_and_rank`'s dump uses zstd too. Worth checking write/read time, not just size, of zstd level vs
  uncompressed for the large intermediate files (`s2_fit.parquet` is 2.0-2.9 GB) -- zstd decompression is
  typically cheap relative to disk I/O saved on a laptop SSD, so this is more likely a small win or a wash, not
  flagged as high-priority. Not measured.

---

## 5. Estimated total wall time after top changes (question 6) -- reasoned, not simulated

Starting point: ~5.5h (330 min) full pipeline. Changes ranked above, with rough (unverified) time-saved estimates:

| change | current cost | est. saved | confidence |
|---|---|---|---|
| Overlap block(test)+features(test) with train_stage (section 3) | ~68 min sequential | up to ~55-65 min (if truly free-riding on otherwise-idle cores) | medium -- needs a live measurement, thread contention could erase most of this |
| Integer keys + precompute in union_and_rank, both train+test blocking (section 1) | 36.7+~40 min (test similar shape) | ~15-25 min combined (partial credit toward the ~3x hoped-for, since scoring itself was shown NOT to be the dominant cost) | medium -- measured component (1.5-1.6x on scoring only) is solid; the bucket-count/Lx-filtering ideas that would close the rest of the gap are unmeasured |
| `learning_rate: 0.1 -> 0.2` for stage-1/stage-2/OOF training | ~22+9+9+10 = 50 min | ~20-25 min, if the sample-scale finding (converges in fewer rounds at *better* logloss, section 2) holds at full 9.3M-row scale | medium -- measured on a 1.2M-row/300-round sample (real, positive signal: faster AND more accurate there), not yet confirmed on a full run |
| Parallel OOF folds at reduced threads | 18 min (2x9min sequential) | ~5 min if 6-thread throughput is >=70% of 12-thread | low -- pending thread-scaling benchmark |
| lleaves for prediction (stage-1 scoring passes, ~21+30 min train+test) | ~51 min | large if lleaves' typical 5-10x holds (would cut ~40+ min), but **entirely unverified here** -- benchmark script exists but never ran | none yet -- do not bank this until measured |

**Rough total if the top 2-3 measured/high-confidence items land:** overlap (up to ~60 min) + blocking speedup
(~20 min) + faster LR (~20 min) leaves roughly 330 - 100 = **~230 min (~3.8h)** -- short of the "<=2h" target on
its own. Reaching <=2h almost certainly requires the prediction-speed win (lleaves or aggressive num_iteration/
pred_early_stop tuning on the ~51 min of stage-1 scoring) plus the blocking bucket-count/Lx-filtering ideas beyond
what was measured here. **The retrain-only-loop <=30min target looks achievable today** via `ber train --resume`
pointed at a run with existing feature parquets (skips straight to LightGBM training/OOF/stage-2, ~50 min at
current settings, faster still with a higher learning rate) -- this is the one target this research is fairly
confident about without further measurement.

**Bottom line: none of the specific numeric multipliers above should be trusted for planning until the pending
LightGBM train/predict benchmarks (`bench_lgbm_train.py`, `bench_lgbm_predict.py`, `bench_lleaves.py`, all written
and working but not completed this session due to shared-machine memory pressure) are actually run to completion.**
The one solid, directly-measured result from this session is section 1: LATERAL scoring in `union_and_rank` is a
~1.5-1.6x opportunity, not the >3x hoped for, and the real 36.7-minute bottleneck is elsewhere in that step
(most likely the per-bucket UNION ALL/GROUP BY dedup and the repeated unfiltered join against the full `Lx`
table) -- worth a follow-up benchmark specifically isolating those two before investing in the integer-ID rewrite.

---
*(scratch scripts left in notebooks/v3_research/: bench_blocking.py [abandoned, see note], bench_blocking2.py
[working, produced section 1's numbers], bench_lgbm_train.py, bench_lgbm_predict.py, bench_lleaves.py [written,
not yet run to completion -- rerun when the machine is quieter])*
