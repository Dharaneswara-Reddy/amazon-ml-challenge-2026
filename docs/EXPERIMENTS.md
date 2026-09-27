# Experiments

| Run | Rule | Holdout F0.5 | Stage-1 holdout (thr / expF) | Candidate recall | Notes |
|---|---|---|---|---|---|
| `20260925T161436_b538130_1242e7cd_v2` | expected_f | 0.9715 | 0.9698 / 0.9700 | 0.9420 | |
| `20260926T032546_e2e7494_678411c5_v3` | threshold | 0.9876 | 0.9863 / 0.9864 | 0.9793 | |
| `20260926T063247_1a0d20f_99304c0b_v4` | expected_f | 0.9879 | 0.9865 / 0.9866 | 0.9794 | |
| `20260926T070539_1f9ad5d_99304c0b_v4s` | expected_f | 0.9877 | 0.9863 / 0.9864 | 0.9793 | |
| `20260926T114709_10acc83_99304c0b_v5` | expected_f | 0.9882 | 0.9862 / 0.9863 | 0.9783 | |
| `stage3_all3ctx` (v5 + e5/bge/qwen stage 3 + per-S1 rank/gap context of each score; not adopted, +0.00004 = noise) | threshold | 0.98942 | | | |
