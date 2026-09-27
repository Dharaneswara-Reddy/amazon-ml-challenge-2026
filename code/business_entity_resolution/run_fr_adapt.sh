#!/usr/bin/env bash
# France e5 adaptation (D-031) on top of the variant-x run: swap in the adapted French e5 scores (quantile-matched),
# stage 3 with competition context for all countries, predict, keep India/US from #17 and France from the adapted run.
# Usage: run_fr_adapt.sh [PREDICT_CONFIG]  (e.g. configs/variant_x_predict_frshift.yaml for the France shift)
# Needs: $R/colab_e5_fr/ce_test.parquet from ce/ce_colab_e5_adapt.ipynb. Result: artifacts/out_fr_adapt/*.tsv
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
B=.venv/bin/ber; PY=.venv/bin/python; L=artifacts/logs; C=code/business_entity_resolution
XP=${1:-$C/configs/variant_x_predict.yaml}; R=$HOME/Desktop/BER_GPU/results
RUN=$($PY -c "import json;print(json.load(open('$L/x_train.json'))['run_dir'])")
step() { echo "=== $(date +%T) $*"; }
step swap;    $PY $C/ce/swap_country_scores.py $R/colab+$R/colab_e5_ext $R/colab_e5_fr/ce_test.parquet $R/e5_fr_swapped France
step stage3;  $PY $C/ce/stage3.py $RUN artifacts/stage3_x_fradapt --ctx e5=$R/e5_fr_swapped bge=$R/kaggle_bge qwen=$R/kaggle_qwen > $L/fradapt_stage3.log 2>&1
step predict; $B --override $XP predict --run $RUN --stage3 artifacts/stage3_x_fradapt --log-file $L/fradapt_predict.log > $L/fradapt_predict.json
mkdir -p artifacts/out_fradapt_all && cp output/*.tsv artifacts/out_fradapt_all/
step merge;   $PY $C/ce/merge_by_country.py artifacts/out_x_iu artifacts/out_fradapt_all artifacts/out_fr_adapt France
cp artifacts/out_fr_adapt/*.tsv output/
step "ALL DONE"
