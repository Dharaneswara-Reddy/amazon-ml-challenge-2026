#!/usr/bin/env bash
# Rebuild with blocking variant x (D-030): block3 test -> features -> train -> predict -> stage 3 (India/US ctx + wide e5,
# France no-ctx) -> per-country merge into artifacts/out_x_final, copied to output/.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
B=.venv/bin/ber; PY=.venv/bin/python; L=artifacts/logs; C=code/business_entity_resolution
X=$C/configs/variant_x.yaml; XP=$C/configs/variant_x_predict.yaml
R=$HOME/Desktop/BER_GPU/results
step() { echo "=== $(date +%T) $*"; }
step block3 test;     $B --override $X block3 --split test --log-file $L/x_block3_test.log > $L/x_block3_test.json
step features train;  $B --override $X features --split train --log-file $L/x_features_train.log > $L/x_features_train.json
step features test;   $B --override $X features --split test --log-file $L/x_features_test.log > $L/x_features_test.json
step train;           $B --override $X train --name v5x --log-file $L/x_train.log > $L/x_train.json
RUN=$($PY -c "import json;print(json.load(open('$L/x_train.json'))['run_dir'])")
step predict $RUN;    $B --override $XP predict --run $RUN --log-file $L/x_predict.log > $L/x_predict.json
step stage3 iu;       $PY $C/ce/stage3.py $RUN artifacts/stage3_x_iu --ctx e5=$R/colab+$R/colab_e5_ext bge=$R/kaggle_bge qwen=$R/kaggle_qwen > $L/x_stage3_iu.log 2>&1
$B --override $XP predict --run $RUN --stage3 artifacts/stage3_x_iu --log-file $L/x_predict_iu.log > $L/x_predict_iu.json
mkdir -p artifacts/out_x_iu && cp output/*.tsv artifacts/out_x_iu/
step stage3 fr;       $PY $C/ce/stage3.py $RUN artifacts/stage3_x_fr e5=$R/colab bge=$R/kaggle_bge qwen=$R/kaggle_qwen > $L/x_stage3_fr.log 2>&1
$B --override $XP predict --run $RUN --stage3 artifacts/stage3_x_fr --log-file $L/x_predict_fr.log > $L/x_predict_fr.json
mkdir -p artifacts/out_x_fr && cp output/*.tsv artifacts/out_x_fr/
step merge;           $PY $C/ce/merge_by_country.py artifacts/out_x_iu artifacts/out_x_fr artifacts/out_x_final France
cp submissions/14_20260927-1713/*.tsv output/
step "ALL DONE"
