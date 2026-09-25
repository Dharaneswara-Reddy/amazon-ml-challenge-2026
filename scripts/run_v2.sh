#!/usr/bin/env bash
# Pipeline v2 runbook (docs/DESIGN_V2.md §6). Phases so training can wait for approval:
#   scripts/run_v2.sh prep      learn maps → normalize v2 → ranker sample → ranker → block train/test → log-odds → features
#   scripts/run_v2.sh train     stage 1 + OOF + stage 2 + decision choice on tune; scores holdout (prints run dir)
#   scripts/run_v2.sh predict <run_dir>   test prediction with prior shift → official validator → freeze submission
#   scripts/run_v2.sh rest      resume after block_train (e.g. after the recall gate was reviewed and accepted)
#   scripts/run_v2.sh rest_features   resume at the feature step (candidates and log-odds already built)
#   scripts/run_v2.sh train_resume <run_dir>   finish an interrupted train (reuses its saved models) → predict
#   scripts/run_v2.sh all       prep → train → predict in one unattended chain (live view: scripts/progress.sh)
# Heavy steps take the shared lock so only one runs at a time; every step logs to artifacts/logs/v2_<step>.log.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
B=.venv/bin/ber
PY=.venv/bin/python
L=artifacts/logs
mkdir -p "$L"
step() { echo "=== $(date +%T) $*"; }
run() { local name=$1; shift; step "$name"; flock artifacts/heavy.lock "$B" "$@" --log-file "$L/v2_$name.log" > "$L/v2_$name.json"; }

case "${1:?usage: run_v2.sh prep|train|predict <run_dir>|all}" in
  prep)
    run learn_maps learn-maps
    run normalize_train normalize --split train --force
    run normalize_test normalize --split test --force
    run block_sample block --split train --sample-pct 5
    run train_ranker train-ranker
    run block_train block --split train
    R=$($PY -c "import json;print(json.load(open('$L/v2_block_train.json'))['recall']['recall_final'])")
    echo "blocking recall_final=$R"
    $PY -c "import sys; sys.exit(0 if $R >= 0.94 else 1)" || { echo "RECALL GATE FAILED (< 0.94, v1 level)"; exit 3; }
    "$0" prep_rest
    ;;
  prep_rest)
    run block_test block --split test
    run learn_logodds learn-logodds
    "$0" prep_features
    ;;
  prep_features)
    run features_train features --split train
    run features_test features --split test
    step "prep done"
    ;;
  train)
    run train train --name v2
    $PY -c "import json;print(json.load(open('$L/v2_train.json'))['run_dir'])"
    ;;
  train_resume)
    RUN=${2:?usage: run_v2.sh train_resume <run_dir>}
    run train train --name v2 --resume "$RUN"
    "$0" predict "$RUN"
    step "all done"
    ;;
  predict)
    RUN=${2:?usage: run_v2.sh predict <run_dir>}
    run predict predict --run "$RUN"
    scripts/freeze_submission.sh "$RUN" "v2: blocking v2 + features v2 + stage 2 + tune-chosen decision + prior shift"
    ;;
  rest|rest_features)
    if [ "$1" = rest ]; then "$0" prep_rest; else "$0" prep_features; fi
    "$0" train
    "$0" predict "$($PY -c "import json;print(json.load(open('$L/v2_train.json'))['run_dir'])")"
    step "all done"
    ;;
  all)
    "$0" prep
    "$0" train
    "$0" predict "$($PY -c "import json;print(json.load(open('$L/v2_train.json'))['run_dir'])")"
    step "all done"
    ;;
  *) echo "unknown phase $1" >&2; exit 2 ;;
esac
