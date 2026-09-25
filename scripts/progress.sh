#!/usr/bin/env bash
# Live pipeline view: run `watch -n 5 scripts/progress.sh` from the repo root.
cd "$(git rev-parse --show-toplevel)"
L=artifacts/logs
CHAIN=$(ls -t $L/chain*.out 2>/dev/null | head -1)
echo "PIPELINE PROGRESS   $(date +%H:%M:%S)   (log: ${CHAIN:-none})"
echo "---------------------------------------------------------------"
[ -n "$CHAIN" ] && grep -E "^===|recall_final=|rule_metrics|FROZEN|run_dir=|^metrics|GATE|exit=" "$CHAIN"
echo; echo "Blocking (latest):"
grep -ho "blocking step=.*\|blocking union bucket=.*" $L/block_train.log $L/block_test.log 2>/dev/null | tail -4 | sed 's/^/  /'
echo; echo "Features (latest):"
grep -ho "stage=features.*status=[a-z]*\|features written.*" $L/features_*.log 2>/dev/null | tail -3 | cut -c1-140 | sed 's/^/  /'
echo; echo "Training (latest):"
grep -hE "\[[0-9]+\].*valid|Early stopping|Did not meet|train_done" $L/train.log 2>/dev/null | tail -4 | cut -c1-160 | sed 's/^/  /'
echo; echo "Errors:"
grep -hE "ERROR|Traceback|Error" "${CHAIN%.out}.err" 2>/dev/null | tail -2 | cut -c1-160 | sed 's/^/  /'
echo; echo "Machine:"
ps -eo pcpu,rss,etime,args --sort=-pcpu | grep "bin/ber " | grep -v grep | awk '{printf "  ber: %s%% CPU, %.1f GB RAM, running %s\n",$1,$2/1048576,$3}'
free -h | awk 'NR==2{print "  RAM used " $3 " / " $2 ", available " $7}'
