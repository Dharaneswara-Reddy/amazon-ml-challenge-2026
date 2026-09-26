#!/usr/bin/env bash
# Live view of the v2 pipeline (scripts/run_v2.sh). From the repo root:  watch -n 5 scripts/progress.sh
cd "$(git rev-parse --show-toplevel)"
L=artifacts/logs
V=v2; [ -f $L/v3_all.out ] && V=v3; [ -f $L/v4_all.out ] && V=v4
OUT=$L/${V}_all.out
PY=.venv/bin/python
STEPS=(learn_maps normalize_train normalize_test block_sample train_ranker block_train block_test learn_logodds
       features_train features_test train predict)
[ $V = v3 ] && STEPS=(block3_train block3_test learn_logodds features_train features_test train predict)
[ $V = v4 ] && STEPS=(learn_selfvocab features_train features_test train predict)
now=$(date +%s)
alive=$(pgrep -f "^[^ ]*bash scripts/run_v[0-9s]*.sh" >/dev/null && echo RUNNING || echo "NOT RUNNING")
bar() { local d=$1 t=$2 w=30 f s; f=$(( t > 0 ? d * w / t : 0 )); printf -v s '%*s' "$f" ''; printf '[%s%*s] %s/%s' "${s// /#}" $((w - f)) '' "$d" "$t"; }
hms() { printf '%dh%02dm' $(($1 / 3600)) $(($1 % 3600 / 60)); }
start_of() { grep -E "^=== [0-9:]+ $1\$" "$OUT" 2>/dev/null | tail -1 | awk '{print $2}'; }
secs() { date -d "$1" +%s 2>/dev/null; }

echo "PIPELINE ${V^^}   $(date +%H:%M:%S)   chain: $alive   (full log: $OUT)"
echo "────────────────────────────────────────────────────────────────────────────"
cur=""
for s in "${STEPS[@]}"; do
  t=$(start_of "$s")
  if [ -z "$t" ]; then printf '  [ ]  %-16s\n' "$s"; continue; fi
  cur=$s
  if [ -s "$L/${V}_$s.json" ]; then printf '  [x]  %-16s started %s  done\n' "$s" "$t"
  elif [ "$alive" = RUNNING ]; then printf '  [>]  %-16s started %s, %s elapsed\n' "$s" "$t" "$(hms $(( now - $(secs "$t") )))"
  else printf '  [!]  %-16s started %s, FAILED or stopped (see errors)\n' "$s" "$t"
  fi
done

LOG=$L/${V}_$cur.log
if [ -n "$cur" ] && [ -f "$LOG" ]; then
  echo; echo "Current step: $cur"
  case $cur in
    normalize_*) l=$(grep -o "normalize rows=[0-9]*/[0-9]*" "$LOG" | tail -1 | sed 's/normalize rows=//')
                 [ -n "$l" ] && echo "  rows  $(bar "${l%/*}" "${l#*/}")"
                 grep -o "stage_start stage=normalize.*source=[0-9]" "$LOG" | tail -1 | sed 's/^/  /' ;;
    block3_*)    grep -oE "block3 pass=.*|block3 union bucket=[0-9/]+ rows=[0-9]+|block3 cut scored bucket=[0-9/]+" "$LOG" | tail -8 | cut -c1-110 | sed 's/^/  /' ;;
    block_*)     grep -o "blocking step=.*" "$LOG" | tail -12 | sed 's/^/  pass done: /'
                 l=$(grep -o "blocking union bucket=[0-9]*/[0-9]*" "$LOG" | tail -1 | sed 's/.*bucket=//')
                 [ -n "$l" ] && echo "  union+rank buckets $(bar "${l%/*}" "${l#*/}")" ;;
    features_*)  grep -o "stage_start stage=features.*\|stage_end stage=features[^ ]* [^ ]* [^ ]* seconds=[^ ]*\|features written.*rows=[0-9]*" "$LOG" | tail -4 | cut -c1-110 | sed 's/^/  /'
                 l=$(grep -o "features bucket=[0-9]*/[0-9]*" "$LOG" | tail -1 | sed 's/.*bucket=//')
                 [ -n "$l" ] && echo "  buckets (current group) $(bar "${l%/*}" "${l#*/}")" ;;
    train)       RUN=$(ls -td artifacts/runs/*_${V} 2>/dev/null | head -1)
                 last=$(stat -c %Y "$(ls -t "$RUN"/* 2>/dev/null | head -1)" 2>/dev/null || echo "$now")
                 pids=$(pgrep -d, -f "bin/ber |multiprocessing.spawn"); cpu=$( [ -n "$pids" ] && top -b -n1 -p "$pids" | awk 'NR>7 {c+=$9} END {printf "%.0f", c}' )
                 ms() { if [ -e "$RUN/$1" ]; then printf '    [x] %s\n' "$2"; elif [ "$3" = cur ]; then printf '    [>] %s   <- running %dm%02ds, CPU %s%%\n' "$2" $(( (now - last) / 60 )) $(( (now - last) % 60 )) "$cpu"; else printf '    [ ] %s\n' "$2"; fi; }
                 nxt() { for f in "$@"; do [ -e "$RUN/$f" ] || { echo "$f"; return; }; done; }
                 n=$(nxt fit_in.parquet model1.txt pred1_fit_in.parquet pred1_fit.parquet pred1_holdout.parquet s2_holdout.parquet model2.txt pred2_holdout.parquet decision_params.json manifest.json)
                 st() { [ "$n" = "$1" ] && echo cur; }
                 parts=$(ls "$RUN"/protocol_out_*.parquet 2>/dev/null | wc -l); pin=$(ls "$RUN"/protocol_in_*.parquet 2>/dev/null | wc -l)
                 echo "  run: $RUN"
                 ms fit_in.parquet      "copy training rows" "$(st fit_in.parquet)"
                 ms model1.txt          "stage-1 model (model 1/4)" "$(st model1.txt)"
                 ms pred1_fit_in.parquet "fold A + fold B models (models 2-3/4) and their out-of-fold predictions" "$(st pred1_fit_in.parquet)"
                 ms pred1_fit.parquet   "score all 23.9M train pairs with stage 1  ($( if [ "$pin" -gt 0 ]; then echo "part $((parts > 0 ? parts : 1))/3 running"; elif [ "$parts" -gt 0 ]; then echo "3/3 scored, merging"; fi ))" "$(st pred1_fit.parquet)"
                 ms pred1_holdout.parquet "score tune + holdout with stage 1" "$(st pred1_holdout.parquet)"
                 ms s2_holdout.parquet  "build stage-2 features (fit, tune, holdout)" "$(st s2_holdout.parquet)"
                 ms model2.txt          "stage-2 model (model 4/4)" "$(st model2.txt)"
                 ms pred2_holdout.parquet "score tune + holdout with stage 2" "$(st pred2_holdout.parquet)"
                 ms decision_params.json "choose decision rule on tune, score holdout (local F0.5)" "$(st decision_params.json)"
                 ms manifest.json       "save run manifest" "$(st manifest.json)"
                 echo "  latest log lines:"
                 grep -oE "\[[0-9]+\].*|trained lightgbm best_iteration=[0-9]+|oof fold=.*|stage2 features written.*rows=[0-9]+|train_done.*" "$LOG" | tail -3 | cut -c1-120 | sed 's/^/    /' ;;
    predict)     RUN=$(ls -td artifacts/runs/*_${V} 2>/dev/null | head -1)
                 ms=("pred1_test.parquet:score test with stage 1" "s2_test.parquet:build stage-2 test features" "pred2_test.parquet:score test with stage 2" "predict.json:prior shift + decision + write output TSVs")
                 for i in 0 1 2 3; do
                   f=${ms[$i]%%:*}; label=${ms[$i]#*:}; nf=${ms[$((i + 1))]%%:*}
                   if [ -e "$RUN/$f" ] && { [ $i -eq 3 ] || [ -e "$RUN/$nf" ]; }; then echo "    [x] $label"
                   elif [ -e "$RUN/$f" ] || [ $i -eq 0 ]; then
                     pct=""
                     if [ $i -eq 0 ]; then
                       pp=$(pgrep -f "python .venv/bin/ber predict" | head -1)
                       rb=$(awk '/^read_bytes/ {print $2}' /proc/$pp/io 2>/dev/null)
                       tot=$(stat -c %s artifacts/cache/feat_v2_test.parquet 2>/dev/null)
                       [ -n "$rb" ] && [ -n "$tot" ] && pct=$(( rb * 100 / tot )) && [ $pct -gt 100 ] && pct=100
                     elif [ $i -eq 2 ] && [ -e "$RUN/pred2_test.parquet" ]; then
                       # same rows and columns as the finished stage-1 file, so the size ratio tracks progress
                       pct=$(( $(stat -c %s "$RUN/pred2_test.parquet") * 100 / $(stat -c %s "$RUN/pred1_test.parquet") ))
                       [ $pct -gt 99 ] && pct=99
                     fi
                     if [ -n "$pct" ]; then echo "    [>] $label   $(bar "$pct" 100 | sed 's| [0-9]*/100$||') ${pct}%"
                     else echo "    [>] $label"; fi
                   else echo "    [ ] $label"; fi
                 done
                 grep -oE "prior_shift.*|write_outputs.*" "$LOG" | tail -2 | cut -c1-140 | sed 's/^/    /' ;;
    *)           tail -3 "$LOG" | cut -c25-160 | sed 's/^/  /' ;;
  esac
fi

echo; echo "Results so far:"
$PY - <<'PYEOF' 2>/dev/null
import json, pathlib
L = pathlib.Path("artifacts/logs")
V = "v4" if (L / "v4_all.out").exists() else ("v3" if (L / "v3_all.out").exists() else "v2")
def j(name):
    p = L / f"{V}_{name}.json"
    try:
        return json.loads(p.read_text()) if p.stat().st_size else None
    except Exception:
        return None
for split in (("block3_train", "block3_test") if V == "v3" else ()):
    b = j(split)
    if b:
        r = b.get("recall") or {}
        line = f"  {split}: pairs={b.get('final_pairs', 0):,}"
        line += f"  recall={r['recall']:.4f}  pairs/S1={r['pairs_per_s1']:.1f}" if r else f"  pairs/S1={b.get('pairs_per_s1', 0):.1f}"
        line += "".join(f"  {k}={b[k]:.4f}" for k in ("recall_fit", "recall_tune", "recall_holdout") if k in b)
        print(line)
for split in (() if V == "v3" else ("sample", "train", "test")):
    b = j(f"block_{split}")
    if b:
        r = b.get("recall", {})
        extra = f"  recall={r['recall_final']:.4f}" if "recall_final" in r else ""
        m = b.get("cand_per_left_mean_p50_p99_max", [0])
        print(f"  block {split:6s} pairs={b.get('final_pairs', 0):,}  cand/S1={m[0]:.1f}{extra}")
rk = pathlib.Path("artifacts/cache/learned/ranker.json")
if rk.exists():
    d = json.loads(rk.read_text()); print(f"  ranker K={d.get('k')}  weights={ {k: round(v, 2) for k, v in d.get('weights', {}).items()} }")
t = j("train")
if t:
    mf = pathlib.Path(t["run_dir"]) / "manifest.json"
    m = json.loads(mf.read_text()).get("metrics", {}) if mf.exists() else {}
    print(f"  run {t['run_dir']}")
    for k in ("rule", "holdout_f05", "stage1_holdout", "stage2_holdout", "holdout_candidate_recall", "best_iteration_stage1", "best_iteration_stage2"):
        if k in m: print(f"    {k}: {m[k]}")
p = j("predict")
if p:
    print(f"  predict: rule={p.get('rule')} test_prior_em={p.get('test_prior_em')} logit_delta={p.get('logit_delta')}")
    print(f"    matches/entity by country: {p.get('matches_per_entity_by_country')}")
PYEOF
grep -E "FROZEN|submissions/[0-9]|PASS|FAIL" "$OUT" 2>/dev/null | tail -3 | sed 's/^/  /'

echo; echo "Errors:"
grep -hE "Traceback|Error|ERROR|Killed|status=error" "$OUT" "$LOG" 2>/dev/null | tail -3 | cut -c1-160 | sed 's/^/  /'

echo; echo "Machine:"
pids=$(pgrep -d, -f "bin/ber |multiprocessing.spawn")
if [ -n "$pids" ]; then
  c=$(top -b -n1 -p "$pids" | awk 'NR>7 {c+=$9} END {printf "%.0f", c}')
  ps -o rss= -p "$pids" | awk -v c="$c" '{r+=$1; n++} END {printf "  pipeline processes: %d, CPU now %s%% (1200%% = all 12 threads), RAM %.1f GB\n", n, c, r/1048576}'
fi
free -h | awk 'NR==2{print "  RAM used " $3 " / " $2 ", available " $7} NR==3{print "  swap used " $3 " / " $2}'
echo "  spill dir: $(du -sh artifacts/tmp 2>/dev/null | cut -f1)   disk free: $(df -h . | awk 'NR==2{print $4}')"
