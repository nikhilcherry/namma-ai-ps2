#!/usr/bin/env bash
# Thread-count sweep on the Pi: same model, same frames, 1..4 cores. Untested on device.
#   bash tools/thread_sweep.sh <clip|folder> [model.pte] [extra main.py args]
# Uses taskset (always available) and also passes --threads N (works only if this executorch build exposes it;
# main.py prints which). Compare mean/p95 and FPS per row; pick the knee, not the maximum.
set -e
cd "$(dirname "$0")/.."
SRC="${1:?usage: bash tools/thread_sweep.sh <clip or folder> [model.pte]}"; shift
MODEL="${1:-bundle/models/yolo11n_raspberry_executorch_optimized.pte}"; [ $# -gt 0 ] && shift
PY="${PY:-python}"; [ -x .venv-pi/bin/python ] && PY=.venv-pi/bin/python
OUT="${OUT:-out_threads}"; mkdir -p "$OUT"
printf '| cores | threads flag | mean ms | median ms | p95 ms | FPS (infer) | throttled |\n|---|---|---|---|---|---|---|\n' | tee "$OUT/threads.md"
BAD=0
for n in 1 2 3 4; do
  cores="0-$((n-1))"; [ "$n" = 1 ] && cores=0
  PIN=""; command -v taskset >/dev/null && PIN="taskset -c $cores"      # no taskset (not Linux): --threads alone
  if ! $PIN $PY main.py --source "$SRC" --model "$MODEL" --out "$OUT/t$n" --deploy --threads "$n" "$@" > "$OUT/t$n.log" 2>&1; then
    echo "| $n | failed, see $OUT/t$n.log | | | | | |" | tee -a "$OUT/threads.md"; BAD=1; continue
  fi
  $PY tools/thread_row.py "$OUT/t$n/summary.json" "$n" | tee -a "$OUT/threads.md"
done
echo "table: $OUT/threads.md  (Arm's reference, 1 thread, Pi 5: INT8 134 ms, FP32 169 ms)"
exit $BAD
