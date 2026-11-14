#!/usr/bin/env bash
# Everything that has to be measured on the Pi, in order, in one command. Untested on device (written on a laptop).
#   bash tools/pi_day.sh <clip.mp4 | image folder> [gate|junction] [extra main.py args, e.g. --every 3 --zones zones.json]
#
# Steps (a failing step is recorded and the rest still run):
#   1 pi_check        throttle / temperature / governor / load before anything is timed
#   2 parity          runner.py vs Arm's reference output, FP32 and INT8
#   3 keygen          signing key for the stamp (kept off git), only if none exists
#   4 run_all         every approved model on your clip: counts, benchmark table, report, deliverables/
#   5 thread_sweep    1-4 cores for the INT8 and the FP32 640 model
#   6 verify          stamp signature of the run that fed the report
#   7 handcount       10 clean frames for a human to count (the accuracy number still needs that human)
# Output: out/pi_day/<step>.log and out/pi_day/SUMMARY.md (paste its table into the pitch notes).
cd "$(dirname "$0")/.."
SRC="${1:?usage: bash tools/pi_day.sh <clip or folder> [gate|junction] [main.py args]}"; shift
STORY="gate"; case "${1:-}" in gate|junction) STORY="$1"; shift;; esac
PY="${PY:-python}"; [ -x .venv-pi/bin/python ] && PY=.venv-pi/bin/python; export PY
MAIN="${MAIN_MODEL:-profile_accurate}"
D=out/pi_day; mkdir -p "$D"
INT8=bundle/models/yolo11n_raspberry_executorch_optimized.pte
FP32=bundle/models/yolo11n_raspberry_executorch_original.pte
: > "$D/results.tsv"

step() {   # step <name> <command...>
  local name="$1"; shift
  echo; echo "=== [$name] $*"
  if "$@" > "$D/$name.log" 2>&1; then
    echo "ok    $name"; printf '%s\tok\n' "$name" >> "$D/results.tsv"
  else
    echo "FAIL  $name   (see $D/$name.log, last lines:)"; tail -n 5 "$D/$name.log"; printf '%s\tFAIL\n' "$name" >> "$D/results.tsv"
  fi
}

step pi_check     bash tools/pi_check.sh
step parity_fp32  $PY tools/parity.py --model "$FP32"
step parity_int8  $PY tools/parity.py --model "$INT8"
[ -f .stamp_key ] || [ -n "${STAMP_KEY:-}" ] || step keygen $PY stamp.py keygen
step run_all      bash run_all.sh "$SRC" "$STORY" "$@"
step thread_int8  env OUT="$D/threads_int8" bash tools/thread_sweep.sh "$SRC" "$INT8" "$@"
step thread_fp32  env OUT="$D/threads_fp32" bash tools/thread_sweep.sh "$SRC" "$FP32" "$@"
step verify       $PY stamp.py verify "out/$MAIN"
step handcount    $PY tools/handcount.py sheet --source "$SRC" --run "out/$MAIN"

{
  echo "# Pi day summary ($(date '+%F %T'))"; echo
  BOARD="$(uname -m)"; [ -r /proc/device-tree/model ] && BOARD="$(tr -d '\0' < /proc/device-tree/model)"
  echo "Board: $BOARD   Python: $($PY --version 2>&1)"; echo
  echo "| step | result |"; echo "|---|---|"
  while IFS=$'\t' read -r n r; do echo "| $n | $r |"; done < "$D/results.tsv"
  [ -f out/benchmark.md ] && { echo; echo "## Benchmark (out/benchmark.md)"; cat out/benchmark.md; }
  for t in int8 fp32; do [ -f "$D/threads_$t/threads.md" ] && { echo; echo "## Threads, $t (Arm reference, 1 thread, Pi 5: INT8 134 ms, FP32 169 ms)"; cat "$D/threads_$t/threads.md"; }; done
  echo; echo "Still needs a human: fill handcount/handcount.csv from the clean frames, then"
  echo "  python tools/handcount.py table --run out/$MAIN"
} > "$D/SUMMARY.md"
echo; echo "Summary: $D/SUMMARY.md"; cat "$D/results.tsv" | tr '\t' ' '
