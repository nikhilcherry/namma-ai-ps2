#!/usr/bin/env bash
# One command for the day: every model on the organiser's clip, benchmark table, report, safe copy.
#   bash run_all.sh <clip.mp4 | image folder> [gate|junction] [extra main.py args, e.g. --every 3 --zones zones.json]
# Outputs: out/<model>/..., out/benchmark.md, deliverables/ (the safe folder)
set -e
cd "$(dirname "$0")"
SRC="${1:?usage: bash run_all.sh <clip or folder> [gate|junction] [main.py args]}"; shift
STORY="gate"; case "${1:-}" in gate|junction) STORY="$1"; shift;; esac
PY="${PY:-python}"; [ -x .venv-pi/bin/python ] && PY=.venv-pi/bin/python
MAIN="${MAIN_MODEL:-yolo11n_raspberry_executorch_optimized}"      # the model the report and samples come from
OUT="${OUT:-out}"

FAILED=""
for pte in bundle/models/*.pte; do
  name="$(basename "$pte" .pte)"
  echo; echo "=== $name ==="
  if ! $PY main.py --source "$SRC" --model "$pte" --out "$OUT/$name" --blur "$@"; then
    echo "!! $name failed on this board, skipping"; FAILED="$FAILED $name"; rm -f "$OUT/$name/summary.json"
  fi
done
[ -f "$OUT/$MAIN/summary.json" ] || MAIN="yolo11n_raspberry_executorch_original"   # INT8 failed (Pi 4): use FP32

echo; $PY tools/bench_table.py "$OUT" --fill-readme
$PY tools/report.py "$OUT/$MAIN" --story "$STORY" ${SITE:+--site "$SITE"} || echo "!! report failed (not a deliverable)"
$PY stamp.py verify "$OUT/$MAIN"

D=deliverables; mkdir -p "$D"
cp "$OUT/benchmark.md" "$D/"
cp "$OUT/$MAIN/counts.csv" "$OUT/$MAIN/summary.txt" "$OUT/$MAIN/summary.json" "$OUT/$MAIN/stamp.json" "$D/"
cp "$OUT/$MAIN"/report.* "$OUT/$MAIN/heatmap.jpg" "$D/" 2>/dev/null || true
# two sample outputs: the frame with most people and the frame with most vehicles
$PY - "$OUT/$MAIN" "$D" <<'EOF'
import csv, shutil, sys
run, d = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(f"{run}/counts.csv")))
a = max(rows, key=lambda r: int(r["person"]))["frame"]
b = max((r for r in rows if r["frame"] != a), key=lambda r: int(r["vehicle"]), default=rows[0])["frame"]
for i, f in enumerate([a, b], 1):
    shutil.copy(f"{run}/{f}_det.jpg", f"{d}/sample_{i}_{f}.jpg")
print(f"samples: {a} (most people), {b} (most vehicles)")
EOF
echo
echo "DONE. Model used for outputs: $MAIN${FAILED:+   FAILED:$FAILED}"
echo "Safe folder: $D/  ->  now: git add -A && git commit -m 'Pi run' && git push"
echo "Next: python tools/handcount.py sheet --source \"$SRC\" --run $OUT/$MAIN"
