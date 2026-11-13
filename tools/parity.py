"""Parity check: runner.py against Arm's reference output for the same photo.

  python tools/parity.py [--model bundle/models/yolo11n_raspberry_executorch_original.pte]

Reference: bundle/models/sample_input.jpg + detections.json (Arm's runner, conf 0.001 style output, all 80 classes).
Only the classes this pipeline keeps are compared. FAIL (exit 1) when a reference box of a kept class with
score >= 0.5 has no match here (same class, IoU >= 0.9, score within --tol), or when we output a kept-class box
with score >= 0.5 that the reference does not have. INT8 vs FP32 differences are real: use --tol 0.05 for the INT8 file.
Status: written without ExecuTorch available, untested on device.
"""
import argparse, json, sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import runner


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "bundle/models/yolo11n_raspberry_executorch_original.pte"))
ap.add_argument("--tol", type=float, default=0.02, help="allowed score difference")
ap.add_argument("--min-score", type=float, default=0.5)
a = ap.parse_args()

img = cv2.imread(str(ROOT / "bundle/models/sample_input.jpg"))
ref = [d for d in json.loads((ROOT / "bundle/models/detections.json").read_text())
       if d["class"] in runner.NAMES.values() and d["score"] >= a.min_score]
det = runner.load(a.model)
ours, _ = det.detect(img, 0.30, iou=0.7)           # same NMS IoU as Arm's config.yaml, per-class
ours = [d for d in ours if d[1] >= a.min_score]
bad = 0
for r in ref:
    m = [o for o in ours if o[0] == r["class"] and iou(o[2], r["box"]) >= 0.9]
    if not m or abs(m[0][1] - r["score"]) > a.tol:
        bad += 1; print(f"MISSING/DIFFERENT: ref {r['class']} {r['score']:.3f} {[round(v) for v in r['box']]} -> {m[0][1] if m else None}")
for o in ours:
    if not any(o[0] == r["class"] and iou(o[2], r["box"]) >= 0.9 for r in ref):
        bad += 1; print(f"EXTRA: {o[0]} {o[1]:.3f} {[round(v) for v in o[2]]}")
print(f"{'PASS' if not bad else 'FAIL'}: {len(ref)} reference boxes of kept classes, {len(ours)} ours, {bad} mismatches ({Path(a.model).name})")
sys.exit(1 if bad else 0)
