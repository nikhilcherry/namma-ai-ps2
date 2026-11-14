"""Parity check: runner.py against Arm's reference output for the same photo.

  python tools/parity.py [--model bundle/models/yolo11n_raspberry_executorch_original.pte]

Reference: bundle/models/sample_input.jpg + detections.json (Arm's runner, all 80 classes). The sample is an
indoor scene (couch, chair, tv, plants): it holds no person or vehicle, so comparing only the classes the pipeline
keeps would test nothing. The check therefore widens runner's class tables to all 80 COCO classes, which exercises
the same letterbox, output layout, box mapping and NMS. FAIL (exit 1) when a reference box with score >= 0.5 has no
match here (same class, IoU >= 0.9, score within --tol), when we output a box with score >= 0.5 the reference lacks,
or when there is nothing to compare. Measured 14 Nov on a Windows laptop (ExecuTorch 1.1.0, torch 2.10.0): FP32 max score difference 0.015 (all 5
boxes, IoU >= 0.9), INT8 max 0.056, hence the defaults. The reference is FP32 output, so INT8 differs by quantisation.
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
ap.add_argument("--tol", type=float, default=None,
                help="allowed score difference (default 0.02 for FP32, 0.06 for a file with INT8/optimized in its name)")
ap.add_argument("--min-score", type=float, default=0.5)
a = ap.parse_args()
if a.tol is None:
    a.tol = 0.06 if ("optimized" in a.model or "int8" in a.model.lower()) else 0.02

img = cv2.imread(str(ROOT / "bundle/models/sample_input.jpg"))
COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter "
        "bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
        "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass "
        "cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
        "potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink "
        "refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").split()
runner.IDS = list(range(80))
runner.NAMES = {i: n.replace("_", " ") for i, n in enumerate(COCO)}
ref = [d for d in json.loads((ROOT / "bundle/models/detections.json").read_text()) if d["score"] >= a.min_score]
if not ref:
    sys.exit("FAIL: no reference boxes above --min-score, nothing to compare")
det = runner.load(a.model)
ours, _ = det.detect(img, 0.30, iou=0.7)           # same NMS IoU as Arm's config.yaml, per-class (ungrouped)
ours = [d for d in ours if d[1] >= a.min_score]
bad = 0
for r in ref:
    m = [o for o in ours if o[0] == r["class"] and iou(o[2], r["box"]) >= 0.9]
    if not m or abs(m[0][1] - r["score"]) > a.tol:
        bad += 1; print(f"MISSING/DIFFERENT: ref {r['class']} {r['score']:.3f} {[round(v) for v in r['box']]} -> {m[0][1] if m else None}")
for o in ours:
    if not any(o[0] == r["class"] and iou(o[2], r["box"]) >= 0.9 for r in ref):
        bad += 1; print(f"EXTRA: {o[0]} {o[1]:.3f} {[round(v) for v in o[2]]}")
print(f"{'PASS' if not bad else 'FAIL'}: {len(ref)} reference boxes, {len(ours)} ours, {bad} mismatches ({Path(a.model).name})")
sys.exit(1 if bad else 0)
