"""Fair per-model inference timing: interleaved rounds on one frame, so drift hits every model equally.

  python tools/bench_models.py [--frame devset/frames/v_00.jpg] [--rounds 40] [--threads 0] [models...]

Each round runs every model once (order rotated), after a warm-up round. Prints median, p95 and the ratio to the
first model. Machine timings drift a lot on a laptop (the same model has measured 41 ms and 240 ms in different
sessions), so only the ratios inside one run mean anything, and none of it is a Pi number.
"""
import argparse, statistics, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import runner

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="*", default=["yolo11n_raspberry_executorch_original", "yolo11n_fp32_384x640",
                                              "yolo11n_fp32_480x864", "yolo11n_fp32_960",
                                              "yolo11n_raspberry_executorch_optimized"])
ap.add_argument("--frame", default=str(ROOT / "devset/frames/v_00.jpg"))
ap.add_argument("--rounds", type=int, default=40)
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
if a.threads:
    print("threads:", runner.set_threads(a.threads))
img = cv2.imread(a.frame)
print(f"frame {img.shape[1]}x{img.shape[0]}, {a.rounds} interleaved rounds")
dets = {m: runner.ExecutorchDetector(str(ROOT / f"bundle/models/{m}.pte")) for m in a.models}
for d in dets.values():
    d.raw(img, 0.05)                                   # warm-up
ts = {m: [] for m in a.models}
for r in range(a.rounds):
    order = a.models[r % len(a.models):] + a.models[:r % len(a.models)]
    for m in order:
        ts[m].append(dets[m].raw(img, 0.05)[1])
base = statistics.median(ts[a.models[0]])
print("\n| model | input HxW | median ms | p95 ms | vs first |\n|---|---|---|---|---|")
for m in a.models:
    med = statistics.median(ts[m])
    print(f"| {m} | {dets[m].hw[0]}x{dets[m].hw[1]} | {med:.1f} | {np.percentile(ts[m], 95):.1f} | {med / base:.2f}x |")
