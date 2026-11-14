"""Rectangular vs square inputs: count error against a PROXY and relative time, on devset/frames.

  python tools/rect_eval.py [--models name1 name2 ...] [--proxy bundle/models/yolo11s_fp32_640.pte] [--aspect 1.5]

Reference = proxy model with 5-pass tiling (0.35/0.35, grouped NMS 0.5, min height 2%), as tools/sweep.py. Candidates:
one full-frame pass, person/vehicle conf 0.30, grouped NMS 0.5. Prints per model: input HxW, sum |diff| people and
vehicles over the frames, signed bias, mean inference ms. Time is from THIS machine: compare models with each other
only; it is not a Pi number. `--aspect` keeps only frames at least that wide (w/h), since a rectangular input only
helps wide frames; use 0 for all frames.
"""
import argparse, statistics, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import runner

VEH = {"car", "bus", "truck", "motorcycle", "bicycle"}
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=["yolo11n_raspberry_executorch_original", "yolo11n_fp32_384x640",
                                                 "yolo11n_fp32_480x864", "yolo11n_fp32_960"])
ap.add_argument("--proxy", default=str(ROOT / "bundle/models/yolo11s_fp32_640.pte"))
ap.add_argument("--frames", default=str(ROOT / "devset/frames"))
ap.add_argument("--aspect", type=float, default=0.0)
a = ap.parse_args()

paths = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
imgs = [im for im in (cv2.imread(str(p)) for p in paths) if im is not None and im.shape[1] / im.shape[0] >= a.aspect]
print(f"{len(imgs)} frames (aspect >= {a.aspect}); widths/heights: "
      f"{sorted({(im.shape[1], im.shape[0]) for im in imgs})[:6]}")


def count(cand, h, tiled):
    d = runner.postprocess(cand, 0.35 if tiled else 0.30, 0.35 if tiled else 0.30, 0.5, True, 0.02 if tiled else 0.0, h)
    if tiled:
        d = runner.containment_filter(d)
    return sum(1 for x in d if x[0] == "person"), sum(1 for x in d if x[0] in VEH)


proxy = runner.ExecutorchDetector(a.proxy)
ref = [count(runner.tiled_candidates(proxy, im, 0.05)[0], im.shape[0], True) for im in imgs]
print(f"reference (proxy {Path(a.proxy).name}, tiled): {sum(r[0] for r in ref)} people, {sum(r[1] for r in ref)} vehicles\n")
print("| model | input HxW | sum abs diff people | bias | sum abs diff vehicles | bias | mean ms (this machine) |")
print("|---|---|---|---|---|---|---|")
for m in a.models:
    det = runner.ExecutorchDetector(str(ROOT / f"bundle/models/{m}.pte"))
    det.raw(imgs[0], 0.05)                                    # warm-up
    cs, ms = [], []
    for im in imgs:
        c, t = det.raw(im, 0.05); cs.append(c); ms.append(t)
    got = [count(c, im.shape[0], False) for c, im in zip(cs, imgs)]
    ep = np.array([g[0] - r[0] for g, r in zip(got, ref)]); ev = np.array([g[1] - r[1] for g, r in zip(got, ref)])
    print(f"| {m} | {det.hw[0]}x{det.hw[1]} | {int(np.abs(ep).sum())} | {int(ep.sum()):+d} | "
          f"{int(np.abs(ev).sum())} | {int(ev.sum()):+d} | {statistics.mean(ms):.0f} |")
