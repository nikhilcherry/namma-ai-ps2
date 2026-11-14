"""Does --group-score (vehicle score summed over its group) help? Measured against a PROXY, not human truth.

  python tools/group_eval.py [--model bundle/models/yolo11n_raspberry_executorch_original.pte]
                             [--proxy bundle/models/yolo11s_fp32_640.pte]

Reference per frame = proxy model with 5-pass tiling at 0.35/0.35, grouped NMS 0.5, min height 2% (the same proxy
tools/sweep.py uses). Candidate = the model, one full-frame pass, grouped NMS 0.5, person 0.30, with the vehicle
threshold swept, group scoring off and on. Output: per vehicle threshold, sum of |candidate - reference| vehicles over
all frames and the signed bias. A proxy says how close a cheap setting gets to the expensive one, not how close it is
to reality: confirm on devset/truth.csv before changing a default.
"""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import runner

VEH = {"car", "bus", "truck", "motorcycle", "bicycle"}
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "bundle/models/yolo11n_raspberry_executorch_original.pte"))
ap.add_argument("--proxy", default=str(ROOT / "bundle/models/yolo11s_fp32_640.pte"))
ap.add_argument("--frames", default=str(ROOT / "devset/frames"))
a = ap.parse_args()
frames = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
imgs = [cv2.imread(str(p)) for p in frames]


def nveh(cand, cv, h, tiled):
    d = runner.postprocess(cand, 0.35 if tiled else 0.30, cv, 0.5, True, 0.02 if tiled else 0.0, h)
    if tiled:
        d = runner.containment_filter(d)
    return sum(1 for x in d if x[0] in VEH), sum(1 for x in d if x[0] == "person")


proxy = runner.ExecutorchDetector(a.proxy)
ref = []
for im in imgs:
    cand, _, _ = runner.tiled_candidates(proxy, im, 0.05)
    ref.append(nveh(cand, 0.35, im.shape[0], True)[0])
print(f"reference (proxy {Path(a.proxy).name}, tiled): {sum(ref)} vehicles over {len(imgs)} frames")

det = runner.ExecutorchDetector(a.model)
rows = {}
for gs in (False, True):
    det.group_score = gs
    cands = [det.raw(im, 0.05)[0] for im in imgs]
    for cv in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        got = [nveh(c, cv, im.shape[0], False)[0] for c, im in zip(cands, imgs)]
        err = np.array(got) - np.array(ref)
        rows[(gs, cv)] = (int(np.abs(err).sum()), int(err.sum()))
print("\n| vehicle thr | abs diff, plain | bias, plain | abs diff, group-score | bias, group-score |\n|---|---|---|---|---|")
for cv in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
    (pa, pb), (ga, gb) = rows[(False, cv)], rows[(True, cv)]
    print(f"| {cv} | {pa} | {pb:+d} | {ga} | {gb:+d} |")
bp = min((v[0], -k[1]) for k, v in rows.items() if not k[0])
bg = min((v[0], -k[1]) for k, v in rows.items() if k[0])
print(f"\nbest plain: abs diff {bp[0]} at thr {-bp[1]}   best group-score: abs diff {bg[0]} at thr {-bg[1]}   (PROXY reference)")
