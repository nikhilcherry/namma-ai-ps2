"""Compare multi-model / multi-scale recipes against the PROXY reference (yolo11m + tiling). Not human truth.
Recipe = list of (model, region) passes whose raw candidates are merged with grouped NMS + containment filter.
  python tools/hybrid_eval.py"""
import sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runner
from sweep import counts

R = Path(__file__).resolve().parent.parent
M = lambda n: runner.ExecutorchDetector(R / f"bundle/models/{n}.pte")
dets = {k: M(k) for k in ["yolo11n_raspberry_executorch_original", "yolo11n_fp32_960", "yolo11s_fp32_640", "yolo11m_fp32_640"]}
N640, N960, S640, M640 = list(dets)
frames = sorted((R / "devset/frames").glob("*.jpg"))


def tiles(img):
    h, w = img.shape[:2]; tw, th = int(w / 2 * 1.2), int(h / 2 * 1.2)
    return [(x0, y0, img[y0:y0 + th, x0:x0 + tw]) for x0 in (0, w - tw) for y0 in (0, h - th)]


def passes(img, spec):
    parts, ms = [], 0.0
    for model, region in spec:
        d = dets[model]
        if region == "full":
            c, t = d.raw(img, 0.05); parts.append(c); ms += t
        else:
            for x0, y0, crop in tiles(img):
                (xy, sc, ci), t = d.raw(crop, 0.05); ms += t
                xy = xy.copy(); xy[:, [0, 2]] += x0; xy[:, [1, 3]] += y0; parts.append((xy, sc, ci))
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3)), ms


def result(cand, h, iou=0.5, cv=.35):
    d = runner.containment_filter(runner.postprocess(cand, .35, cv, iou, True, .02, h))
    n = [x[0] for x in d]
    return np.array([n.count("person"), sum(1 for x in n if x != "person")])


recipes = {
    "n640 full": [(N640, "full")],
    "n960 full": [(N960, "full")],
    "s640 full": [(S640, "full")],
    "n640 full+tiles (current accurate)": [(N640, "full"), (N640, "tiles")],
    "n960 full + n640 tiles (hybrid)": [(N960, "full"), (N640, "tiles")],
    "s640 full + n640 tiles": [(S640, "full"), (N640, "tiles")],
    "s640 full + s640 tiles": [(S640, "full"), (S640, "tiles")],
    "n960 full + s640 full": [(N960, "full"), (S640, "full")],
}
CVS = [0.35, 0.30, 0.25, 0.20, 0.15]
ref = {}; vt = {(k, c): np.zeros(2) for k in recipes for c in CVS}; tot = {k: [np.zeros(2), 0.0, np.zeros(2)] for k in recipes}
for p in frames:
    img = cv2.imread(str(p))
    ref[p.stem] = result(passes(img, [(M640, "full"), (M640, "tiles")])[0], img.shape[0])
    for k, spec in recipes.items():
        c, ms = passes(img, spec)
        r = result(c, img.shape[0])
        tot[k][0] += abs(r - ref[p.stem]); tot[k][1] += ms; tot[k][2] += r - ref[p.stem]
        for c_ in CVS:
            vt[(k, c_)] += result(c, img.shape[0], cv=c_) - ref[p.stem]
print(f"reference (m640 full+tiles) totals over {len(frames)} frames: {sum(ref.values())}")
print(f"{'recipe':<40}{'|diff| ppl':>11}{'|diff| veh':>11}{'bias ppl':>10}{'bias veh':>10}{'ms/frame':>10}")
for k, (d, ms, b) in sorted(tot.items(), key=lambda kv: kv[1][0].sum()):
    print(f"{k:<40}{int(d[0]):>11}{int(d[1]):>11}{b[0]:>+10.0f}{b[1]:>+10.0f}{ms / len(frames):>10.0f}")

print("\nvehicle threshold study: signed vehicle bias (model - reference) / sum|diff| not shown; reference vehicles total", sum(v[1] for v in ref.values()))
print(f"{'recipe':<40}" + "".join(f"{'cv='+str(c):>9}" for c in CVS))
for k in recipes:
    print(f"{k:<40}" + "".join(f"{int(vt[(k, c)][1]):>+9d}" for c in CVS))
