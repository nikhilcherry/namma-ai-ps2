"""Threshold choice without truth: cross-model agreement. For each threshold t, count people / vehicles on every dev
frame with every cached model (full-frame candidates, grouped NMS), and measure how much the models disagree with each
other (mean pairwise |count difference| per frame, normalised by the mean count). Independent detectors share few
errors, so disagreement is lowest where each model's own threshold-sensitivity is lowest. This is a consistency
signal, not accuracy: models can all be wrong the same way (autos, tiny people).
  python tools/consensus.py [--models a b c]"""
import argparse, itertools, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner
from sweep import cache_model, counts

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="*", default=["yolo11n_raspberry_executorch_original", "yolo11n_raspberry_executorch_optimized",
                "yolo11n_fp32_960", "yolo11s_fp32_640", "yolo11m_fp32_640"])
ap.add_argument("--variant", default="full", choices=["full", "tiled"])
a = ap.parse_args()
R = Path(__file__).resolve().parent.parent
frames = sorted((R / "devset/frames").glob("*.jpg"))
ents = {m: cache_model(R / f"bundle/models/{m}.pte", frames, R / "devset/cache") for m in a.models}
TH = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
names = [p.stem for p in frames]
print(f"{len(a.models)} models, {len(names)} frames, variant={a.variant}\n")
for which, label in ((0, "people"), (1, "vehicles")):
    print(f"{label}: threshold | total count per model | mean pairwise |diff| per frame | disagreement / mean count")
    best = None
    for t in TH:
        C = np.array([[counts(ents[m][f], a.variant, t if which == 0 else 1.1, 1.1 if which == 0 else t, 0.6, True, 0.0)[which]
                       for f in names] for m in a.models], float)            # [models, frames]
        pair = np.mean([np.abs(C[i] - C[j]).mean() for i, j in itertools.combinations(range(len(a.models)), 2)])
        rel = pair / max(1e-9, C.mean())
        print(f"  {t:.2f} | {' '.join(f'{int(x):4d}' for x in C.sum(1))} | {pair:5.2f} | {rel:5.2f}")
        best = min(best, (rel, t)) if best else (rel, t)
    print(f"  -> lowest relative disagreement at {best[1]:.2f}\n")
