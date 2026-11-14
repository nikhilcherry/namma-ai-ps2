"""Export YOLO11 FP32 at a rectangular input size to ExecuTorch/XNNPACK and copy it into bundle/models.

  python tools/export_rect.py 384x640 480x864 [--weights yolo11n.pt]

Sizes are HxW and must be multiples of 32. A 16:9 clip letterboxed into 640x640 wastes about 44% of the
compute on padding; 384x640 keeps the same pixels per object for about 60% of the work.
Needs ultralytics + executorch + a matching torch (see bundle/requirements-pi.txt) and, once, the weights file
(Ultralytics downloads yolo11n.pt if it is missing; this is export tooling, the Pi pipeline stays offline).
Output name: bundle/models/<weights stem>_fp32_<H>x<W>.pte
"""
import argparse, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("sizes", nargs="+", help="HxW, e.g. 384x640")
ap.add_argument("--weights", default="yolo11n.pt")
a = ap.parse_args()

from ultralytics import YOLO

for sz in a.sizes:
    h, w = (int(v) for v in sz.lower().split("x"))
    if h % 32 or w % 32:
        sys.exit(f"{sz}: both sides must be multiples of 32")
    out = YOLO(a.weights).export(format="executorch", imgsz=[h, w])
    src = Path(out)
    pte = next(src.glob("*.pte")) if src.is_dir() else src
    dst = ROOT / "bundle/models" / f"{Path(a.weights).stem}_fp32_{h}x{w}.pte"
    shutil.copy(pte, dst)
    print(f"{sz}: {dst} ({dst.stat().st_size / 2**20:.1f} MB)")
