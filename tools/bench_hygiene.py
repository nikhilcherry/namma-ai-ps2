"""M5: benchmark that survives questions. >=100 timed frames after warm-up, same frames for every model,
mean / median / p95 (never the best run), every config run TWICE, board + throttle state written next to the table.

  python tools/bench_hygiene.py --frames devset/frames --models n_int8=bundle/models/a.pte n_fp32=bundle/models/b.pte
  python tools/bench_hygiene.py --frames devset/frames --all           every bundle/models/*.pte
Writes out/bench_hygiene.md. A >10% difference between the two runs of a config is flagged as throttling."""
import argparse, glob, platform, statistics, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runner

ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True)
ap.add_argument("--models", nargs="*", default=[], help="label=path.pte")
ap.add_argument("--all", action="store_true")
ap.add_argument("--n", type=int, default=100, help="timed frames per run (cycles the folder)")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--out", default="out/bench_hygiene.md")
a = ap.parse_args()

files = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
imgs = [cv2.imread(str(f)) for f in files]
models = [m.split("=", 1) if "=" in m else (Path(m).stem, m) for m in a.models]
if a.all:
    models += [(Path(p).stem, p) for p in sorted(glob.glob("bundle/models/*.pte"))]


def sh(c):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def state():
    return {"throttled": sh("vcgencmd get_throttled").split("=")[-1] or "n/a",
            "temp": sh("vcgencmd measure_temp").split("=")[-1] or "n/a"}


def run(path):
    det = runner.load(path)
    for i in range(a.warmup):
        det.detect(imgs[i % len(imgs)])
    ms = []
    for i in range(a.n):
        ms.append(det.detect(imgs[i % len(imgs)])[1])
    return ms


board = (Path("/proc/device-tree/model").read_text().strip("\x00\n") if Path("/proc/device-tree/model").exists()
         else platform.machine() + " (not a Pi)")
head = [f"# Benchmark ({board})", "",
        f"Python {platform.python_version()}, {platform.platform()}, {a.n} timed frames per run after {a.warmup} warm-up, "
        f"{len(imgs)} distinct frames cycled, inference time only. Start state: {state()}.", "",
        "| model | run | mean ms | median ms | p95 ms | FPS (1000/mean) | temp after | throttled after |",
        "|---|---|---|---|---|---|---|---|"]
rows, flags = [], []
for label, path in models:
    means = []
    for r in (1, 2):
        ms = run(path)
        s = state()
        means.append(statistics.mean(ms))
        rows.append(f"| {label} | {r} | {statistics.mean(ms):.1f} | {statistics.median(ms):.1f} | "
                    f"{np.percentile(ms, 95):.1f} | {1000 / statistics.mean(ms):.2f} | {s['temp']} | {s['throttled']} |")
        print(rows[-1], flush=True)
    d = abs(means[0] - means[1]) / min(means)
    if d > 0.10:
        flags.append(f"- **{label}: runs differ by {d:.0%}** (> 10%). Throttling or noise: cool the board, close apps, rerun.")
tail = ["", *(flags or ["Both runs of every model agree within 10%."]),
        "", "Report mean, median and p95 together. Do not quote the faster run alone."]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text("\n".join(head + rows + tail) + "\n")
print("\n".join(tail)); print("wrote", a.out)
