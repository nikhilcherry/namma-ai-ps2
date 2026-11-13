"""Count people and vehicles per image/frame with a YOLO11n ExecuTorch (.pte) model."""
import argparse, csv, statistics, time
from pathlib import Path
from ultralytics import YOLO

VEHICLE = {"car", "motorcycle", "bus", "truck", "bicycle"}
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/yolo11n_executorch_model")
ap.add_argument("--input", required=True)
ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--out", default="out")
a = ap.parse_args()

model = YOLO(a.model, task="detect")
files = sorted(p for p in Path(a.input).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
Path(a.out).mkdir(exist_ok=True)
model.predict(str(files[0]), imgsz=640, verbose=False)  # warm-up
times, rows = [], []
for f in files:
    r = model.predict(str(f), imgsz=640, conf=a.conf, verbose=False)[0]
    names = [r.names[int(c)] for c in r.boxes.cls]
    person, vehicle = names.count("person"), sum(n in VEHICLE for n in names)
    ms = r.speed["inference"]; times.append(ms)
    r.save(filename=str(Path(a.out) / f"{f.stem}_det.jpg"))
    rows.append([f.stem, person, vehicle, round(ms, 1)])
    print(f"{f.stem:<13} person={person:<3} vehicle={vehicle:<3} {ms:5.1f} ms")
with open(Path(a.out) / "counts.csv", "w", newline="") as fh:
    csv.writer(fh).writerows([["frame", "person", "vehicle", "infer_ms"], *rows])
m = statistics.mean(times)
size = sum(p.stat().st_size for p in Path(a.model).glob("*.pte")) / 2**20
print("─" * 44)
print(f"frames={len(files)}  mean={m:.1f} ms  thru≈{1000/m:.1f} img/s  model={size:.1f} MB")
