"""PS2: count people and vehicles per frame with an ExecuTorch YOLO11n .pte, plus
small box-maths add-ons (congestion, crowd level, pedestrians on road, head blur).

  python main.py --source samples/                  folder of images
  python main.py --source clip.mp4 --every 3        video, every 3rd frame
  python main.py --source clip.mp4 --zones zones.json --deploy   CSV only, no frames saved
"""
import argparse, csv, resource, statistics, time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import insights, stamp

KEEP = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]
IMG = {".jpg", ".jpeg", ".png", ".bmp"}
COLUMNS = ["frame", "t_s", "person", "car", "bus", "truck", "motorcycle", "bicycle", "vehicle",
           "person_tiled", "occupancy", "traffic_level", "crowd_level", "ped_on_road", "riders",
           "infer_ms"]

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="image folder or video file")
ap.add_argument("--model", default="models/yolo11n_executorch_model")
ap.add_argument("--out", default="out")
ap.add_argument("--conf", type=float, default=0.30)
ap.add_argument("--every", type=int, default=1, help="video: use every Nth frame")
ap.add_argument("--zones", default=None, help="JSON with a normalised 'road' polygon")
ap.add_argument("--no-addons", action="store_true", help="plain counting only")
ap.add_argument("--blur", action="store_true", help="blur heads in saved frames")
ap.add_argument("--deploy", action="store_true", help="write the CSV only, no frames")
ap.add_argument("--tile", choices=["off", "auto"], default="off")
a = ap.parse_args()

src = Path(a.source)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model = YOLO(a.model, task="detect")


def frames():
    """Yield (name, t_seconds, BGR image). Folder: stills, video: sampled frames."""
    if src.is_dir():
        for i, f in enumerate(sorted(p for p in src.iterdir() if p.suffix.lower() in IMG)):
            yield f.stem, float(i), cv2.imread(str(f))
        return
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % a.every == 0:
            yield f"f{i:05d}", i / fps, img
        i += 1


def detect(img, classes=None):
    r = model.predict(img, imgsz=640, conf=a.conf, verbose=False, classes=classes)[0]
    dets = []
    for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        n = r.names[int(c)]
        if n in KEEP:
            dets.append((n, s, tuple(b)))
    return dets, r.speed["inference"]


def tiled_people(img):
    """2x2 tiles with 20% overlap, people only, merged by NMS. Returns count."""
    h, w = img.shape[:2]; tw, th = int(w / 2 * 1.2), int(h / 2 * 1.2)
    boxes, scores = [], []
    for x0 in (0, w - tw):
        for y0 in (0, h - th):
            r = model.predict(img[y0:y0 + th, x0:x0 + tw], imgsz=640, conf=a.conf, classes=[0],
                              verbose=False)[0]
            for b, s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                boxes.append([b[0] + x0, b[1] + y0, b[2] - b[0], b[3] - b[1]]); scores.append(s)
    if not boxes:
        return 0
    return len(cv2.dnn.NMSBoxes(boxes, scores, a.conf, 0.5))


files = list(frames())
if not files:
    raise SystemExit(f"no frames found in {src}")
is_video = not src.is_dir()
detect(files[0][2])  # warm-up, not timed
times, rows, occ_hist = [], [], deque(maxlen=5)
for name, t, img in files:
    t0 = time.perf_counter()
    dets, ms = detect(img); times.append(ms)
    names = [d[0] for d in dets]
    cnt = {k: names.count(k) for k in KEEP}
    veh = sum(cnt[k] for k in KEEP if k != "person")
    row = {"frame": name, "t_s": round(t, 2), **cnt, "vehicle": veh, "person_tiled": "",
           "occupancy": "", "traffic_level": "", "crowd_level": "", "ped_on_road": "",
           "riders": "", "infer_ms": round(ms, 1)}
    if not a.no_addons:
        h, w = img.shape[:2]
        zone = insights.load_zone(a.zones, w, h); zmask = insights.zone_mask(zone, w, h)
        occ = insights.occupancy(dets, zmask)
        occ_hist.append(occ)
        occ_s = float(np.median(occ_hist)) if is_video else occ   # smoothing only for real video
        ped, riders = insights.pedestrians(dets, zmask)
        n_ppl = cnt["person"]
        heights = [b[3] - b[1] for n, _, b in dets if n == "person"]
        if a.tile == "auto" and heights and statistics.median(heights) < 60:
            n_ppl = row["person_tiled"] = tiled_people(img)
        row.update(occupancy=round(occ_s, 3), traffic_level=insights.traffic_level(occ_s),
                   crowd_level=insights.crowd_level(n_ppl), ped_on_road=ped, riders=riders)
    rows.append(row)
    print(f"{name:<14} person={cnt['person']:<3} vehicle={veh:<3} "
          f"{row['traffic_level'] or '-':<7} {row['crowd_level'] or '-':<5} {ms:6.1f} ms")
    if not a.deploy:
        vis = img.copy()
        if a.blur:
            insights.blur_heads(vis, dets)
        for n, s, (x1, y1, x2, y2) in dets:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
            cv2.putText(vis, f"{n} {s:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 220, 0), 1)
        lines = [f"people {cnt['person']}  vehicles {veh}"]
        if not a.no_addons:
            lines.append(f"traffic {row['traffic_level']} ({row['occupancy']})  crowd {row['crowd_level']}")
            lines.append(f"on road {row['ped_on_road']}  riders {row['riders']}")
            insights.draw_overlay(vis, zone, lines)
        else:
            insights.draw_overlay(vis, np.zeros((0, 1, 2), np.int32), lines)
        cv2.imwrite(str(out / f"{name}_det.jpg"), vis)

with open(out / "counts.csv", "w", newline="") as fh:
    w_ = csv.DictWriter(fh, fieldnames=COLUMNS); w_.writeheader(); w_.writerows(rows)

# stamp hashes the rows exactly as read back from the CSV
with open(out / "counts.csv") as fh:
    st = stamp.write(out, list(csv.DictReader(fh)), a.model, a.source)

pte = sorted(Path(a.model).glob("*.pte")) if Path(a.model).is_dir() else [Path(a.model)]
size_mb = pte[0].stat().st_size / 2**20
m, med = statistics.mean(times), statistics.median(times)
p95 = float(np.percentile(times, 95))
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
peak_ppl = max(rows, key=lambda r: r["person"])
print("─" * 60)
print(f"frames={len(rows)}  model={pte[0].name} ({size_mb:.1f} MB)")
print(f"inference ms: mean={m:.1f} median={med:.1f} p95={p95:.1f} std={statistics.pstdev(times):.1f}")
print(f"throughput (inference only) ≈ {1000/m:.1f} FPS   peak RAM ≈ {peak_mb:.0f} MB (whole process)")
print(f"peak people: {peak_ppl['person']} at {peak_ppl['frame']}")
print(f"stamp: model {st['model_sha256'][:12]}…  input {st['source_sha256'][:12]}…  head {st['head'][:16]}…")
(out / "summary.txt").write_text(
    f"frames={len(rows)} model={pte[0].name} size_mb={size_mb:.2f}\n"
    f"infer_ms mean={m:.1f} median={med:.1f} p95={p95:.1f}\nfps_infer_only={1000/m:.1f}\n"
    f"peak_ram_mb={peak_mb:.0f}\nstamp_head={st['head']}\n")
