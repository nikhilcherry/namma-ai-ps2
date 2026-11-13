"""PS2: count people and vehicles per frame with an ExecuTorch YOLO11 .pte, plus
small box-maths add-ons (congestion, crowd level, pedestrians on road, head blur).

  python main.py --source samples/                  folder of images
  python main.py --source clip.mp4 --every 3        video, every 3rd frame
  python main.py --source clip.mp4 --zones zones.json --deploy   CSV only, no frames saved
  python main.py --source clip.mp4 --profile fast|accurate       one flag sets the whole recipe
"""
import argparse, csv, itertools, json, platform, statistics, time
try:
    import resource
except ImportError:          # Windows laptops
    resource = None
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import insights, runner, stamp

KEEP = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]
IMG = {".jpg", ".jpeg", ".png", ".bmp"}
COLUMNS = ["frame", "t_s", "person", "car", "bus", "truck", "motorcycle", "bicycle", "vehicle",
           "person_smooth", "vehicle_smooth", "person_tiled", "occupancy", "traffic_level",
           "crowd_level", "ped_on_road", "riders", "reused", "infer_ms"]

# Starting guesses, replaced by measurements (tools/sweep.py, benchmark table).
PROFILES = {
    "fast": dict(model="bundle/models/yolo11n_raspberry_executorch_optimized.pte", tile="off",
                 deploy=True, no_addons=False, skip_static=0.5),
    "balanced": dict(model="bundle/models/yolo11n_fp32_960.pte", tile="off", deploy=False, no_addons=False,
                     smooth=True),   # one 960 pass: most of tiling's people gain at ~1.4x cost (devset/FINDINGS.md)
    "ensemble": dict(model="bundle/models/yolo11n_fp32_960.pte", also="bundle/models/yolo11s_fp32_640.pte",
                     tile="off", deploy=False, no_addons=False, smooth=True),   # ~215 ms: beats any single pass (devset/FINDINGS.md)
    "accurate": dict(model="bundle/models/yolo11n_raspberry_executorch_original.pte", tile="all",
                     deploy=False, no_addons=False, smooth=True),
}

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="image folder or video file")
ap.add_argument("--profile", choices=list(PROFILES), help="fast = INT8, no tiling, no images, skip static frames; "
                "accurate = FP32, tiling on, smoothing on. Explicit flags override.")
ap.add_argument("--model", default="bundle/models/yolo11n_raspberry_executorch_optimized.pte",
                help=".pte file (Arm INT8/FP32 via ExecuTorch) or an Ultralytics export dir")
ap.add_argument("--also", default=None, help="second .pte: its full-frame candidates are merged with the first "
                "model's before NMS (a 2-model ensemble; times are summed)")
ap.add_argument("--backend", choices=["auto", "et", "ultralytics"], default="auto")
ap.add_argument("--out", default="out")
ap.add_argument("--conf", type=float, default=0.30, help="person threshold (and default for vehicles)")
ap.add_argument("--conf-person", type=float, default=None)
ap.add_argument("--conf-vehicle", type=float, default=None)
ap.add_argument("--iou", type=float, default=0.70, help="NMS IoU")
ap.add_argument("--ungrouped", action="store_true", help="NMS per class instead of per group (old behaviour)")
ap.add_argument("--min-h", type=float, default=0.0, help="drop boxes shorter than this share of frame height")
ap.add_argument("--every", type=int, default=1, help="video: use every Nth frame")
ap.add_argument("--threads", type=int, default=0,
                help="ExecuTorch CPU threads (0 = runtime default = all cores; Arm's reference used 1). Untested on device")
ap.add_argument("--repeat", type=int, default=1,
                help="image folder: run inference N times over the folder for timing only (rows and counts are written once)")
ap.add_argument("--zones", default=None, help="JSON with a normalised 'road' polygon, optional 'count' polygon")
ap.add_argument("--no-addons", action="store_true", help="plain counting only")
ap.add_argument("--blur", action=argparse.BooleanOptionalAction, default=True,
                help="blur heads in saved frames (default on; --no-blur to disable)")
ap.add_argument("--deploy", action="store_true", help="write the CSV only, no frames")
ap.add_argument("--tile", choices=["off", "auto", "all"], default="off",
                help="auto: people-only 2x2 when people look small; all: full frame + 4 tiles, all classes")
ap.add_argument("--smooth", action="store_true", help="video: rolling median over 5 processed frames")
ap.add_argument("--skip-static", type=float, nargs="?", const=0.5, default=0.0, metavar="MAD",
                help="fixed camera: reuse the last counts if mean abs diff of a 160x90 grey copy is below MAD "
                     "(default 0.5, measured: 2.0 and 1.0 changed a count by 2); a real inference is forced at least every 10th processed frame")
pre, _ = ap.parse_known_args()
if pre.profile:
    ap.set_defaults(**PROFILES[pre.profile])
a = ap.parse_args()
CP = a.conf if a.conf_person is None else a.conf_person
CV = a.conf if a.conf_vehicle is None else a.conf_vehicle
GROUPED = not a.ungrouped

src = Path(a.source)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
THREADS_NOTE = runner.set_threads(a.threads) if a.threads else "default"
if a.threads:
    print("threads:", THREADS_NOTE)
det = runner.load(a.model, a.backend)
det2 = runner.load(a.also, a.backend) if a.also else None
is_video = not src.is_dir()
if a.smooth and not is_video:
    a.smooth = False                       # never smooth a folder of unrelated images
T = {"decode": 0.0, "pre": 0.0, "infer": 0.0, "post": 0.0, "addons": 0.0, "save": 0.0}


def pi_state():
    """Temperature (C) and throttle bits from vcgencmd; None when not on a Pi."""
    import shutil, subprocess
    if not shutil.which("vcgencmd"):
        return None
    try:
        t = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=3).stdout
        th = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3).stdout
        return {"temp_c": float(t.split("=")[1].split("'")[0]), "throttled": th.strip().split("=")[1]}
    except Exception:
        return None


PI0 = pi_state()


def frames():
    """Yield (name, t_seconds, BGR image). Folder: stills, video: sampled frames.
    Frames that --every drops are only grab()bed: the codec still decodes them, but no image is retrieved or colour-converted."""
    if src.is_dir():
        for i, f in enumerate(sorted(p for p in src.iterdir() if p.suffix.lower() in IMG)):
            t0 = time.perf_counter(); img = cv2.imread(str(f)); T["decode"] += time.perf_counter() - t0
            yield f.stem, float(i), img
        return
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; i = 0
    while True:
        t0 = time.perf_counter()
        if not cap.grab():
            break
        img = None
        if i % a.every == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
        T["decode"] += time.perf_counter() - t0
        if img is not None:
            yield f"f{i:05d}", i / fps, img
        i += 1


def detect(img):
    """(KEEP-class dets, model_ms, passes). Tiling modes run several passes; model_ms is their sum."""
    if a.tile == "all" and hasattr(det, "raw"):
        cand, ms, passes = runner.tiled_candidates(det, img, min(CP, CV))
        t0 = time.perf_counter()
        d = runner.postprocess(cand, CP, CV, min(a.iou, 0.5), GROUPED, a.min_h, img.shape[0])
        d = runner.containment_filter(d)
        T["post"] += time.perf_counter() - t0
        return [x for x in d if x[0] in KEEP], ms, passes
    dets, ms = _plain(img)
    return [d for d in dets if d[0] in KEEP], ms, 1


def _plain(img):
    if hasattr(det, "raw"):
        cand, ms = det.raw(img, min(CP, CV))
        if det2 is not None:
            c2, ms2 = det2.raw(img, min(CP, CV)); ms += ms2
            cand = tuple(np.concatenate([cand[k], c2[k]]) for k in range(3))
        t0 = time.perf_counter()
        d = runner.postprocess(cand, CP, CV, a.iou, GROUPED, a.min_h, img.shape[0])
        if det2 is not None:
            d = runner.containment_filter(d)       # same-group box mostly inside a bigger one = the other model's duplicate
        T["post"] += time.perf_counter() - t0
        return d, ms
    return det.detect(img, min(CP, CV))


def tiled_people(img):
    """2x2 tiles with 20% overlap, people only, merged by NMS. Returns count (legacy --tile auto)."""
    h, w = img.shape[:2]; tw, th = int(w / 2 * 1.2), int(h / 2 * 1.2)
    boxes, scores = [], []
    for x0 in (0, w - tw):
        for y0 in (0, h - th):
            d, _ = det.detect(img[y0:y0 + th, x0:x0 + tw], CP, only_person=True)
            for _, s, b in d:
                boxes.append([b[0] + x0, b[1] + y0, b[2] - b[0], b[3] - b[1]]); scores.append(s)
    if not boxes:
        return 0
    return len(cv2.dnn.NMSBoxes(boxes, scores, CP, 0.5))


class ZoneCache:
    """Zones.json is read and the road mask rasterised once per frame size, not once per frame."""
    def __init__(self, path):
        self.path, self.k, self.cp, self.rd = path, None, None, None

    def _fill(self, w, h):
        if self.k != (w, h):
            zone = insights.load_zone(self.path, w, h)
            self.k, self.cp = (w, h), insights.load_count_region(self.path, w, h)
            self.rd = (zone, insights.zone_mask(zone, w, h))

    def count_poly(self, w, h):
        self._fill(w, h); return self.cp

    def road(self, w, h):
        self._fill(w, h); return self.rd


ZC = ZoneCache(a.zones)


def small_grey(img):
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA)


files = frames()            # streamed: a 20 s 1080p clip held in memory would not fit on a Pi
first = next(files, None)
if first is None:
    raise SystemExit(f"no frames found in {src}")
detect(first[2])  # warm-up, not timed
for k in T:
    T[k] = 0.0
for d_ in (det, det2):
    if d_ is not None and hasattr(d_, "pre_ms"):
        d_.pre_ms = 0.0
files = itertools.chain([first], files)
times, e2e, rows, occ_hist, alldets = [], [], [], deque(maxlen=5), {}
last_grey, last_state, since_real, n_real = None, None, 0, 0
for name, t, img in files:
    t0 = time.perf_counter()
    reused = 0
    grey = small_grey(img) if a.skip_static else None
    if (a.skip_static and last_state is not None and since_real < 9 and
            float(np.mean(cv2.absdiff(grey, last_grey))) < a.skip_static):
        dets, ms, reused = last_state, 0.0, 1
        since_real += 1
    else:
        dets, ms, _ = detect(img)
        n_real += 1; since_real = 0
        last_grey, last_state = grey, dets
    ta = time.perf_counter()
    zones_poly = ZC.count_poly(img.shape[1], img.shape[0])
    dets = insights.in_region(dets, zones_poly)
    times.append(ms)
    names = [d[0] for d in dets]
    cnt = {k: names.count(k) for k in KEEP}
    veh = sum(cnt[k] for k in KEEP if k != "person")
    row = {"frame": name, "t_s": round(t, 2), **cnt, "vehicle": veh, "person_smooth": "", "vehicle_smooth": "",
           "person_tiled": "", "occupancy": "", "traffic_level": "", "crowd_level": "", "ped_on_road": "",
           "riders": "", "reused": reused, "infer_ms": round(ms, 1)}
    if not a.no_addons:
        h, w = img.shape[:2]
        zone, zmask = ZC.road(w, h)
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
    T["addons"] += time.perf_counter() - ta
    rows.append(row)
    e2e.append((time.perf_counter() - t0) * 1000)   # pre-process + inference + post-process + add-ons
    alldets[name] = {"w": img.shape[1], "h": img.shape[0],
                     "boxes": [[n, round(s, 3)] + [round(v, 1) for v in b] for n, s, b in dets]}
    print(f"{name:<14} person={cnt['person']:<3} vehicle={veh:<3} "
          f"{row['traffic_level'] or '-':<7} {row['crowd_level'] or '-':<5} {ms:6.1f} ms{' (reused)' if reused else ''}")
    if not a.deploy:
        ts = time.perf_counter()
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
        T["save"] += time.perf_counter() - ts

if a.repeat > 1 and not is_video:       # timing only: more samples for mean / median / p95, no extra rows
    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG)
    for _ in range(a.repeat - 1):
        for p in imgs:
            times.append(detect(cv2.imread(str(p)))[1])
PI1 = pi_state()

# rolling median over 5 processed frames, centred; video only
for k, key in (("person", "person_smooth"), ("vehicle", "vehicle_smooth")):
    vals = [r[k] for r in rows]
    for i, r in enumerate(rows):
        r[key] = int(statistics.median(vals[max(0, i - 2):i + 3])) if a.smooth else r[k]

with open(out / "counts.csv", "w", newline="") as fh:
    w_ = csv.DictWriter(fh, fieldnames=COLUMNS); w_.writeheader(); w_.writerows(rows)

# stamp hashes the rows exactly as read back from the CSV
with open(out / "counts.csv") as fh:
    st = stamp.write(out, list(csv.DictReader(fh)), a.model, a.source)

(out / "detections.json").write_text(json.dumps(alldets))

pte = sorted(Path(a.model).glob("*.pte")) if Path(a.model).is_dir() else [Path(a.model)]
size_mb = pte[0].stat().st_size / 2**20
real = [t for t in times if t > 0] or [0.0]
m, med = statistics.mean(real), statistics.median(real)
p95 = float(np.percentile(real, 95))
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 if resource else 0.0
peak_ppl = max(rows, key=lambda r: r["person"])
print("─" * 60)
print(f"frames={len(rows)}  model={pte[0].name} ({size_mb:.1f} MB)  profile={a.profile or 'custom'}")
print(f"inference ms: mean={m:.1f} median={med:.1f} p95={p95:.1f} std={statistics.pstdev(real):.1f}"
      + (f"   (frames needing inference: {n_real} of {len(rows)})" if a.skip_static else ""))
print(f"throughput (inference only) ≈ {1000/m:.1f} FPS   peak RAM ≈ {peak_mb:.0f} MB (whole process)")
me = statistics.mean(e2e)
print(f"end to end (no image saving): mean={me:.1f} ms ≈ {1000/me:.1f} FPS")
T["pre"] = (getattr(det, "pre_ms", 0.0) + (getattr(det2, "pre_ms", 0.0) if det2 else 0.0)) / 1000
T["infer"] = sum(times) / 1000
tot = sum(T.values()) or 1.0
shares = {k: round(100 * v / tot, 1) for k, v in T.items()}
print("time shares %: " + "  ".join(f"{k} {v}" for k, v in shares.items()))
print(f"peak people: {peak_ppl['person']} at {peak_ppl['frame']}")
print(f"stamp: model {st['model_sha256'][:12]}…  input {st['source_sha256'][:12]}…  head {st['head'][:16]}…")
(out / "summary.txt").write_text(
    f"frames={len(rows)} model={pte[0].name} size_mb={size_mb:.2f}\n"
    f"infer_ms mean={m:.1f} median={med:.1f} p95={p95:.1f}\nfps_infer_only={1000/m:.1f}\n"
    f"e2e_ms mean={me:.1f}\nfps_end_to_end={1000/me:.1f}\n"
    f"peak_ram_mb={peak_mb:.0f}\nstamp_head={st['head']}\n")
try:
    board = Path("/proc/device-tree/model").read_text().strip("\x00\n")
except OSError:
    board = platform.machine() + " (not a Pi)"
(out / "summary.json").write_text(json.dumps({
    "board": board, "model": pte[0].name, "input": getattr(det, "size", 640), "size_mb": round(size_mb, 2),
    "profile": a.profile or "custom", "tile": a.tile, "also": a.also, "conf_person": CP, "conf_vehicle": CV, "iou": a.iou,
    "grouped_nms": GROUPED, "min_h": a.min_h, "smooth": a.smooth,
    "frames": len(rows), "frames_inferred": n_real, "mean_ms": round(m, 1), "median_ms": round(med, 1),
    "p95_ms": round(p95, 1), "std_ms": round(statistics.pstdev(real), 1), "fps_infer": round(1000 / m, 2),
    "e2e_ms": round(me, 1), "fps_e2e": round(1000 / me, 2), "stage_share_pct": shares,
    "peak_ram_mb": round(peak_mb),
    "people_total": sum(r["person"] for r in rows), "vehicle_total": sum(r["vehicle"] for r in rows),
    "repeat": a.repeat, "threads": a.threads, "threads_note": THREADS_NOTE, "samples": len(times), "pi_start": PI0, "pi_end": PI1,
    "throttled_during_run": bool(PI1 and (PI1["throttled"] != "0x0" or (PI0 and PI0["throttled"] != PI1["throttled"]))),
    "zones": bool(a.zones and Path(a.zones).exists()), "source": str(a.source), "every": a.every, "python": platform.python_version()}, indent=1))
