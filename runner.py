"""Detectors for main.py. ExecutorchDetector needs only executorch, torch, numpy, cv2
(no ultralytics), and follows the pre/post-processing in Arm's model card (config.yaml)."""
import time
from pathlib import Path

import cv2
import numpy as np

NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
IDS = sorted(NAMES)


GROUP = {0: "person", 2: "car", 5: "car", 7: "car", 1: "bike", 3: "bike"}
GROUP_ID = {"person": 0, "car": 1, "bike": 2}   # grouped NMS: car/bus/truck, bike/motorcycle
VEHICLE_IDS = {1, 2, 3, 5, 7}


def postprocess(cand, conf_person=0.30, conf_vehicle=0.30, iou=0.7, grouped=False,
                min_h=0.0, frame_h=None, only_person=False):
    """cand = (xyxy [n,4], score [n], cid [n]) raw candidates in image pixels, before NMS.
    Returns [(name, score, (x1,y1,x2,y2))]. Two thresholds, NMS per class or per group, min box height."""
    xyxy, score, cid = cand
    if len(score) == 0:
        return []
    thr = np.where(cid == 0, conf_person, conf_vehicle)
    keep = score >= thr
    if only_person:
        keep &= cid == 0
    if min_h and frame_h:
        keep &= (xyxy[:, 3] - xyxy[:, 1]) >= min_h * frame_h
    if not keep.any():
        return []
    xyxy, score, cid = xyxy[keep], score[keep], cid[keep]
    grp = np.array([GROUP_ID[GROUP[int(c)]] if grouped else int(c) for c in cid])
    wh = [[float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])] for r in xyxy]
    idx = cv2.dnn.NMSBoxesBatched(wh, score.tolist(), grp.tolist(), 0.0, iou)
    return [(NAMES[int(cid[i])], float(score[i]), tuple(float(v) for v in xyxy[i]))
            for i in np.array(idx).flatten()]


def set_threads(n):
    """Ask ExecuTorch for n CPU threads. The documented call is C++ (_unsafe_reset_threadpool); whether this Python
    build exposes it is unverified, so every known spelling is tried and the outcome is returned as text.
    Untested on device. Fallback that always works: `taskset -c 0-<n-1> python main.py ...` (tools/thread_sweep.sh)."""
    for modname in ("executorch.extension.pybindings.portable_lib", "executorch.extension.pybindings._portable_lib",
                    "executorch.extension.pybindings.pybindings", "executorch.extension.threadpool"):
        try:
            mod = __import__(modname, fromlist=["x"])
        except Exception:
            continue
        for fn in ("_unsafe_reset_threadpool", "set_threadpool_size", "set_num_threads"):
            f = getattr(mod, fn, None)
            if f:
                f(int(n))
                return f"{modname}.{fn}({n})"
    return "no Python threadpool binding found in this executorch build; use taskset (tools/thread_sweep.sh)"


class ExecutorchDetector:
    """YOLO11 .pte (INT8 or FP32): input [1,3,S,S] in 0..1, output [1,84,N]. S is read from the file."""

    def __init__(self, pte, iou=0.7):
        import torch
        from executorch.runtime import Runtime
        self.torch, self.iou = torch, iou
        self.method = Runtime.get().load_program(str(pte)).load_method("forward")
        try:                                   # input H and W, read from the .pte itself (rectangular ok)
            sz = self.method.metadata.input_tensor_meta(0).sizes()
            self.hw = (int(sz[-2]), int(sz[-1]))
        except Exception:
            self.hw = (640, 640)
        self.size = max(self.hw)

    def raw(self, img_bgr, floor=0.05):
        """Raw candidates above `floor` before NMS: ((xyxy, score, cid), inference_ms). Pixels of img_bgr."""
        t_in = time.perf_counter()
        h0, w0 = img_bgr.shape[:2]
        SH, SW = self.hw
        s = min(SW / w0, SH / h0)
        nw, nh = round(w0 * s), round(h0 * s)
        # resize first, colour-convert the small image (not the full 1080p frame)
        small = cv2.cvtColor(cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        canvas = np.full((SH, SW, 3), 114, np.uint8)
        px, py = (SW - nw) // 2, (SH - nh) // 2
        canvas[py:py + nh, px:px + nw] = small
        x = self.torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float().div_(255).contiguous()
        t0 = time.perf_counter()
        self.pre_ms = getattr(self, "pre_ms", 0.0) + (t0 - t_in) * 1000
        out = self.method.execute([x])[0]
        ms = (time.perf_counter() - t0) * 1000
        pred = out.numpy()[0].T                      # [N, 84]
        cls = pred[:, [4 + i for i in IDS]]
        best = cls.argmax(1)
        score = cls[np.arange(len(cls)), best]
        keep = score > floor
        b, score, best = pred[keep, :4], score[keep], best[keep]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - px) / s).clip(0, w0)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - py) / s).clip(0, h0)
        cid = np.array([IDS[i] for i in best], dtype=np.int64)
        return (xyxy.astype(np.float32), score.astype(np.float32), cid), ms

    def detect(self, img_bgr, conf=0.30, only_person=False, conf_vehicle=None, iou=None,
               grouped=False, min_h=0.0):
        """Returns ([(name, score, (x1,y1,x2,y2))], inference_ms). Boxes in input-image pixels."""
        cv = conf if conf_vehicle is None else conf_vehicle
        cand, ms = self.raw(img_bgr, floor=min(conf, cv))
        return postprocess(cand, conf, cv, self.iou if iou is None else iou, grouped, min_h,
                           img_bgr.shape[0], only_person), ms


def containment_filter(dets, thr=0.7):
    """Drop a box that is mostly inside a larger box of the same group (halves cut by a tile edge)."""
    out = []
    for i, (n, sc, b) in enumerate(dets):
        a = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        dup = False
        for j, (n2, _, b2) in enumerate(dets):
            if i == j or GROUP_BY_NAME[n2] != GROUP_BY_NAME[n]:
                continue
            a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            if a2 <= a:
                continue
            ix = max(0, min(b[2], b2[2]) - max(b[0], b2[0]))
            iy = max(0, min(b[3], b2[3]) - max(b[1], b2[1]))
            if ix * iy / a > thr:
                dup = True
                break
        if not dup:
            out.append((n, sc, b))
    return out


GROUP_BY_NAME = {NAMES[k]: v for k, v in GROUP.items()}


def needs_tiles(cand, frame_h, conf=0.30, low=(0.15, 0.30), n_low=40, small_h=0.10):
    """Adaptive trigger from the full-frame candidates: many borderline boxes (crowd / small objects) or small
    people. Rule read off the 24 dev frames (devset/FINDINGS.md): treat as a starting point, not a tuned value."""
    xyxy, score, cid = cand
    if int(((score > low[0]) & (score < low[1])).sum()) >= n_low:
        return True
    pm = (cid == 0) & (score >= conf)
    if pm.any():
        hs = (xyxy[pm, 3] - xyxy[pm, 1]) / frame_h
        return float(np.median(hs)) < small_h
    return False


def tiled_candidates(det, img, floor=0.05, full=None, only_if=None):
    """One full-frame pass plus four tiles with 20% overlap, all classes. Returns (candidates, ms, passes).
    only_if(full_cand, frame_h) -> bool: skip the tiles (1 pass) when it says no (adaptive tiling)."""
    h, w = img.shape[:2]
    tw, th = int(w / 2 * 1.2), int(h / 2 * 1.2)
    parts, total = [], 0.0
    c, ms = full if full is not None else det.raw(img, floor); parts.append(c); total += ms
    if only_if is not None and not only_if(c, h):
        return c, total, 1
    for x0 in (0, w - tw):
        for y0 in (0, h - th):
            (xy, sc, ci), ms = det.raw(img[y0:y0 + th, x0:x0 + tw], floor)
            total += ms
            xy = xy.copy(); xy[:, [0, 2]] += x0; xy[:, [1, 3]] += y0
            parts.append((xy, sc, ci))
    cand = tuple(np.concatenate([p[k] for p in parts]) for k in range(3))
    return cand, total, 5


class UltralyticsDetector:
    """Fallback for models exported with Ultralytics (laptop use)."""

    def __init__(self, path):
        from ultralytics import YOLO
        self.model = YOLO(str(path), task="detect")

    def detect(self, img_bgr, conf=0.30, only_person=False, **_):
        r = self.model.predict(img_bgr, imgsz=640, conf=conf, verbose=False,
                               classes=[0] if only_person else IDS)[0]
        dets = [(r.names[int(c)], s, tuple(b)) for b, c, s in
                zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        return dets, r.speed["inference"]


def load(model, backend="auto"):
    p = Path(model)
    if backend == "auto":
        backend = "et" if p.suffix == ".pte" else "ultralytics"
    return ExecutorchDetector(p) if backend == "et" else UltralyticsDetector(p)
