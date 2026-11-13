"""Detectors for main.py. ExecutorchDetector needs only executorch, torch, numpy, cv2
(no ultralytics), and follows the pre/post-processing in Arm's model card (config.yaml)."""
import time
from pathlib import Path

import cv2
import numpy as np

NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
IDS = sorted(NAMES)


class ExecutorchDetector:
    """YOLO11n .pte (INT8 or FP32): input [1,3,S,S] in 0..1, output [1,84,N]. S is read from the file."""

    def __init__(self, pte, iou=0.7):
        import torch
        from executorch.runtime import Runtime
        self.torch, self.iou = torch, iou
        self.method = Runtime.get().load_program(str(pte)).load_method("forward")
        try:                                   # square input side, read from the .pte itself
            self.size = int(self.method.metadata.input_tensor_meta(0).sizes()[-1])
        except Exception:
            self.size = 640

    def detect(self, img_bgr, conf=0.30, only_person=False):
        """Returns ([(name, score, (x1,y1,x2,y2))], inference_ms). Boxes in input-image pixels."""
        h0, w0 = img_bgr.shape[:2]
        S = self.size
        s = min(S / w0, S / h0)
        nw, nh = round(w0 * s), round(h0 * s)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        canvas = np.full((S, S, 3), 114, np.uint8)
        px, py = (S - nw) // 2, (S - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = self.torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float().div_(255).contiguous()
        t0 = time.perf_counter()
        out = self.method.execute([x])[0]
        ms = (time.perf_counter() - t0) * 1000
        pred = out.numpy()[0].T                      # [8400, 84]
        ids = [0] if only_person else IDS
        cls = pred[:, [4 + i for i in ids]]
        best = cls.argmax(1)
        score = cls[np.arange(len(cls)), best]
        keep = score > conf
        if not keep.any():
            return [], ms
        b, score, best = pred[keep, :4], score[keep], best[keep]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - px) / s).clip(0, w0)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - py) / s).clip(0, h0)
        wh = [[float(r[0]), float(r[1]), float(r[2] - r[0]), float(r[3] - r[1])] for r in xyxy]
        cid = [int(ids[i]) for i in best]
        idx = cv2.dnn.NMSBoxesBatched(wh, score.tolist(), cid, conf, self.iou)
        dets = [(NAMES[cid[i]], float(score[i]), tuple(float(v) for v in xyxy[i])) for i in np.array(idx).flatten()]
        return dets, ms


class UltralyticsDetector:
    """Fallback for models exported with Ultralytics (laptop use)."""

    def __init__(self, path):
        from ultralytics import YOLO
        self.model = YOLO(str(path), task="detect")

    def detect(self, img_bgr, conf=0.30, only_person=False):
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
