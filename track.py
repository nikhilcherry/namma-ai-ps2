"""Unique counts across video frames, without identities. Boxes are linked frame to frame by overlap (IoU) only:
no appearance features, no faces, and track numbers live only for the run. Optional counting line: a confirmed
track whose foot point ends up on the other side of the line from where it started is counted once.

Direction: `in` = towards the side on your right when you walk the line from its first point to its second
(in image coordinates: a line drawn left -> right counts moving DOWN the image as in). Swap the points to swap.
"""
import numpy as np

GROUP = {"person": "people", "car": "vehicles", "bus": "vehicles", "truck": "vehicles",
         "motorcycle": "vehicles", "bicycle": "vehicles"}


def iou_matrix(a, b):
    """a [n,4], b [m,4] xyxy -> [n,m] IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[None]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / np.maximum(1e-9, area(a) + area(b) - inter)


class Tracker:
    # ponytail: greedy IoU matching, no motion model. Packed crowds and frame gaps (--every > 3) swap or lose
    # tracks, which over-counts unique objects. Upgrade path: ByteTrack (Kalman prediction + low-score second pass).
    def __init__(self, line=None, iou_min=0.3, min_hits=3, max_miss=5):
        self.line = None if line is None else np.asarray(line, float)
        self.iou_min, self.min_hits, self.max_miss = iou_min, min_hits, max_miss
        self.tracks, self.next_id = [], 1
        self.unique = {"people": 0, "vehicles": 0}
        self.cross = {"people_in": 0, "people_out": 0, "vehicles_in": 0, "vehicles_out": 0}

    def _side(self, box):
        """+1 / -1 side of the line for the foot point, 0 on it; None when the foot is past the line's ends."""
        (x1, y1), (x2, y2) = self.line
        px, py = (box[0] + box[2]) / 2, box[3]
        dx, dy = x2 - x1, y2 - y1
        t = ((px - x1) * dx + (py - y1) * dy) / max(1e-9, dx * dx + dy * dy)
        if not 0.0 <= t <= 1.0:
            return None
        return int(np.sign(dx * (py - y1) - dy * (px - x1)))

    def update(self, dets):
        """dets: [(name, score, (x1,y1,x2,y2))] of one processed frame. Returns, per det, its track id once the
        track is confirmed (seen in min_hits frames), else None."""
        ids = [None] * len(dets)
        used = set()
        for grp in ("people", "vehicles"):
            di = [i for i, d in enumerate(dets) if GROUP.get(d[0]) == grp]
            tr = [t for t in self.tracks if t["grp"] == grp]
            m = iou_matrix([t["box"] for t in tr], [dets[i][2] for i in di])
            taken_t, taken_d = set(), set()
            for k in np.argsort(-m, axis=None):
                ti, dj = divmod(int(k), m.shape[1])
                if m[ti, dj] < self.iou_min:
                    break
                if ti in taken_t or dj in taken_d:
                    continue
                taken_t.add(ti); taken_d.add(dj)
                t = tr[ti]; t["box"] = dets[di[dj]][2]; t["hits"] += 1; t["miss"] = 0
                self._seen(t)
                used.add(id(t))
                if t["hits"] >= self.min_hits:
                    ids[di[dj]] = t["id"]
            for dj, i in enumerate(di):
                if dj not in taken_d:
                    t = {"id": self.next_id, "grp": grp, "box": dets[i][2], "hits": 1, "miss": 0,
                         "start": None, "counted": False}
                    self.next_id += 1
                    self.tracks.append(t); used.add(id(t))
                    self._seen(t)
        for t in self.tracks:
            if id(t) not in used:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]
        return ids

    def _seen(self, t):
        if t["hits"] == self.min_hits:
            self.unique[t["grp"]] += 1
        if self.line is None or t["counted"]:
            return
        s = self._side(t["box"])
        if not s:
            return
        if t["start"] is None:
            t["start"] = s
        elif s != t["start"] and t["hits"] >= self.min_hits:
            t["counted"] = True
            self.cross[f"{t['grp']}_{'in' if s > 0 else 'out'}"] += 1

    def totals(self):
        return {"people_unique": self.unique["people"], "vehicles_unique": self.unique["vehicles"], **self.cross}
