"""Add-ons computed from one frame's boxes. Plain numpy/cv2, no second model.

A congestion score, crowd level, pedestrians on the road, and head blur.
Counts only: no tracking, no identities.
"""
import json
from pathlib import Path

import cv2
import numpy as np

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}
BIKES = {"motorcycle", "bicycle"}


def load_zone(path, w, h):
    """Road polygon in pixels. JSON holds normalised [[x,y],...]; default = bottom 60%."""
    if path and Path(path).exists():
        pts = json.loads(Path(path).read_text())["road"]
    else:
        pts = [[0, 0.4], [1, 0.4], [1, 1], [0, 1]]
    return np.array([[int(x * w), int(y * h)] for x, y in pts], np.int32)


def zone_mask(zone, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [zone], 1)
    return m


def occupancy(dets, zmask):
    """Share of the road zone covered by vehicle boxes (overlaps count once)."""
    h, w = zmask.shape
    paint = np.zeros((h, w), np.uint8)
    for name, _, (x1, y1, x2, y2) in dets:
        if name in VEHICLES:
            paint[max(0, int(y1)):int(y2), max(0, int(x1)):int(x2)] = 1
    area = int(zmask.sum())
    return float((paint & zmask).sum()) / area if area else 0.0


def traffic_level(occ):
    return "FREE" if occ < 0.25 else "BUSY" if occ < 0.50 else "JAMMED"


def crowd_level(n_people, low=5, high=12):
    return "LOW" if n_people < low else "MED" if n_people < high else "HIGH"


def _overlap_frac(a, b):
    """Intersection area / area of a."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def pedestrians(dets, zmask):
    """(ped_on_road, riders). A person mostly inside a bike box is a rider."""
    h, w = zmask.shape
    bikes = [b for n, _, b in dets if n in BIKES]
    ped = riders = 0
    for name, _, box in dets:
        if name != "person":
            continue
        if any(_overlap_frac(box, bk) > 0.4 for bk in bikes):
            riders += 1
            continue
        fx, fy = int((box[0] + box[2]) / 2), min(h - 1, int(box[3]))
        if 0 <= fx < w and zmask[fy, fx]:
            ped += 1
    return ped, riders


def blur_heads(img, dets):
    """Blur the top quarter of every person box (faces) in place."""
    for name, _, (x1, y1, x2, y2) in dets:
        if name != "person":
            continue
        x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(x2), int(y2)
        hh = max(2, (y2 - y1) // 4)
        roi = img[y1:y1 + hh, x1:x2]
        if roi.size:
            k = max(3, (roi.shape[1] // 2) | 1)
            img[y1:y1 + hh, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    return img


def draw_overlay(img, zone, text_lines):
    cv2.polylines(img, [zone], True, (0, 200, 255), 2)
    for i, t in enumerate(text_lines):
        cv2.putText(img, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(img, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def load_count_region(path, w, h):
    """Optional 'count' polygon in zones.json (normalised). None = count the whole frame."""
    if path and Path(path).exists():
        pts = json.loads(Path(path).read_text()).get("count")
        if pts:
            return np.array([[int(x * w), int(y * h)] for x, y in pts], np.int32)
    return None


def in_region(dets, poly):
    """Keep boxes whose foot point (bottom centre) is inside the counting polygon."""
    if poly is None:
        return dets
    return [d for d in dets
            if cv2.pointPolygonTest(poly.astype(np.float32), ((d[2][0] + d[2][2]) / 2, d[2][3]), False) >= 0]
