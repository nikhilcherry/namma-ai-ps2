"""Accuracy check against a human count.

  python tools/handcount.py sheet --source clip.mp4 --run out/<model> [-n 10]
      -> handcount/<frame>.jpg (clean frames, no boxes), handcount/sheet.jpg, handcount/handcount.csv
  ...a person who has NOT seen the model's output fills person_hand and vehicle_hand in the CSV...
  python tools/handcount.py table --run out/<model>
      -> handcount/accuracy.md  (model vs human, per frame and overall)

Count rule to tell the human counter: every person and every vehicle (car, bus, truck, auto,
two-wheeler, bicycle) at least half visible. A rider on a two-wheeler is a person AND a vehicle.
"""
import argparse, csv, sys
from pathlib import Path

import cv2
import numpy as np

IMG = {".jpg", ".jpeg", ".png", ".bmp"}
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["sheet", "table"])
ap.add_argument("--source")
ap.add_argument("--run", required=True, help="out/<model> folder with counts.csv")
ap.add_argument("--dir", default="handcount")
ap.add_argument("-n", type=int, default=10)
a = ap.parse_args()
hd = Path(a.dir); hd.mkdir(exist_ok=True)
with open(Path(a.run) / "counts.csv") as f:
    model = {r["frame"]: r for r in csv.DictReader(f)}
names = list(model)


def grab(src, wanted):
    """{frame name: image} using the same names main.py gives."""
    src, got = Path(src), {}
    if src.is_dir():
        for p in src.iterdir():
            if p.suffix.lower() in IMG and p.stem in wanted:
                got[p.stem] = cv2.imread(str(p))
        return got
    cap = cv2.VideoCapture(str(src)); i = 0
    last = max(int(w[1:]) for w in wanted)
    while i <= last:
        ok, img = cap.read()
        if not ok:
            break
        if f"f{i:05d}" in wanted:
            got[f"f{i:05d}"] = img
        i += 1
    return got


if a.cmd == "sheet":
    if not a.source:
        sys.exit("sheet needs --source")
    n = min(a.n, len(names))
    pick = [names[round(i * (len(names) - 1) / max(1, n - 1))] for i in range(n)]
    pick = list(dict.fromkeys(pick))
    imgs = grab(a.source, set(pick))
    tiles = []
    for k in pick:
        cv2.imwrite(str(hd / f"{k}.jpg"), imgs[k])
        t = cv2.resize(imgs[k], (640, 360))
        cv2.putText(t, k, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(t, k, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(hd / "sheet.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
    csvp = hd / "handcount.csv"
    if csvp.exists():
        sys.exit(f"{csvp} already exists, not overwriting your counts. Frames re-saved.")
    with open(csvp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "person_hand", "vehicle_hand", "counted_by"])
        w.writerows([k, "", "", ""] for k in pick)
    print(f"{len(pick)} clean frames in {hd}/. Fill {csvp} (open the .jpg files, not the *_det.jpg ones), then:\n"
          f"  python tools/handcount.py table --run {a.run}")
else:
    with open(hd / "handcount.csv") as f:
        hand = [r for r in csv.DictReader(f) if r["person_hand"].strip() != "" and r["vehicle_hand"].strip() != ""]
    if not hand:
        sys.exit("handcount.csv has no filled rows yet")
    out = ["| Frame | People (human) | People (model) | Diff | Vehicles (human) | Vehicles (model) | Diff |",
           "|---|---|---|---|---|---|---|"]
    tp = tv = hp = hv = ep = ev = 0; within = 0
    for r in hand:
        m = model[r["frame"]]
        ph, vh = int(r["person_hand"]), int(r["vehicle_hand"])
        pm = int(m["person_tiled"]) if m.get("person_tiled") else int(m["person"])
        vm = int(m["vehicle"])
        out.append(f"| {r['frame']} | {ph} | {pm} | {pm - ph:+d} | {vh} | {vm} | {vm - vh:+d} |")
        hp += ph; hv += vh; tp += pm; tv += vm; ep += abs(pm - ph); ev += abs(vm - vh)
        within += abs(pm - ph) <= 2 and abs(vm - vh) <= 2
    n = len(hand)
    out.append(f"| **Total** | **{hp}** | **{tp}** | {tp - hp:+d} | **{hv}** | **{tv}** | {tv - hv:+d} |")
    who = sorted({r["counted_by"].strip() for r in hand if r.get("counted_by", "").strip()})
    out += ["",
            f"- Frames hand-counted: **{n}**" + (f" (by {', '.join(who)})" if who else ""),
            f"- Mean absolute error per frame: people **{ep / n:.1f}**, vehicles **{ev / n:.1f}**",
            f"- Share of the human count found: people **{100 * tp / max(1, hp):.0f}%**, vehicles **{100 * tv / max(1, hv):.0f}%**",
            f"- Frames where both counts are within ±2 of the human: **{within} of {n}**",
            f"- Model run: `{a.run}`. This is {n} frames from one clip, not a benchmark."]
    text = "\n".join(out) + "\n"
    (hd / "accuracy.md").write_text(text)
    print(text)
