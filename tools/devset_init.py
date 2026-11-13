"""T1 helper: dedupe devset/frames, write the blank devset/truth.csv (rule on top, fixed tune/check split),
and make numbered clean contact sheets so one person can count without seeing any model output.

  python tools/devset_init.py        then fill person,vehicle in devset/truth.csv, then python tools/sweep.py
Never overwrites a truth.csv that already has counts."""
import csv, hashlib, random
from pathlib import Path

import cv2
import numpy as np

D = Path(__file__).resolve().parent.parent / "devset"
RULE = ("# RULE (do not change after counting starts): count every person and every vehicle (car, bus, truck, auto, "
        "two-wheeler, bicycle) that is at least half visible and taller than 2% of the frame height. "
        "A rider is a person AND a vehicle.")

seen, dups = set(), []
for p in sorted((D / "frames").glob("*.jpg")):
    im = cv2.imread(str(p))
    h = hashlib.md5(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (16, 16)).tobytes()).hexdigest()
    if h in seen:
        dups.append(p)
    seen.add(h)
for p in dups:
    print("duplicate, removing", p.name); p.unlink()
frames = sorted((D / "frames").glob("*.jpg"))
t = D / "truth.csv"
if t.exists() and any(r.get("person", "").strip() for r in csv.DictReader(l for l in open(t) if not l.startswith("#"))):
    raise SystemExit("truth.csv already has counts, not touching it")
names = [p.stem for p in frames]
rng = random.Random(7)
check = set(rng.sample(names, max(2, len(names) // 3)))          # fixed seed: the split never moves
with open(t, "w", newline="") as f:
    f.write(RULE + "\n")
    w = csv.writer(f); w.writerow(["frame", "person", "vehicle", "split"])
    for n in names:
        w.writerow([n, "", "", "check" if n in check else "tune"])
sheets = D / "count_sheets"; sheets.mkdir(exist_ok=True)
for k in range(0, len(frames), 6):
    tiles = []
    for p in frames[k:k + 6]:
        im = cv2.resize(cv2.imread(str(p)), (640, 400))
        for col, th in ((0, 5), (255, 2)):
            cv2.putText(im, p.stem, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (col,) * 3, th)
        tiles.append(im)
    while len(tiles) % 2: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(sheets / f"sheet_{k // 6 + 1:02d}.jpg"), np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
print(f"{len(names)} frames, {len(check)} check / {len(names) - len(check)} tune -> {t}; sheets in {sheets}/")
print("Count from the full-size files in devset/frames/, the sheets are only for orientation.")
