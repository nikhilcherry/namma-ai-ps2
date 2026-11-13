"""Check: low-resolution occupancy() stays within 0.02 of full resolution on random boxes and zones.
  python tools/check_occupancy.py      (exit 1 on failure)"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import insights

rng = np.random.default_rng(0)
W, H, worst = 1920, 1080, 0.0
for trial in range(200):
    zone = np.array([[int(x * W), int(y * H)] for x, y in [(0, rng.uniform(.2, .5)), (1, rng.uniform(.2, .5)), (1, 1), (0, 1)]], np.int32)
    zm = insights.zone_mask(zone, W, H)
    dets = []
    for _ in range(int(rng.integers(1, 25))):
        x, y = rng.uniform(0, W - 100), rng.uniform(0, H - 100)
        dets.append(("car", 0.9, (x, y, x + rng.uniform(40, 400), y + rng.uniform(40, 300))))
    worst = max(worst, abs(insights.occupancy(dets, zm) - insights.occupancy(dets, zm, exact=True)))
print(f"worst |low-res - full-res| occupancy over 200 random frames: {worst:.4f}")
sys.exit(0 if worst <= 0.02 else 1)
