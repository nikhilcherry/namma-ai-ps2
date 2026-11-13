"""Fast vs accurate profile: FPS, time per frame, and how many people / vehicles the fast one loses.
  python tools/profile_diff.py out/profile_fast out/profile_accurate
The pitch line: "we lose X people per frame to go Y times faster"."""
import csv, json, sys
from pathlib import Path


def load(d):
    d = Path(d)
    rows = list(csv.DictReader(open(d / "counts.csv")))
    return json.loads((d / "summary.json").read_text()), {r["frame"]: r for r in rows}


(fs, fr), (as_, ar) = load(sys.argv[1]), load(sys.argv[2])
common = [f for f in fr if f in ar]
col = lambda r, k: int(r.get(k + "_smooth") or r[k])
dp = [col(ar[f], "person") - col(fr[f], "person") for f in common]
dv = [col(ar[f], "vehicle") - col(fr[f], "vehicle") for f in common]
n = max(1, len(common))
speedup = as_["e2e_ms"] / max(1e-9, fs["e2e_ms"])
print("| profile | model | tile | frames inferred | inference ms (mean/median/p95) | end-to-end FPS | peak RAM MB (whole process) |")
print("|---|---|---|---|---|---|---|")
for s in (fs, as_):
    print(f"| {s['profile']} | {s['model']} | {s['tile']} | {s.get('frames_inferred', s['frames'])}/{s['frames']} | "
          f"{s['mean_ms']}/{s['median_ms']}/{s['p95_ms']} | {s['fps_e2e']} | {s['peak_ram_mb']} |")
print(f"\nFast is {speedup:.1f}x faster end to end. Against the accurate profile on {len(common)} common frames it "
      f"differs by {sum(dp)/n:+.1f} people and {sum(dv)/n:+.1f} vehicles per frame "
      f"(mean absolute: {sum(map(abs, dp))/n:.1f} / {sum(map(abs, dv))/n:.1f}).")
print("Accurate profile counts are not ground truth: compare both with tools/handcount.py.")
