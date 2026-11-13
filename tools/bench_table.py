"""Benchmark table from every run under out/<model>/ (written by run_all.sh).

  python tools/bench_table.py out/ [--ref yolo11n_raspberry_executorch_original] [--fill-readme]

Writes out/benchmark.md. Count columns compare each model with the reference model on the
same frames: mean absolute difference per frame (people, vehicles).
"""
import argparse, csv, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ref", default="yolo11n_raspberry_executorch_original")
ap.add_argument("--fill-readme", action="store_true")
a = ap.parse_args()
root = Path(a.out)


def counts(d):
    with open(d / "counts.csv") as f:
        return {r["frame"]: (int(r["person"]), int(r["vehicle"])) for r in csv.DictReader(f)}


runs = sorted(p.parent for p in root.glob("*/summary.json") if not p.parent.name.startswith("profile_"))   # profiles go in profiles.md
if not runs:
    raise SystemExit(f"no runs found under {root}/")
ref = counts(root / a.ref) if (root / a.ref / "counts.csv").exists() else None
board = json.loads((runs[0] / "summary.json").read_text())["board"]
head = ("| Model | Input | Mean ms | Median ms | p95 ms | FPS (infer) | FPS (end to end) | Model MB | Peak RAM MB "
        "| People total | Vehicles total | Δ people/frame | Δ vehicles/frame |")
lines = [f"Board: **{board}** · frames per run: {json.loads((runs[0] / 'summary.json').read_text())['frames']}",
         "", head, "|" + "---|" * 13]
for d in runs:
    s = json.loads((d / "summary.json").read_text())
    dp = dv = "ref" if d.name == a.ref else "-"
    if ref and d.name != a.ref:
        c = counts(d); common = [k for k in c if k in ref]
        if common:
            dp = f"{sum(abs(c[k][0] - ref[k][0]) for k in common) / len(common):.2f}"
            dv = f"{sum(abs(c[k][1] - ref[k][1]) for k in common) / len(common):.2f}"
    lines.append(f"| {d.name} | {s['input']} | {s['mean_ms']} | {s['median_ms']} | {s['p95_ms']} | {s['fps_infer']} "
                 f"| {s['fps_e2e']} | {s['size_mb']} | {s['peak_ram_mb']} | {s['people_total']} | {s['vehicle_total']} "
                 f"| {dp} | {dv} |")
lines += ["", f"Δ = mean absolute count difference per frame against `{a.ref}` (FP32, 640).",
          "End to end = pre-processing + inference + post-processing + add-ons, without saving images."]
text = "\n".join(lines) + "\n"
(root / "benchmark.md").write_text(text)
print(text)
if a.fill_readme:
    rd = Path("README.md"); s = rd.read_text()
    new = re.sub(r"(<!-- PI_TABLE_START -->).*?(<!-- PI_TABLE_END -->)", lambda m: m.group(1) + "\n" + text + m.group(2),
                 s, flags=re.S)
    if new != s:
        rd.write_text(new); print("README.md table filled")
    else:
        print("README.md: markers not found, table not filled")
