"""Fill the numeric [N] blanks in docs/submission.md and docs/pitch.md from deliverables/.

Usage: python tools/fill_docs.py [--write]      (default: dry run, prints what it would fill)
Fills only what the files measured. Leaves [names], [10]/accuracy lines (need a hand count) and
anything it cannot match. Refuses to run on a non-Pi board or a throttled run unless --force.
"""
import json, re, sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
D = R / "deliverables"
force = "--force" in sys.argv
write = "--write" in sys.argv

s = json.loads((D / "summary.json").read_text())
prof = (D / "profiles.md").read_text()
rows = {}
for line in prof.splitlines():
    c = [x.strip() for x in line.strip("|").split("|")]
    if len(c) >= 7 and c[0] in ("fast", "accurate"):
        m, md, p95 = c[4].split("/")
        rows[c[0]] = dict(mean=m, med=md, p95=p95, fps=c[5], ram=c[6], model=c[1])
problems = []
if "Raspberry" not in s["board"] and not force: problems.append(f"board is '{s['board']}', not a Pi")
if s.get("throttled_during_run") and not force: problems.append("run was throttled")
if "yolo11n" not in s["model"] and not force: problems.append("main model is not yolo11n")
if problems:
    sys.exit("REFUSING: " + "; ".join(problems) + "  (use --force to preview anyway)")

hdr = (D / "counts.csv").read_text().splitlines()[0].split(",")
cols = {n: i for i, n in enumerate(hdr)}
peak = {k: max(int(l.split(",")[cols[k]]) for l in (D / "counts.csv").read_text().splitlines()[1:] if l)
        for k in ("person", "vehicle")}
f, a = rows.get("fast"), rows.get("accurate")
board = s["board"]

def sub_once(text, pat, vals):
    it = iter(vals)
    return re.sub(pat, lambda m: next(it, m.group(0)), text)

sub = D.parent / "docs" / "submission.md"
t = sub.read_text()
t = sub_once(t, r"\[N\]", [str(s["frames"]), str(s["people_total"]), str(s["vehicle_total"]),
                          str(peak["person"]), str(peak["vehicle"]),
                          f["mean"], f["med"], f["p95"], f["fps"], f["fps"], f["ram"]] if f else [])
t = t.replace("[Pi model, RAM, OS]", board)
print("submission.md remaining blanks:", re.findall(r"\[[^\]]*\]", t))
if write: sub.write_text(t)
print(("WROTE" if write else "dry run:"), "submission.md; frames", s["frames"], "profile", s["profile"])
if a and s["profile"] != "accurate":
    print("WARNING: submission.md says counts come from the accurate profile; summary.json used", s["profile"])
print("pitch.md: fill by hand from the same files (script differs per story); blanks left:",
      len(re.findall(r"\[N\]|\[Pi model\]|\[10\]", (R / "docs/pitch.md").read_text())))
