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
story = next((a.split("=")[1] for a in sys.argv if a.startswith("--story=")), None)  # gate | junction

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
if s["profile"] != "accurate" and not force:
    problems.append(f"deliverables come from profile '{s['profile']}', but submission.md says the counts are from the accurate profile")
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
                          f["mean"], f["med"], f["p95"], f"{1000 / float(f['mean']):.2f}", f["fps"], f["ram"]]   # profiles.md has end-to-end FPS only
                         + ([a["mean"]] if a else []) if f else [])
t = t.replace("[Pi model, RAM, OS]", board)
if story in ("gate", "junction"):   # keep this story's sentence, drop the other one
    t = re.sub(r"\[(gate|junction): ([^\]]*)\](?:\s*/\s*)?", lambda m: m.group(2) if m.group(1) == story else "", t)
print("submission.md remaining blanks:", re.findall(r"\[[^\]]*\]", t))
if write: sub.write_text(t)
print(("WROTE" if write else "dry run:"), "submission.md; frames", s["frames"], "profile", s["profile"])
print("pitch.md: fill by hand from the same files (script differs per story); blanks left:",
      len(re.findall(r"\[N\]|\[Pi model\]|\[10\]", (R / "docs/pitch.md").read_text())))


# ---- pitch.md: needs --story=gate|junction (a human decision after seeing the clip) ----
if story not in ("gate", "junction"):
    print("pitch.md untouched: pass --story=gate or --story=junction to trim, resolve and fill it")
    sys.exit(0)
rowsc = [l.split(",") for l in (D / "counts.csv").read_text().splitlines()[1:] if l]
col = lambda n: [x[cols[n]] for x in rowsc]
n = len(rowsc)
busy = round(100 * sum(v in ("BUSY", "JAMMED") for v in col("traffic_level")) / n)
ped = max(int(v) for v in col("ped_on_road"))
avgv = round(sum(int(v) for v in col("vehicle")) / n, 1)
delta = re.search(r"differs by ([+-]?[\d.]+) people and ([+-]?[\d.]+) vehicles", prof)
dp, dv = (delta.group(1), delta.group(2)) if delta else ("[N]", "[N]")
size = f"{s['size_mb']:.1f}" if s["profile"] == "fast" else (f["model"] and "4.7")
pt = (R / "docs/pitch.md").read_text()
head, rest = pt.split("## Script A", 1)
A, rest2 = rest.split("## Script B", 1)
B, tail = rest2.split("## Questions to expect", 1)
A = "## Script A" + A; B = "## Script B" + B
body = A if story == "gate" else B
body = re.sub(r"## Script [AB]: [^\n]*\(.*?\)", "## Script", body, count=1)
have_acc = (R / "handcount/accuracy.md").exists()
if not have_acc:   # no hand count: drop every accuracy claim instead of leaving blanks
    body = re.sub(r"\*\*1:00 Can you trust it\.\*\*.*?(?=\*\*1:20)", "", body, flags=re.S)
    tail = re.sub(r"\*\*Accuracy:\*\*.*?\n\n", "", tail, flags=re.S)
pick = 0 if story == "gate" else 1
tail = re.sub(r"\[(gate|junction): ([^\]]*)\](?: / \[(?:gate|junction): [^\]]*\])?",
              lambda m: m.group(2) if m.group(1) == ("gate" if pick == 0 else "junction") else m.group(0), tail)
if story == "junction":
    tail = re.sub(r"\[gate: [^\]]*\] / \[junction: ([^\]]*)\]", r"\1", tail)
nums = ([str(peak["person"]), str(ped), str(busy)] if story == "gate"
        else [str(peak["vehicle"]), str(avgv), str(busy)])
nums += [f["mean"], f["fps"], size, f["ram"], dp if story == "gate" else dv, a["mean"] if a else "[N]"] if f else []
body = sub_once(body, r"\[N\]", nums).replace("[Pi model]", board)
tail = tail.replace("Nano gives [N] FPS", f"Nano gives {f['fps']} FPS").replace("[fill from benchmark]", "see deliverables/benchmark.md")
out = head.split("\n")[0].replace("Pick ONE script after seeing the clip.", f"{story} story.") + "\n\n" + body.strip() + "\n\n## Questions to expect" + tail
left = re.findall(r"\[[^\]]*\]", out)
print("pitch.md remaining blanks:", left)
if write: (R / "docs/pitch.md").write_text(out)
print(("WROTE" if write else "dry run:"), "pitch.md for", story, "| accuracy lines", "kept" if have_acc else "removed (no handcount/accuracy.md)")
