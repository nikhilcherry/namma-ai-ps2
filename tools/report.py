"""One-page site report from a run folder: the decision the counts lead to.

  python tools/report.py out/<model> --story gate|junction [--site "Govt school gate, 8:30 am"]
  -> out/<model>/report.html (self-contained) and report.pdf when a Chromium browser is installed

Every number comes from counts.csv / summary.json / detections.json of that run. The verdict is a
plain rule printed on the page, not a model.
"""
import argparse, base64, csv, json, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--story", choices=["gate", "junction"], default="gate")
ap.add_argument("--site", default="")
ap.add_argument("--handcount", default="handcount/accuracy.md")
ap.add_argument("--repo", default="github.com/nikhilcherry/namma-ai-ps2")
a = ap.parse_args()
run = Path(a.run)
with open(run / "counts.csv") as f:
    rows = list(csv.DictReader(f))
S = json.loads((run / "summary.json").read_text())
D = json.loads((run / "detections.json").read_text())
I = lambda r, k: int(r[k]) if r.get(k) not in ("", None) else 0
def first(r, *keys):
    """First column that is filled in (a smoothed 0 is a real 0, not a missing value)."""
    return next((int(r[k]) for k in keys if r.get(k) not in ("", None)), 0)


ppl = [first(r, "person_tiled", "person_smooth", "person") for r in rows]
veh = [first(r, "vehicle_smooth", "vehicle") for r in rows]
ped = [I(r, "ped_on_road") for r in rows]
addons = rows[0]["traffic_level"] != ""
lv = Counter(r["traffic_level"] for r in rows)
n = len(rows)
busy = (lv["BUSY"] + lv["JAMMED"]) / n
pk = max(range(n), key=lambda i: ppl[i]); pv = max(range(n), key=lambda i: veh[i]); pp = max(range(n), key=lambda i: ped[i])
mix = {k: sum(I(r, k) for r in rows) for k in ["car", "bus", "truck", "motorcycle", "bicycle"]}
tot = max(1, sum(mix.values()))
span = float(rows[-1]["t_s"]) - float(rows[0]["t_s"])
is_video = all(k.startswith("f") and k[1:].isdigit() for k in D)
where = (lambda i: f"{float(rows[i]['t_s']):.1f} s") if is_video else (lambda i: rows[i]["frame"])

if a.story == "gate":
    title = "Does this gate need a traffic marshal?"
    need = addons and max(ped) >= 5 and busy >= 0.30
    verdict = ("Marshal recommended for this window" if need else "No marshal needed on this evidence")
    if not S.get("zones"): verdict = "Provisional: " + verdict + " (road zone not calibrated)"
    rule = "Rule: recommend a marshal when 5 or more pedestrians are on the road in any frame AND traffic is BUSY or JAMMED in at least 30% of frames."
    why = (f"Peak of {max(ped)} pedestrians on the carriageway at {where(pp)}; traffic was BUSY or JAMMED in "
           f"{busy:.0%} of frames." if addons else "Add-ons were off, so there is no on-road count.")
    kpis = [("Peak people in view", max(ppl), where(pk)), ("Peak pedestrians on the road", max(ped), where(pp)),
            ("Frames BUSY or JAMMED", f"{busy:.0%}", f"{lv['JAMMED']} jammed"), ("Mean vehicles per frame", f"{sum(veh)/n:.1f}", f"peak {max(veh)}")]
    use = "A ward engineer leaves this box at a school gate for a week and reads one page: at which times, if any, people and traffic share the road."
else:
    title = "Is this junction keeping up with its traffic?"
    need = addons and busy >= 0.50
    verdict = ("Review the signal timing for this approach" if need else "Signal timing looks adequate on this evidence")
    if not S.get("zones"): verdict = "Provisional: " + verdict + " (road zone not calibrated)"
    rule = "Rule: flag the approach when traffic is BUSY or JAMMED in at least 50% of frames."
    why = (f"Traffic was BUSY or JAMMED in {busy:.0%} of frames ({lv['JAMMED']} JAMMED); peak of {max(veh)} vehicles in view at {where(pv)}."
           if addons else "Add-ons were off, so there is no congestion level.")
    kpis = [("Peak vehicles in view", max(veh), where(pv)), ("Mean vehicles per frame", f"{sum(veh)/n:.1f}", f"{n} frames"),
            ("Frames BUSY or JAMMED", f"{busy:.0%}", f"{lv['JAMMED']} jammed"), ("Peak people in view", max(ppl), where(pk))]
    use = "A traffic engineer leaves this box at an approach for a week and reads one page: when the queue fills the road, and what kind of vehicles fill it."


def b64(img, q=78, width=900):
    if img.shape[1] > width:
        img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])))
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1]).decode()


def chart(w=560, h=105):
    top = max(max(ppl), max(veh), 1); pad = 24
    def line(v, col):
        pts = " ".join(f"{pad + i * (w - 2 * pad) / max(1, n - 1):.1f},{h - pad - x * (h - 2 * pad) / top:.1f}" for i, x in enumerate(v))
        return f'<polyline fill="none" stroke="{col}" stroke-width="2" points="{pts}"/>'
    xl = "time →" if is_video else "frame →"
    return (f'<svg viewBox="0 0 {w} {h}" width="100%"><line x1="{pad}" y1="{h-pad}" x2="{w-pad}" y2="{h-pad}" stroke="#bbb"/>'
            f'<text x="2" y="{pad+4}" font-size="10" fill="#666">{top}</text><text x="2" y="{h-pad}" font-size="10" fill="#666">0</text>'
            f'{line(ppl, "#1565c0")}{line(veh, "#e65100")}'
            f'<text x="{pad}" y="{h-6}" font-size="10" fill="#1565c0">■ people</text><text x="{pad+70}" y="{h-6}" font-size="10" fill="#e65100">■ vehicles</text>'
            f'<text x="{w-pad-40}" y="{h-6}" font-size="10" fill="#666">{xl}</text></svg>')


figs = []
peak_name = rows[pp if (a.story == "gate" and addons and max(ped)) else (pk if a.story == "gate" else pv)]["frame"]
pf = run / f"{peak_name}_det.jpg"
if pf.exists():
    figs.append((b64(cv2.imread(str(pf))), f"Peak frame ({peak_name}). Heads blurred, boxes from the on-device model."))
sizes = {(d["w"], d["h"]) for d in D.values()}
if is_video and len(sizes) == 1 and pf.exists():       # one fixed view: where people stood
    (W, H), = sizes
    acc = np.zeros((H, W), np.float32)
    for d in D.values():
        for nm, _, x1, y1, x2, y2 in d["boxes"]:
            if nm == "person":
                cv2.circle(acc, (int((x1 + x2) / 2), min(H - 1, int(y2))), max(6, W // 80), 1.0, -1)
    if acc.max() > 0:
        acc = cv2.GaussianBlur(acc, (0, 0), W / 40); acc /= acc.max()
        base = (cv2.imread(str(pf)) * 0.55).astype(np.uint8)
        col = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
        m = (acc > 0.08)[..., None]
        hm = np.where(m, (base * 0.45 + col * 0.55).astype(np.uint8), base)
        cv2.imwrite(str(run / "heatmap.jpg"), hm)
        figs.append((b64(hm), "Where people stood: foot points of every person box over the clip."))

acc_html = ""
hp = Path(a.handcount)
if hp.exists():
    bullets = [l[2:].replace("**", "") for l in hp.read_text(encoding="utf-8").splitlines() if l.startswith("- ")][:4]
    acc_html = "<h2>Checked against a human count</h2><ul>" + "".join(f"<li>{b}</li>" for b in bullets) + "</ul>"
mixrow = " · ".join(f"{k} {v} ({v/tot:.0%})" for k, v in mix.items() if v)
T = S.get("track")
track_html = ""
if T:
    track_html = (f"<h2>Unique counts (tracked)</h2><p>About <b>{T['people_unique']}</b> different people and "
                  f"<b>{T['vehicles_unique']}</b> different vehicles over the clip"
                  + (f"; across the counting line: people {T['people_in']} in / {T['people_out']} out, "
                     f"vehicles {T['vehicles_in']} in / {T['vehicles_out']} out" if S.get("track_line") else "")
                  + ".</p><p class=\"n\">Boxes are linked frame to frame by overlap only: no faces or identities. "
                  "Crowds and gaps between frames split tracks, so these are upper estimates.</p>")
limit_unique = ("Tracked totals link boxes by overlap only and over-count in crowds; per-frame counts count the same person in every frame."
                if T else "Counts per frame, not unique people: the same person is counted in every frame they appear in.")
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
@page {{ size: A4; margin: 10mm 12mm; }} body {{ font: 9.5pt/1.35 system-ui, "Noto Sans", sans-serif; color: #1c1b19; max-width: 186mm; margin: auto; }}
h1 {{ font-size: 18pt; margin: 0; }} h2 {{ font-size: 11pt; margin: 7pt 0 2pt; }} .sub {{ color: #666; margin: 2pt 0 8pt; }}
.v {{ padding: 8pt 10pt; border-left: 5pt solid {'#c62828' if need else '#2e7d32'}; background: #f5f3ee; }} .v b {{ font-size: 14pt; }}
.k {{ display: flex; gap: 6pt; margin: 8pt 0; }} .k div {{ flex: 1; border: 0.5pt solid #d9d6cf; padding: 6pt; }} .k b {{ display: block; font-size: 18pt; }}
.k span {{ color: #666; font-size: 8.5pt; }} .f {{ display: flex; gap: 6pt; }} .f figure {{ flex: 1; margin: 0; }} img {{ width: 100%; }}
figcaption, .n {{ color: #666; font-size: 8.5pt; }} ul {{ margin: 2pt 0; padding-left: 14pt; }} p {{ margin: 2pt 0; }}
</style></head><body>
<h1>{title}</h1>
<div class="sub">{a.site + " · " if a.site else ""}{n} frames{f" over {span:.0f} s" if is_video else ""} · counted offline on {S['board']} · source: {Path(S['source']).name}{' · <b>COMPARISON RUN: model not organiser-confirmed, not for submission</b>' if not S.get('yolo11n_only', True) else ''}</div>
<div class="v"><b>{verdict}</b><p>{why}</p><p class="n">{rule} This is one short clip; a real decision needs a week of counts from the same spot.</p></div>
<div class="k">{''.join(f'<div><b>{v}</b>{k}<br><span>{s}</span></div>' for k, v, s in kpis)}</div>
<h2>Counts across the clip</h2>{chart()}
<p class="n">Vehicle mix (box labels, summed over frames): {mixrow or 'none'}. Auto-rickshaws have no class in this model and appear as car or truck, so the merged vehicle count is the reliable one.</p>
<div class="f">{''.join(f'<figure><img src="data:image/jpeg;base64,{b}"><figcaption>{c}</figcaption></figure>' for b, c in figs)}</div>
{track_html}
{acc_html}
<h2>How it was measured</h2>
<p>{S['model']} ({S['size_mb']} MB, input {S['input']}) on ExecuTorch/XNNPACK: {S['mean_ms']} ms mean inference (p95 {S['p95_ms']} ms),
{S['fps_infer']} FPS inference only, {S['fps_e2e']} FPS end to end, peak RAM {S['peak_ram_mb']} MB. No network used. Only counts leave the device; no faces or frames are stored in deploy mode.</p>
<h2>Limits</h2>
<ul><li>{limit_unique}</li>
<li>The road zone and the BUSY/JAMMED cut-offs are set by hand for each site.</li>
<li>Small or half-hidden people are missed; the error against a human count is stated above when measured. Head blur in saved frames covers only detected people, so a missed person is not blurred.</li></ul>
<h2>What this is for</h2><p>{use}</p>
<p class="n">Code and raw outputs: {a.repo} · results are hash-chained (stamp.json), head {json.loads((run / 'stamp.json').read_text())['head'][:16]}…</p>
</body></html>"""
(run / "report.html").write_text(html, encoding="utf-8")
print(f"wrote {run / 'report.html'}  verdict: {verdict}")
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hk_pdf
    hk_pdf.from_html(html, run / "report.pdf")
    print(f"wrote {run / 'report.pdf'}")
except Exception as e:      # no browser on the Pi: open report.html on any laptop and print to PDF
    print(f"PDF skipped ({str(e)[:80]}); open report.html in a browser and print to PDF")
