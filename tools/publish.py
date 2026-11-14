"""Publish a run as a static page on GitHub Pages, with a QR code that opens it.

  python tools/publish.py out/<run> --area banashankari-gate [--story gate] [--site "Govt school, 8:30 am"]
  -> docs/r/<area>/index.html   the site report, numbers only (no frames: the repo is public)
     docs/r/<area>/report.json  the headline numbers + the stamp head, for anyone to check against stamp.json
     docs/r/<area>/qr.png       QR code for https://<owner>.github.io/<repo>/r/<area>/
  then: git add docs/r/<area> && git commit && git push

One-time setup (repo admin): GitHub -> Settings -> Pages -> Deploy from branch: master, folder /docs.
The page changes only when you push: it is a snapshot of that run, not a live feed.
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--area", required=True, help="short id for the spot, e.g. banashankari-gate (a-z, 0-9, -)")
ap.add_argument("--story", choices=["gate", "junction"], default="gate")
ap.add_argument("--site", default="")
ap.add_argument("--base", default=None, help="page base URL (default: from the git remote, https://<owner>.github.io/<repo>)")
a = ap.parse_args()
if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", a.area):
    raise SystemExit("--area: use lowercase letters, digits and '-' only (it becomes part of the URL)")
run = Path(a.run)

base = a.base
if not base:
    remote = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"], capture_output=True, text=True).stdout
    m = re.search(r"github\.com[:/]([^/]+)/([^/.\s]+)", remote)
    if not m:
        raise SystemExit("no GitHub remote found: pass --base https://<owner>.github.io/<repo>")
    base = f"https://{m[1].lower()}.github.io/{m[2]}"
url = f"{base.rstrip('/')}/r/{a.area}/"

dest = ROOT / "docs" / "r" / a.area
subprocess.run([sys.executable, str(ROOT / "tools" / "report.py"), str(run), "--story", a.story, "--site", a.site,
                "--no-images", "--out", str(dest / "index.html")], check=True)

S = json.loads((run / "summary.json").read_text())
st = json.loads((run / "stamp.json").read_text())
W = json.loads((run / "warrant.json").read_text()) if (run / "warrant.json").exists() else None
(dest / "report.json").write_text(json.dumps({
    "area": a.area, "site": a.site, "url": url, "board": S["board"], "model": S["model"], "frames": S["frames"],
    "people_total": S["people_total"], "vehicle_total": S["vehicle_total"], "mean_ms": S["mean_ms"], "fps_infer": S["fps_infer"],
    "crossing_check": None if W is None else {"verdict": W["verdict"], **{k: W["criteria"]["i_pv2"][k] for k in ("P_per_h", "V_per_h", "PV2", "basis")},
                                              "wait_los": W["criteria"]["iii_wait"]["level_of_service"]},
    "stamp_head": st["head"], "signed": "hmac" in st}, indent=1), encoding="utf-8")

qr = cv2.QRCodeEncoder.create().encode(url)                     # OpenCV is already installed: no qrcode package needed
qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)   # quiet zone, or phones struggle to scan
cv2.imwrite(str(dest / "qr.png"), cv2.resize(qr, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST))
(ROOT / "docs" / ".nojekyll").touch()                            # serve files as they are (docs/*.md hold Liquid-like text)
print(f"page: {url}\nwrote {dest}/index.html, report.json, qr.png\nnext: git add docs/r/{a.area} docs/.nojekyll && git commit && git push")
