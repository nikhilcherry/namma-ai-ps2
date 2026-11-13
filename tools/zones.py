"""Draw the road zone by clicking on the first frame, in a browser (works headless over the LAN).

  python tools/zones.py --source clip.mp4 [--out zones.json] [--port 18081]
  -> open http://<pi-ip>:18081, click the corners of the road in order, press Save.

Or skip the browser:  python tools/zones.py --source clip.mp4 --grid   writes zone_grid.jpg with a
10% grid to read the corners from, then write zones.json by hand: {"road": [[x,y],...]} in 0..1.
"""
import argparse, json, socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2

IMG = {".jpg", ".jpeg", ".png", ".bmp"}
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--out", default="zones.json")
ap.add_argument("--port", type=int, default=18081)
ap.add_argument("--grid", action="store_true")
a = ap.parse_args()
src = Path(a.source)
if src.is_dir():
    img = cv2.imread(str(sorted(p for p in src.iterdir() if p.suffix.lower() in IMG)[0]))
else:
    ok, img = cv2.VideoCapture(str(src)).read()
    if not ok:
        raise SystemExit(f"cannot read {src}")
h, w = img.shape[:2]
if a.grid:
    for i in range(1, 10):
        x, y = w * i // 10, h * i // 10
        cv2.line(img, (x, 0), (x, h), (0, 255, 255), 1); cv2.line(img, (0, y), (w, y), (0, 255, 255), 1)
        cv2.putText(img, f".{i}", (x + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f".{i}", (3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite("zone_grid.jpg", img); raise SystemExit("wrote zone_grid.jpg")
JPG = cv2.imencode(".jpg", img)[1].tobytes()
PAGE = b"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Road zone</title><style>body{font:16px system-ui;margin:12px;background:#111;color:#eee}
#w{position:relative;display:inline-block;max-width:100%}img{max-width:100%;display:block}svg{position:absolute;inset:0;width:100%;height:100%}
button{font-size:18px;padding:10px 18px;margin:8px 8px 8px 0}</style>
<p>Click the corners of the road, in order. <span id=s></span></p>
<div id=w><img id=i src=/frame.jpg><svg id=v viewBox="0 0 1 1" preserveAspectRatio=none><polygon id=p fill="rgba(255,200,0,.25)" stroke="#fc0" stroke-width=".004"/></svg></div>
<div><button onclick="pts.pop();draw()">Undo</button><button onclick="pts=[];draw()">Clear</button><button onclick=save()>Save</button></div>
<script>let pts=[];const draw=()=>{p.setAttribute('points',pts.map(q=>q.join(',')).join(' '));s.textContent=pts.length+' points'};
w.onclick=e=>{const r=i.getBoundingClientRect();pts.push([+((e.clientX-r.left)/r.width).toFixed(3),+((e.clientY-r.top)/r.height).toFixed(3)]);draw()};
const save=async()=>{if(pts.length<3){s.textContent='need at least 3 points';return}
 const r=await fetch('/save',{method:'POST',body:JSON.stringify({road:pts})});s.textContent=await r.text()};</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        body, ct = (JPG, "image/jpeg") if self.path.startswith("/frame") else (PAGE, "text/html; charset=utf-8")
        self.send_response(200); self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        z = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        road = [[float(x), float(y)] for x, y in z["road"]]
        Path(a.out).write_text(json.dumps({"road": road}))
        msg = f"saved {a.out} with {len(road)} points. Stop this server (Ctrl+C) and re-run with --zones {a.out}"
        print(msg); body = msg.encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("10.255.255.255", 1)); ip = s.getsockname()[0]
except OSError:
    ip = "127.0.0.1"
print(f"zone picker on http://{ip}:{a.port}  (frame {w}x{h})")
HTTPServer(("0.0.0.0", a.port), H).serve_forever()
