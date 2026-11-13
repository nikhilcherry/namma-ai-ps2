"""Live photo demo: a phone on the same network takes a photo, the Pi counts it.

  python tools/live.py [--model bundle/models/...pte] [--port 18080]
  -> open http://<pi-ip>:18080 on a phone

The photo is decoded in memory and never written to disk. Heads are blurred in the reply.
Needs no internet: the page is one inline HTML file served by the Pi.
"""
import argparse, base64, json, socket, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import insights, runner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="bundle/models/yolo11n_raspberry_executorch_optimized.pte")
ap.add_argument("--port", type=int, default=18080)
ap.add_argument("--conf", type=float, default=0.30)
a = ap.parse_args()
det = runner.load(a.model)
det.detect(np.zeros((480, 640, 3), np.uint8), a.conf)   # warm-up
try:
    BOARD = Path("/proc/device-tree/model").read_text().strip("\x00\n")
except OSError:
    BOARD = socket.gethostname()

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Edge counter</title>
<style>body{font:17px system-ui,sans-serif;margin:0;padding:16px;background:#111;color:#eee;max-width:720px;margin:auto}
h1{font-size:20px;margin:0 0 4px}.s{color:#9a9a9a;font-size:14px;margin-bottom:14px}
label{display:block;background:#2e7d32;color:#fff;text-align:center;padding:18px;border-radius:10px;font-size:20px;font-weight:600}
input{display:none}#n{display:flex;gap:10px;margin:14px 0}#n div{flex:1;background:#1e1e1e;border-radius:10px;padding:12px;text-align:center}
#n b{display:block;font-size:40px}img{width:100%;border-radius:10px}#m{color:#9a9a9a;font-size:14px}</style>
<h1>People &amp; vehicle counter</h1><div class=s>Running on: __BOARD__ · model __MODEL__ · no cloud, photo not stored</div>
<label>Take a photo<input type=file accept="image/*" capture=environment id=f></label>
<div id=n><div><b id=p>–</b>people</div><div><b id=v>–</b>vehicles</div></div><div id=m></div><img id=o>
<script>
f.onchange=async()=>{const file=f.files[0];if(!file)return;m.textContent='counting on the Pi…';
 const bmp=await createImageBitmap(file);const s=Math.min(1,1280/Math.max(bmp.width,bmp.height));
 const c=document.createElement('canvas');c.width=bmp.width*s;c.height=bmp.height*s;c.getContext('2d').drawImage(bmp,0,0,c.width,c.height);
 const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',0.9));const t=performance.now();
 const r=await fetch('/count',{method:'POST',body:blob});const j=await r.json();
 p.textContent=j.person;v.textContent=j.vehicle;o.src='data:image/jpeg;base64,'+j.image;
 m.textContent='inference '+j.infer_ms+' ms on the Pi · round trip '+Math.round(performance.now()-t)+' ms · '+j.detail;f.value=''}
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        body = PAGE.replace("__BOARD__", BOARD).replace("__MODEL__", Path(a.model).name).encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        img = cv2.imdecode(np.frombuffer(self.rfile.read(n), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.send_error(400, "not an image"); return
        dets, ms = det.detect(img, a.conf)
        names = [d[0] for d in dets]
        insights.blur_heads(img, dets)
        for nm, s, (x1, y1, x2, y2) in dets:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        person = names.count("person"); veh = len(names) - person
        detail = ", ".join(f"{k} {names.count(k)}" for k in sorted(set(names))) or "nothing found"
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        body = json.dumps({"person": person, "vehicle": veh, "infer_ms": round(ms), "detail": detail,
                           "image": base64.b64encode(jpg).decode()}).encode()
        print(f"{time.strftime('%H:%M:%S')} photo {img.shape[1]}x{img.shape[0]}: person={person} vehicle={veh} {ms:.0f} ms")
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect(("10.255.255.255", 1)); ip = s.getsockname()[0]
except OSError:
    ip = "127.0.0.1"
print(f"live demo on http://{ip}:{a.port}  (phone must be on the same network; Ctrl+C to stop)")
ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever()
