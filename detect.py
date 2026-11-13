import json, time, sys, statistics
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

KEEP = {0:"person",1:"bicycle",2:"car",3:"motorcycle",5:"bus",7:"truck"}
COL = {"person":(255,196,0),"car":(0,200,255),"motorcycle":(255,80,80),"bus":(140,255,120),"truck":(200,140,255),"bicycle":(255,140,220)}
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
model = YOLO("models/yolo11n_executorch_model", task="detect")
picks = sys.argv[1:]
out = {}
for name in picks:
    src = Path("img/raw")/name
    model.predict(str(src), imgsz=640, verbose=False)  # warm-up
    ts = []
    for _ in range(10):
        r = model.predict(str(src), imgsz=640, conf=0.30, verbose=False)[0]
        ts.append(r.speed["inference"])
    im = Image.open(src).convert("RGB"); W,H = im.size
    d = ImageDraw.Draw(im); lw = max(2, W//500)
    f = ImageFont.truetype(FONT, max(14, W//75))
    counts = {}
    for b in r.boxes:
        c = int(b.cls); 
        if c not in KEEP: continue
        lab = KEEP[c]; counts[lab] = counts.get(lab,0)+1
        x1,y1,x2,y2 = map(float,b.xyxy[0]); col = COL[lab]
        d.rectangle([x1,y1,x2,y2], outline=col, width=lw)
        t = f"{lab} {float(b.conf):.2f}"
        tb = d.textbbox((0,0),t,font=f); tw,th = tb[2]-tb[0], tb[3]-tb[1]
        ty = y1-th-6 if y1-th-6>0 else y1
        d.rectangle([x1,ty,x1+tw+8,ty+th+6], fill=col)
        d.text((x1+4,ty+1), t, fill=(0,0,0), font=f)
    im.save(f"img/out/{src.stem}_det.jpg", quality=90)
    out[name] = {"counts":counts, "total":sum(counts.values()), "ms_mean":statistics.mean(ts), "ms_med":statistics.median(ts),
                 "pre":r.speed["preprocess"], "post":r.speed["postprocess"], "size":[W,H]}
    print(name, counts, f"{statistics.median(ts):.1f} ms")
json.dump(out, open("img/out/results.json","w"), indent=1)
