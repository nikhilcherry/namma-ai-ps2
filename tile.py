import json, time, torch, torchvision
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
model = YOLO("models/yolo11n_executorch_model", task="detect")
src="img/raw/c13.jpg"; im=Image.open(src).convert("RGB"); W,H=im.size
ov=0.2; tw,th=int(W/2*(1+ov)),int(H/2*(1+ov))
boxes=[];scores=[]; t0=time.perf_counter()
for x0 in (0,W-tw):
  for y0 in (0,H-th):
    r=model.predict(im.crop((x0,y0,x0+tw,y0+th)),imgsz=640,conf=0.30,classes=[0],verbose=False)[0]
    for b in r.boxes:
      x1,y1,x2,y2=map(float,b.xyxy[0]); boxes.append([x1+x0,y1+y0,x2+x0,y2+y0]); scores.append(float(b.conf))
B=torch.tensor(boxes);S=torch.tensor(scores);keep=torchvision.ops.nms(B,S,0.5)
dt=(time.perf_counter()-t0)*1000
d=ImageDraw.Draw(im);lw=max(2,W//500)
for i in keep.tolist(): d.rectangle(B[i].tolist(),outline=(255,196,0),width=lw)
im.save("img/out/c13_tiled.jpg",quality=90)
print(json.dumps({"tiled_person":len(keep),"tiled_ms_total":round(dt,1)}))
