"""T9: own post-training INT8 quantization of YOLO11n at a chosen input size (ExecuTorch XNNPACK, PT2E).

  .venv/bin/python tools/quantize.py --size 480 --out bundle/models/yolo11n_int8_480.pte [--fp32-head]
Calibrates on devset/frames (+ any --extra dir). Symmetric per-channel weights, per XNNPACKQuantizer.
--fp32-head leaves the final Detect module unquantized (Arm did this; without it accuracy often collapses)."""
import argparse, glob
from pathlib import Path

import cv2
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="/tmp/claude-1000/exp/yolo11n.pt")
ap.add_argument("--size", type=int, default=480)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", default="devset/frames")
ap.add_argument("--extra", default=None)
ap.add_argument("--fp32-head", action="store_true")
a = ap.parse_args()

from ultralytics import YOLO
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

S = a.size
net = YOLO(a.weights).model.eval().float()
for m in net.modules():
    if hasattr(m, "export"):
        m.export = True
net = net.fuse() if hasattr(net, "fuse") else net


class Wrap(torch.nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n
    def forward(self, x):
        y = self.n(x)
        return y[0] if isinstance(y, (tuple, list)) else y


model = Wrap(net).eval()
ex = (torch.zeros(1, 3, S, S),)
gm = torch.export.export(model, ex).module()

HEAD = [n for n, m in net.named_modules() if m.__class__.__name__ == "Detect"][0]


class Q(XNNPACKQuantizer):
    """Annotate everything, then strip the annotations from the Detect head so it stays FP32."""
    def annotate(self, model):
        model = super().annotate(model)
        if a.fp32_head:
            cut = 0
            for node in model.graph.nodes:
                stack = node.meta.get("nn_module_stack", {})
                if any(v[0].endswith(HEAD) or f".{HEAD}." in v[0] or v[0].endswith("." + HEAD) for v in stack.values()) \
                        and "quantization_annotation" in node.meta:
                    del node.meta["quantization_annotation"]; cut += 1
            print("head nodes left in FP32:", cut)
        return model


q = Q()
q.set_global(get_symmetric_quantization_config(is_per_channel=True))
gm = prepare_pt2e(gm, q)


def prep(p):
    im = cv2.imread(str(p)); h, w = im.shape[:2]; s = min(S / w, S / h)
    nw, nh = round(w * s), round(h * s)
    c = np.full((S, S, 3), 114, np.uint8); px, py = (S - nw) // 2, (S - nh) // 2
    c[py:py + nh, px:px + nw] = cv2.cvtColor(cv2.resize(im, (nw, nh)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(c).permute(2, 0, 1).unsqueeze(0).float().div(255)


files = sorted(glob.glob(f"{a.frames}/*.jpg")) + (sorted(glob.glob(f"{a.extra}/*.jpg")) if a.extra else [])
print("calibrating on", len(files), "images")
with torch.no_grad():
    for f in files:
        gm(prep(f))
gm = convert_pt2e(gm)
edge = to_edge_transform_and_lower(torch.export.export(gm, ex), partitioner=[XnnpackPartitioner()])
Path(a.out).write_bytes(edge.to_executorch().buffer)
print("wrote", a.out, Path(a.out).stat().st_size / 2**20, "MB")
