# Sweep (PROXY reference = yolo11s_fp32_640 + tiling @ 0.35/0.35, NMS 0.5 grouped (NOT human truth))

20 tune / 10 check frames. Settings chosen on tune; the numbers below are on check. Ties go to the higher threshold.

| model | variant | grouped | iou | min_h | conf_person | conf_vehicle | MAE people | MAE vehicles | bias p/v | within ±2 (p/v) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n_raspberry_executorch_original | full | True | 0.6 | 0.0 | 0.15 | 0.15 | 0.70 | 0.50 | -0.10/+0.30 | 9/10 , 9/10 |
| yolo11n_raspberry_executorch_original | tiled | False | 0.45 | 0.04 | 0.25 | 0.15 | 0.70 | 0.50 | -0.10/-0.10 | 9/10 , 9/10 |
| yolo11s_fp32_640 | full | True | 0.45 | 0.04 | 0.15 | 0.15 | 0.60 | 0.90 | -0.20/-0.90 | 10/10 , 10/10 |

## Legacy setting (per-class NMS, IoU 0.7, conf 0.30/0.30, min_h 0), same check frames

| model | MAE people | MAE vehicles | bias p/v |
|---|---|---|---|
| yolo11n_raspberry_executorch_original | 1.90 | 0.90 | -1.90/-0.90 |
| yolo11s_fp32_640 | 0.50 | 0.80 | -0.50/-0.80 |

> Proxy mode: a model's error against a reference it did not make says how far a cheap setting is from the expensive one, not how far it is from reality. Replace with devset/truth.csv.
