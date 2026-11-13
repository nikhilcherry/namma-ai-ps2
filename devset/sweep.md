# Sweep (PROXY reference = yolo11m_fp32_640 + tiling @ 0.35/0.35, NMS 0.5 grouped (NOT human truth))

20 tune / 10 check frames. Settings chosen on tune; the numbers below are on check. Ties go to the higher threshold.

| model | variant | grouped | iou | min_h | conf_person | conf_vehicle | MAE people | MAE vehicles | bias p/v | within ±2 (p/v) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11m_fp32_640 | full | True | 0.45 | 0.02 | 0.15 | 0.15 | 0.60 | 0.50 | +0.00/+0.10 | 9/10 , 10/10 |
| yolo11n_fp32_320 | full | True | 0.7 | 0.0 | 0.15 | 0.15 | 2.20 | 1.70 | -1.00/-1.30 | 7/10 , 8/10 |
| yolo11n_fp32_320 | tiled | False | 0.45 | 0.0 | 0.15 | 0.15 | 0.90 | 1.30 | -0.30/+0.10 | 8/10 , 8/10 |
| yolo11n_fp32_480 | full | True | 0.7 | 0.0 | 0.15 | 0.15 | 1.10 | 0.80 | -0.30/-0.20 | 8/10 , 9/10 |
| yolo11n_fp32_480 | tiled | False | 0.6 | 0.0 | 0.2 | 0.15 | 1.10 | 1.10 | +0.70/+1.10 | 8/10 , 8/10 |
| yolo11n_raspberry_executorch_optimized | full | True | 0.6 | 0.0 | 0.15 | 0.15 | 0.50 | 1.10 | -0.50/+0.50 | 10/10 , 9/10 |
| yolo11n_raspberry_executorch_optimized | tiled | False | 0.45 | 0.0 | 0.3 | 0.15 | 0.60 | 1.40 | +0.40/+1.40 | 10/10 , 8/10 |
| yolo11n_raspberry_executorch_original | full | True | 0.7 | 0.0 | 0.15 | 0.15 | 0.90 | 1.20 | +0.50/+0.60 | 9/10 , 9/10 |
| yolo11n_raspberry_executorch_original | tiled | True | 0.5 | 0.0 | 0.3 | 0.2 | 0.50 | 0.60 | +0.30/+0.60 | 10/10 , 9/10 |
| yolo11s_fp32_640 | full | True | 0.6 | 0.02 | 0.2 | 0.2 | 0.70 | 0.60 | +0.30/+0.00 | 9/10 , 8/10 |
| yolo11s_fp32_640 | tiled | True | 0.45 | 0.02 | 0.4 | 0.2 | 0.50 | 1.40 | -0.30/+1.40 | 10/10 , 7/10 |

## Legacy setting (per-class NMS, IoU 0.7, conf 0.30/0.30, min_h 0), same check frames

| model | MAE people | MAE vehicles | bias p/v |
|---|---|---|---|
| yolo11m_fp32_640 | 1.10 | 0.90 | -1.10/-0.90 |
| yolo11n_fp32_320 | 2.40 | 2.80 | -2.40/-2.80 |
| yolo11n_fp32_480 | 1.80 | 1.80 | -1.80/-1.80 |
| yolo11n_raspberry_executorch_optimized | 1.80 | 1.30 | -1.80/-1.10 |
| yolo11n_raspberry_executorch_original | 2.00 | 1.10 | -2.00/-0.90 |
| yolo11s_fp32_640 | 0.60 | 0.80 | -0.60/-0.80 |

> Proxy mode: a model's error against a reference it did not make says how far a cheap setting is from the expensive one, not how far it is from reality. Replace with devset/truth.csv.
