# Sweep (PROXY reference = yolo11m_fp32_640 + tiling @ 0.35/0.35, NMS 0.5 grouped (NOT human truth))

16 tune / 8 check frames. Settings chosen on tune; the numbers below are on check. Ties go to the higher threshold.

| model | variant | grouped | iou | min_h | conf_person | conf_vehicle | MAE people | MAE vehicles | bias p/v | within ±2 (p/v) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11m_fp32_640 | full | True | 0.45 | 0.02 | 0.15 | 0.15 | 0.75 | 0.38 | +0.00/-0.12 | 7/8 , 8/8 |
| yolo11n_fp32_480 | full | True | 0.7 | 0.0 | 0.15 | 0.15 | 0.38 | 0.75 | -0.38/-0.25 | 8/8 , 7/8 |
| yolo11n_fp32_480 | tiled | False | 0.6 | 0.0 | 0.25 | 0.15 | 0.38 | 1.25 | -0.12/+1.25 | 8/8 , 6/8 |
| yolo11n_int8_480 | full | False | 0.45 | 0.0 | 0.5 | 0.5 | 2.25 | 1.25 | -2.25/-1.25 | 7/8 , 6/8 |
| yolo11n_int8_480 | tiled | False | 0.45 | 0.0 | 0.5 | 0.5 | 2.25 | 1.25 | -2.25/-1.25 | 7/8 , 6/8 |
| yolo11n_int8h_480 | full | True | 0.7 | 0.0 | 0.15 | 0.15 | 0.62 | 0.75 | +0.12/-0.25 | 8/8 , 7/8 |
| yolo11n_int8h_480 | tiled | True | 0.5 | 0.0 | 0.2 | 0.15 | 0.88 | 1.25 | +0.62/+1.25 | 7/8 , 6/8 |

## Legacy setting (per-class NMS, IoU 0.7, conf 0.30/0.30, min_h 0), same check frames

| model | MAE people | MAE vehicles | bias p/v |
|---|---|---|---|
| yolo11m_fp32_640 | 1.00 | 0.50 | -1.00/-0.50 |
| yolo11n_fp32_480 | 1.00 | 0.75 | -1.00/-0.75 |
| yolo11n_int8_480 | 2.25 | 1.25 | -2.25/-1.25 |
| yolo11n_int8h_480 | 1.12 | 0.62 | -1.12/-0.38 |

> Proxy mode: a model's error against a reference it did not make says how far a cheap setting is from the expensive one, not how far it is from reality. Replace with devset/truth.csv.
