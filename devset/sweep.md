# Sweep (PROXY reference = yolo11m_fp32_640 + tiling @ 0.35/0.35, NMS 0.5 grouped (NOT human truth))

16 tune / 8 check frames. Settings chosen on tune; the numbers below are on check. Ties go to the higher threshold.

| model | variant | grouped | iou | min_h | conf_person | conf_vehicle | MAE people | MAE vehicles | bias p/v | within ±2 (p/v) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11m_fp32_640 | full | True | 0.45 | 0.02 | 0.15 | 0.15 | 0.75 | 0.38 | +0.00/-0.12 | 7/8 , 8/8 |
| yolo11n_raspberry_executorch_original | full | True | 0.6 | 0.0 | 0.15 | 0.15 | 0.38 | 0.50 | -0.38/+0.25 | 8/8 , 8/8 |
| yolo11n_raspberry_executorch_original | tiled | True | 0.5 | 0.0 | 0.3 | 0.2 | 0.50 | 0.75 | +0.50/+0.75 | 8/8 , 7/8 |

## Flags for main.py from the winners above (PROXY: do not adopt)

- `yolo11m_fp32_640` full: `--model bundle/models/yolo11m_fp32_640.pte --conf-person 0.15 --conf-vehicle 0.15 --iou 0.45 --min-h 0.02`
- `yolo11n_raspberry_executorch_original` full: `--model bundle/models/yolo11n_raspberry_executorch_original.pte --conf-person 0.15 --conf-vehicle 0.15 --iou 0.6 --min-h 0.0`
- `yolo11n_raspberry_executorch_original` tiled: `--model bundle/models/yolo11n_raspberry_executorch_original.pte --conf-person 0.3 --conf-vehicle 0.2 --iou 0.5 --min-h 0.0 --tile all`

## Legacy setting (per-class NMS, IoU 0.7, conf 0.30/0.30, min_h 0), same check frames

| model | MAE people | MAE vehicles | bias p/v |
|---|---|---|---|
| yolo11m_fp32_640 | 1.00 | 0.50 | -1.00/-0.50 |
| yolo11n_raspberry_executorch_original | 1.12 | 0.38 | -1.12/-0.12 |

> Proxy mode: a model's error against a reference it did not make says how far a cheap setting is from the expensive one, not how far it is from reality. Replace with devset/truth.csv.
