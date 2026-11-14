# Sweep (PROXY reference = yolo11s_fp32_640 + tiling @ 0.35/0.35, NMS 0.5 grouped (NOT human truth))

16 tune / 8 check frames. Settings chosen on tune; the numbers below are on check. Ties go to the higher threshold.

| model | variant | grouped | iou | min_h | conf_person | conf_vehicle | MAE people | MAE vehicles | bias p/v | within ±2 (p/v) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n_fp32_480x864 | full | True | 0.5 | 0.0 | 0.15 | 0.15 | 1.00 | 0.88 | +1.00/+0.88 | 7/8 , 7/8 |
| yolo11n_fp32_480x864 | tiled | False | 0.5 | 0.0 | 0.35 | 0.25 | 0.50 | 0.38 | +0.50/+0.38 | 8/8 , 8/8 |
| yolo11n_raspberry_executorch_optimized | full | True | 0.6 | 0.0 | 0.15 | 0.15 | 0.50 | 0.12 | -0.50/+0.12 | 7/8 , 8/8 |
| yolo11n_raspberry_executorch_optimized | tiled | False | 0.45 | 0.0 | 0.3 | 0.2 | 0.62 | 0.88 | +0.62/+0.88 | 8/8 , 6/8 |
| yolo11n_raspberry_executorch_original | full | True | 0.6 | 0.0 | 0.15 | 0.15 | 0.62 | 0.12 | -0.38/+0.12 | 7/8 , 8/8 |
| yolo11n_raspberry_executorch_original | tiled | False | 0.45 | 0.04 | 0.25 | 0.15 | 0.88 | 0.62 | -0.12/-0.12 | 7/8 , 7/8 |
| yolo11s_fp32_640 | full | True | 0.45 | 0.04 | 0.15 | 0.15 | 0.50 | 0.75 | -0.25/-0.75 | 8/8 , 8/8 |

## Flags for main.py from the winners above (PROXY: do not adopt)

- `yolo11n_fp32_480x864` full: `--model bundle/models/yolo11n_fp32_480x864.pte --conf-person 0.15 --conf-vehicle 0.15 --iou 0.5 --min-h 0.0`
- `yolo11n_fp32_480x864` tiled: `--model bundle/models/yolo11n_fp32_480x864.pte --conf-person 0.35 --conf-vehicle 0.25 --iou 0.5 --min-h 0.0 --ungrouped --tile all`
- `yolo11n_raspberry_executorch_optimized` full: `--model bundle/models/yolo11n_raspberry_executorch_optimized.pte --conf-person 0.15 --conf-vehicle 0.15 --iou 0.6 --min-h 0.0`
- `yolo11n_raspberry_executorch_optimized` tiled: `--model bundle/models/yolo11n_raspberry_executorch_optimized.pte --conf-person 0.3 --conf-vehicle 0.2 --iou 0.45 --min-h 0.0 --ungrouped --tile all`
- `yolo11n_raspberry_executorch_original` full: `--model bundle/models/yolo11n_raspberry_executorch_original.pte --conf-person 0.15 --conf-vehicle 0.15 --iou 0.6 --min-h 0.0`
- `yolo11n_raspberry_executorch_original` tiled: `--model bundle/models/yolo11n_raspberry_executorch_original.pte --conf-person 0.25 --conf-vehicle 0.15 --iou 0.45 --min-h 0.04 --ungrouped --tile all`
- `yolo11s_fp32_640` full: `--model bundle/models/yolo11s_fp32_640.pte --conf-person 0.15 --conf-vehicle 0.15 --iou 0.45 --min-h 0.04`

## Legacy setting (per-class NMS, IoU 0.7, conf 0.30/0.30, min_h 0), same check frames

| model | MAE people | MAE vehicles | bias p/v |
|---|---|---|---|
| yolo11n_fp32_480x864 | 1.00 | 0.62 | -0.75/-0.38 |
| yolo11n_raspberry_executorch_optimized | 1.12 | 0.38 | -1.12/-0.38 |
| yolo11n_raspberry_executorch_original | 1.12 | 0.25 | -1.12/-0.25 |
| yolo11s_fp32_640 | 0.62 | 0.62 | -0.62/-0.62 |

> Proxy mode: a model's error against a reference it did not make says how far a cheap setting is from the expensive one, not how far it is from reality. Replace with devset/truth.csv.
