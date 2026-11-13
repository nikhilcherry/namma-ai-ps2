# Edge people & vehicle counter (Namma AI Hackathon, PS2)

Offline counting of people and vehicles on a Raspberry Pi with an ExecuTorch YOLO11n `.pte`.
Counts only: no tracking, no cloud, no live camera.

```
python main.py --source <image folder | clip.mp4> [--zones zones.json] [--blur] [--deploy] [--tile auto]
python stamp.py verify out/        # re-check the CSV against the hash chain
```

Outputs in `out/`: `counts.csv`, annotated frames, `summary.txt`, `stamp.json`.

## Raspberry Pi measurements (fill on the day)
| Pi board | Model | Mean ms | Median ms | p95 ms | FPS (infer) | Model MB | Peak RAM MB |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Limits
- COCO has no auto-rickshaw class: they appear as car/truck, so we report one merged `vehicle` count.
- Congestion score is box area over a road zone: a relative index, not true density.
- Level cut-offs were set by hand on sample photos and need calibrating per site.
