# Edge people & vehicle counter (Namma AI Hackathon, PS2)

Counts people and vehicles in a clip or a folder of images, fully offline, on a Raspberry Pi with
an ExecuTorch YOLO11n `.pte`. It then turns the counts into one page a ward engineer can act on.
Counts only: no tracking, no identities, no cloud.

## Run it on the Pi

```bash
bash bundle/install_pi.sh                          # offline for Python 3.11 / 3.12 / 3.13
bash run_all.sh <clip.mp4 | image folder> gate     # or: junction.  Add --every 3, --zones zones.json
```

`run_all.sh` runs every model in `bundle/models/` on the same frames and writes:

| Output | What |
|---|---|
| `out/<model>/counts.csv` | per-frame counts: person, car, bus, truck, motorcycle, bicycle, merged `vehicle`, on-road pedestrians, traffic and crowd level, inference ms |
| `out/<model>/*_det.jpg` | annotated frames, heads blurred |
| `out/benchmark.md` | all models side by side: ms (mean, median, p95), FPS, size, RAM, count differences |
| `out/<model>/report.html` (+ `.pdf`) | the one-page site report |
| `deliverables/` | safe copy: CSV, two sample frames, benchmark, report |

Other tools:

```bash
python tools/zones.py --source clip.mp4            # click the road corners in a browser -> zones.json
python tools/handcount.py sheet --source clip.mp4 --run out/<model>    # 10 clean frames for a human to count
python tools/handcount.py table --run out/<model>                      # model vs human table
python tools/live.py                               # phone takes a photo, the Pi counts it (LAN only)
python stamp.py verify out/<model>                 # re-check the CSV against its hash chain
python main.py --source clip.mp4 --deploy          # CSV only: no frames are written at all
```

## Raspberry Pi measurements
<!-- PI_TABLE_START -->
_Filled by `run_all.sh` on the Pi._
<!-- PI_TABLE_END -->

## Accuracy against a human count
<!-- ACCURACY_START -->
_Filled on the day from `handcount/accuracy.md`._
<!-- ACCURACY_END -->

## Models

| File | What | Source |
|---|---|---|
| `yolo11n_raspberry_executorch_optimized.pte` | YOLO11n INT8, 640 input, 4.7 MB | Arm (`Arm/yolo11n-int8-xnnpack-executorch-raspberrypi5`) |
| `yolo11n_raspberry_executorch_original.pte` | YOLO11n FP32, 640 input, 10.2 MB | Arm, same export |
| `yolo11n_fp32_480.pte`, `yolo11n_fp32_320.pte` | YOLO11n FP32 at smaller inputs | our export with Ultralytics, ExecuTorch/XNNPACK |

Pretrained COCO weights, no training by us. Runtime: ExecuTorch 1.1.0, XNNPACK backend, CPU only.
`runner.py` is a thin runner (no Ultralytics on the Pi) and reads the input size from the `.pte`.

## How the add-ons work
- **Road zone:** a polygon drawn once per camera view (`zones.json`).
- **Traffic level:** share of the road zone covered by vehicle boxes: FREE < 25%, BUSY < 50%, JAMMED above.
- **Pedestrians on the road:** people whose foot point is inside the zone, excluding riders (a person box mostly inside a two-wheeler box).
- **Privacy:** the top quarter of every person box is blurred in saved frames; `--deploy` saves no frames.
- **Integrity:** every CSV row is hash-chained with the model and input hashes (`stamp.json`).

## Limits
- Counts are per frame, not unique people or vehicles.
- COCO has no auto-rickshaw class: autos appear as car or truck, so the merged `vehicle` count is the one to trust.
- The traffic level is box area over a road zone: a relative index, not true density.
- The zone and the level cut-offs are set by hand and need calibrating per site.
- Small or half-hidden people are missed; `--tile auto` helps at 4x the inference time.

Models are AGPL-3.0 (Ultralytics / Arm). Sample photos: Wikimedia Commons, credits in `img/raw/meta.json`.
