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
python stamp.py keygen                             # once per device: creates .stamp_key (never commit); runs are then signed (HMAC)
python tools/parity.py                             # runner.py vs Arm's reference output (untested on device)
python tools/check_occupancy.py                    # low-res occupancy vs full-res (max diff 0.007 measured on random frames)
python main.py --source folder/ --repeat 10        # image folder: 10 timing passes, counts written once
# summary.json also records Pi temperature and throttle state before and after a run (untested on device)
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
| `yolo11n_fp32_960.pte`, `yolo11s_fp32_640.pte`, `yolo11m_fp32_640.pte` | larger input / larger models for the accurate profile (s and m need organiser approval before use for submitted counts) | our export with Ultralytics |

Pretrained COCO weights, no training by us. Runtime: ExecuTorch 1.1.0, XNNPACK backend, CPU only.
`runner.py` is a thin runner (no Ultralytics on the Pi) and reads the input size from the `.pte`.

## How the add-ons work
- **Road zone:** a polygon drawn once per camera view (`zones.json`).
- **Traffic level:** share of the road zone covered by vehicle boxes: FREE < 25%, BUSY < 50%, JAMMED above.
- **Pedestrians on the road:** people whose foot point is inside the zone, excluding riders (a person box mostly inside a two-wheeler box).
- **Privacy:** the top quarter of every detected person box is blurred in saved frames (default; `--no-blur` disables); `--deploy` saves no frames.
- **Integrity:** every CSV row is hash-chained with the model and input hashes (`stamp.json`). Without a key this detects accidental or partial edits only, because whoever edits the CSV can rebuild the chain. With a key (`python stamp.py keygen`, or `STAMP_KEY`), `stamp.json` carries an HMAC and `verify` fails on a rebuilt or unsigned stamp. The commit that publishes `stamp.json` head is the anchor.

## Limits
- Counts are per frame, not unique people or vehicles.
- COCO has no auto-rickshaw class: autos appear as car or truck, so the merged `vehicle` count is the one to trust.
- The traffic level is box area over a road zone: a relative index, not true density.
- The zone and the level cut-offs are set by hand and need calibrating per site.
- Small or half-hidden people are missed; `--tile auto` helps at 4x the inference time.
- Blur covers only people the model detected: a missed person is not blurred in a saved frame. Use `--deploy` (no frames saved) when that matters.

Models are AGPL-3.0 (Ultralytics / Arm). Sample photos: Wikimedia Commons, credits in `img/raw/meta.json`.

## Profiles, flags and tools added by the efficiency pass (see EFFICIENCY_PLAN.md, devset/FINDINGS.md)

| `--profile` | recipe | laptop ms/frame |
|---|---|---|
| `fast` | Arm INT8 640, no images, skips unchanged frames (`--skip-static 0.5`) | ~90 when inferring |
| `balanced` | yolo11n FP32 960, one pass, smoothing | ~110 |
| `ensemble` | yolo11n 960 + yolo11s 640 (`--also`), merged before NMS | ~225 |
| `lite` | yolo11n FP32 640, tiles only on crowded/small frames (`--tile adaptive`), smoothing | ~230 |
| `accurate` | yolo11n FP32 640, full frame + 4 tiles, all classes, smoothing | ~450 |

Only the **n** model is organiser-confirmed: `ensemble` (uses yolo11s) is for comparison until they approve s/m. The submitted-counts profile is `accurate` (or `lite`).

Post-processing flags: `--conf-person`, `--conf-vehicle`, `--iou`, `--min-h`, `--ungrouped` (old per-class NMS), optional `count` polygon in `zones.json`.
Counts per recipe are measured against a *proxy* reference until `devset/truth.csv` is filled in; nothing here is an accuracy claim.

Tools: `tools/sweep.py` (threshold/NMS sweep), `tools/hybrid_eval.py` (recipe comparison), `tools/quantize.py` (own INT8, `--fp32-head`),
`tools/bench_hygiene.py` (>=100 frames, two runs, throttle flag), `tools/pi_check.sh` (board readiness), `tools/profile_diff.py`, `tools/devset_init.py`.
Tests: `python tests/test_postprocess.py` and `python tests/test_samples_golden.py` (or `pytest tests`).
