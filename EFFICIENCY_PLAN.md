# Efficiency plan: the most accurate counts and the most frames per second this pipeline can give

Replaces `ACCURACY_PLAN.md` and `SPEED_PLAN.md`. The task descriptions below are the original plan;
what was actually built is in the status table here, and the measurements are in `devset/FINDINGS.md`.

| Task | Status at end of 13 Nov |
|---|---|
| T1 dev set | 24 frames prepared, **not hand-counted**. Decision: skip it; the 10 frames of the organiser's clip are the accuracy check (`docs/DAY.md` step 7) |
| T2 sweep | built (`tools/sweep.py`); only ever run against a proxy model, so defaults were not changed |
| T3 post-processing | built: grouped NMS (default), `--conf-person`, `--conf-vehicle`, `--iou`, `--min-h`, optional count polygon |
| T4 more pixels | built: tiling for all classes (`accurate`), adaptive tiling (`lite`), FP32 960 export (`balanced`) |
| T5 larger models | s and m exported; s is in the bundle, m is laptop-only (77 MB). Not approved by the organisers |
| T6 rolling median | built, on in the smoothing profiles |
| T7 less work per frame | built: `grab()` skipping, resize before colour conversion, `--profile` |
| T8 skip static frames | built, in `fast` at threshold 0.5 |
| T9 INT8 at 480 | built with an FP32 head; about equal to FP32 480 on accuracy, speed unknown; not in the default run |
| Section 4 (M1–M5) | all of it needs the Pi; commands are in `docs/DAY.md` |

## 1. What we are optimising

| Number | Definition | Measured with |
|---|---|---|
| **Count error** | per frame: model count minus human count, for people and for vehicles. Reported as mean absolute error and as "within ±2 on N of M frames" | `tools/handcount.py` (exists) |
| **Inference FPS** | 1000 / mean inference ms, on the Pi | `main.py` (exists) |
| **End-to-end FPS** | decode + pre-process + inference + post-process + add-ons, on the Pi | `main.py` (exists) |

The two goals pull against each other, and the challenge does not need real time. So the plan does
not look for one compromise setting. It builds two profiles and reports both:

| Profile | Used for | Setting (starting guess, to be replaced by measurements) |
|---|---|---|
| **Accurate** | the submitted counts, the report, the hand-count comparison | best FP32 model, tiling on, thresholds from the sweep, rolling median |
| **Fast** | the FPS figure in deliverable 4 | INT8 at 640, no tiling, no image saving, frame skipping |

The benchmark table shows both, with the count difference between them. That sentence in the pitch
("we lose X people per frame to go Y times faster") is the engineering judgement judges look for.

## 2. What is known today

| Finding | Evidence | Consequence |
|---|---|---|
| The confidence threshold is a guess, and it matters | six sample photos: 20 of 69 boxes score between 0.30 and 0.40 | T2, T3 |
| One vehicle can be counted twice | `signal_06`: same vehicle as `car` and `truck`, boxes overlap 98% | T3 |
| Small people are missed | Majestic crowd: 12 people in one pass, 27 with 2×2 tiling | T4 |
| Smaller inputs lose counts fast | two photos, people found: 16 (640 FP32), 14 (640 INT8), 9 (480), 7 (320) | 480/320 are not the fast profile |
| Autos have no class | they land in car or truck | headline only `person` and merged `vehicle` |
| Arm's Pi 5 reference for our two 640 files | FP32 169 ms (5.9 FPS), INT8 134 ms (7.5 FPS), ExecuTorch 1.1.0, XNNPACK | the number to match tomorrow |
| Our Python process is heavy | 500–700 MB peak on laptop and emulator, Arm's runner 59–78 MB | say "whole Python process" next to the RAM figure |
| Unused frames are still decoded | `main.py` calls `cap.read()` on every frame, also the skipped ones | T7 |
| Nothing is measured on a real Pi | only laptop and QEMU | all of section 4 |

## 3. Tonight, in this order

Rule for every accuracy task: it is kept only if the sweep table from T2 gets better. If it does
not, revert it and move on.

### T1. Build the dev set (45 min, needs a human)
Without hand counts, no setting can be chosen honestly.
1. Collect 30 frames into `devset/frames/`: the 14 Bengaluru photos in `img/raw/`, plus 16 frames
   taken evenly from two or three public Indian street clips (daylight, fixed camera if possible).
2. Write the counting rule at the top of `devset/truth.csv` and do not change it afterwards:
   every person and every vehicle (car, bus, truck, auto, two-wheeler, bicycle) at least half
   visible and taller than 2% of the frame height. A rider is a person **and** a vehicle.
3. One person counts all 30 frames from the clean images, never from annotated ones.
4. A second person counts 5 of them. If the two disagree by more than 2 on any frame, the rule is
   unclear: fix the rule, recount.
5. Mark 20 frames `tune` and 10 `check` in the CSV. Settings are chosen on `tune` and the reported
   error comes from `check`.

Done when: `devset/truth.csv` has 30 rows with `frame, person, vehicle, split`.

### T2. Sweep tool (45 min)
New `tools/sweep.py`:
1. For each `.pte`, run every dev frame once and cache the raw candidates (all boxes scoring above
   0.05, before NMS) to `devset/cache/<model>.npz`. Re-running the sweep then costs seconds.
2. Apply every combination of:
   - person threshold 0.15 to 0.50 in steps of 0.05
   - vehicle threshold 0.15 to 0.50 in steps of 0.05
   - NMS IoU 0.45, 0.50, 0.60, 0.70
   - NMS mode: per class (today) or grouped (T3)
   - minimum box height 0, 2%, 4% of the frame height
3. Print, per model, the best setting on `tune` and its error on `check`: mean absolute error for
   people and vehicles, the signed bias (over- or under-counting), and "within ±2 on N of 10".
4. Tie-break towards the higher threshold: fewer false boxes look better in the sample outputs.

Done when: one table ranks models and settings by count error on `check`. This table is the
scoreboard for T3–T6.

### T3. Post-processing fixes (45 min, no speed cost)
| Change | How | File |
|---|---|---|
| Grouped NMS | NMS groups instead of classes: `person` / `car, bus, truck` / `motorcycle, bicycle`. Within a group the higher-scoring box wins and keeps its label | `runner.py` |
| Two thresholds | run the model at the lower of the two, then filter people and vehicles separately. New flags `--conf-person`, `--conf-vehicle` | `runner.py`, `main.py` |
| NMS IoU as a flag | `--iou`, default from the sweep (today fixed at 0.7) | `main.py` |
| Minimum box height | drop boxes shorter than `--min-h` (share of frame height) | `main.py` |
| Optional counting region | a second polygon `count` in `zones.json`: only boxes whose foot point is inside are counted | `insights.py` |
| Headline numbers | report and pitch quote `person` and merged `vehicle` only; per-class columns stay in the CSV | `tools/report.py` |

Done when: `signal_06` reports one vehicle fewer, and the sweep's best setting is the new default.

### T4. More pixels per object (1 hour, costs 4–5× time per frame, accurate profile only)
Two candidates, compare in the sweep, keep one:
- **Tiling for everything.** Today `tiled_people()` handles people only. Extend it: one full-frame
  pass plus four tiles with 20% overlap, all classes, merged with grouped NMS at IoU 0.5. Then drop
  any box that is mostly inside a larger box of the same group (intersection over the smaller area
  above 0.7): that removes halves of objects cut by a tile edge. Cost: 5 inferences per frame.
- **One bigger input.** Export FP32 at 960:
  `YOLO('yolo11n.pt').export(format='executorch', imgsz=960)`, copy to
  `bundle/models/yolo11n_fp32_960.pte`. `runner.py` already reads the input size from the file.
  Cost: about 2.25× the pixels of 640, no seams.

Done when: the sweep shows which of the two has the lower people error on the crowd photos.

### T5. Larger models for the accurate profile (30 min to export)
Published COCO mAP (Ultralytics): 39.5 for n, 47.0 for s, 51.5 for m. That is a bigger step than
any tuning of n.
1. Export both, FP32, 640:
   `YOLO('yolo11s.pt').export(format='executorch', imgsz=640)` and the same for `yolo11m.pt`.
2. Copy to `bundle/models/yolo11s_fp32_640.pte` and `yolo11m_fp32_640.pte`.
3. Check both load in `.venv-twin` (the laptop copy of the Pi's runtime versions).
4. Add them to the sweep.

`run_all.sh` picks up every `.pte` in `bundle/models/` by itself.
Open points: the organisers confirmed n only, so ask tomorrow before using s or m for the submitted
counts. Their time and RAM on the Pi are unknown until measured.

### T6. Rolling median for video (30 min, no speed cost)
A box that flickers off for one frame is a miss in that frame.
- After the run, add `person_smooth` and `vehicle_smooth` to the CSV: the median over a window of
  5 processed frames centred on each frame. Video only, never for a folder of unrelated images.
- The report, the pitch and the hand-count comparison use the smoothed columns for video.
- Use a small `--every` so the 5 frames span well under a second.

Cannot be scored on the dev set if it is all stills; include at least one clip with 10 hand-counted
consecutive frames in T1 if this is to be kept on evidence. Otherwise it is judged tomorrow on the
organiser's clip.

### T7. Do less work per frame (30 min, no accuracy cost)
| Change | How | File |
|---|---|---|
| Skip without decoding | for frames that `--every` drops, call `cap.grab()` and only `cap.retrieve()` the ones that are used | `main.py` |
| Resize before colour conversion | shrink to the model size first, then BGR→RGB on the small image (today the full 1080p frame is converted) | `runner.py` |
| Per-stage timers | time decode, pre-process, inference, post-process, add-ons, image save; print the shares and write them to `summary.json` | `main.py` |
| `--profile fast\|accurate` | one flag that sets model, tiling, thresholds, smoothing and `--deploy` for each profile; `run_all.sh` runs both | `main.py`, `run_all.sh` |

Check: counts on the test clip are identical before and after the first two changes.

### T8. Skip frames where nothing changed (45 min, fast profile, fixed camera only)
1. Keep a 160×90 grey copy of the last frame that went through the model.
2. If the mean absolute difference to the current frame is below 2 (out of 255), reuse the previous
   counts and mark the row `reused=1`.
3. Force a real inference at least every 10th processed frame.
4. Report it as "N of M frames needed inference" and give FPS both with and without it.

Risk: a slow crowd can look like no change. If counts on the test clip differ from the unskipped
run by more than 1 on any frame, lower the threshold or leave the feature off. Off for handheld clips.

### T9. INT8 at 480 (time-boxed to 2 hours, uncertain, do last)
Would combine the quantization gain with the smaller input. Needs our own post-training quantization:
1. Read the ExecuTorch docs for the installed version first; I have not verified the exact API.
2. Export YOLO11n at 480, quantize with the XNNPACK quantizer (symmetric, per-channel weights).
3. Calibrate on 100–200 street images (dev set frames plus COCO val images).
4. Keep the first convolution and the detection head in FP32, as Arm did, or accuracy collapses.
5. Lower to XNNPACK, write `bundle/models/yolo11n_int8_480.pte`, add it to the sweep.

Keep only if its count error on `check` is clearly better than FP32 at 480. Given that 480 already
loses 7 of 16 people on the samples, this may not earn a place even if it works. Drop at 2 hours.

### Tonight's cut order
If time runs out, stop after the last finished item of: T1 → T2 → T3 → T7 → T5 → T4 → T6 → T8 → T9.
T1–T3 alone remove the known double count and replace a guessed threshold with a measured one.

## 4. Tomorrow on the Pi

### M1. Make the board run at full speed, before the first benchmark (10 min)
| Check | Command | Want |
|---|---|---|
| Throttling | `vcgencmd get_throttled` | `0x0` |
| Temperature | `vcgencmd measure_temp` | well under 80 °C during a run |
| Power | Pi 5: the 5 V 5 A supply | no under-voltage warning |
| Cooling | fan or heatsink; any desk fan pointed at it helps | |
| Governor | `echo performance \| sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` | `performance` |
| Load | close the browser and desktop apps, or work over SSH | idle before the run |
| Cores | watch `htop` during a run | all 4 cores busy |

If only one or two cores are busy, the runtime's thread count is the problem. Whether ExecuTorch
1.1.0 exposes that setting in Python is unverified: look it up then.

### M2. First run and sanity check (15 min)
1. `bash run_all.sh <clip> gate --every 3`
2. Compare the two Arm 640 models with Arm's reference (169 ms FP32, 134 ms INT8 on a Pi 5).
   Clearly slower on a Pi 5 means a setup problem: go back to M1 before anything else.
   On a Pi 4 there is no reference; INT8 may be slower than FP32 or fail, and `run_all.sh` falls
   back to FP32.
3. Read the per-stage shares. If inference is under 80% of the time, the clip's decode or the image
   saving is the bottleneck: raise `--every`, use `--deploy` for the timed run.

### M3. Choose the two profiles from `out/benchmark.md` (10 min)
- **Fast:** the highest inference FPS whose count difference to the accurate profile you are willing
  to say out loud.
- **Accurate:** the lowest count error from last night's sweep that finishes the clip in a few
  minutes on this board. If s or m is not allowed or too slow, fall back to n with tiling.

### M4. Calibrate on their clip, honestly (20 min)
1. `python tools/handcount.py sheet --source <clip> --run out/<accurate model>`: 10 clean frames.
2. Someone who has not seen the boxes fills in the counts.
3. Tune the two thresholds on 5 of the frames, only within ±0.10 of last night's values.
4. Report the error on the other 5 and say that this is what was done.

### M5. Benchmark hygiene
- At least 100 timed frames after the warm-up, the same frames for every model.
- Mean, median and p95. Never the best run.
- Run the fast profile twice. A difference above 10% means throttling: cool the board, rerun.
- Write board, OS, Python, temperature and throttle state next to the table.

## 5. What goes in the pitch
- One line for speed: "[fast profile] runs at [N] FPS on this [Pi model]; Arm's own reference for
  this file is 7.5."
- One line for accuracy: "against a human count on [5] frames we did not tune on, we are within
  [N] people and [N] vehicles per frame."
- One line for the trade: "the fast setting loses [N] people per frame, so the counts in our report
  come from the accurate one, which takes [N] seconds per frame."

## 6. Not tonight
| Idea | Why it is the real fix | Why not now |
|---|---|---|
| Fine-tune on Indian road data with an auto-rickshaw class | removes the car/truck confusion and the domain gap | dataset, hours of training, new export, its own validation |
| Crowd-density model | detectors undercount packed crowds by design | a second model, outside PS2 |
| Tracking | per-frame counts are not unique people | not required; unreliable at the Pi's frame rate |
| C++ runner | 59–78 MB instead of 500–700 MB, instant start | a rewrite; inference time barely changes |
| ExecuTorch built for the Pi's CPU | the pip wheel is a generic ARM64 build | hours of build time on unknown hardware |
| Accelerator HAT | 10× and more | not ExecuTorch, not the given hardware |
| Test-time flip augmentation | small gain | 2× time; less than T3–T5, try last if ever |

## 7. What is not known
- Any timing on a real Pi, and which Pi it will be.
- Whether s or m models are allowed.
- How much each accuracy step gains: the six sample photos are too few to promise a number. The
  first real figure is the sweep table from T2.
