# Speed plan: getting the most frames per second out of the Pi

"Speed" here means two different numbers, and the pitch must not mix them:
- **Inference FPS:** 1000 / mean inference ms. What deliverable 4 asks for.
- **End-to-end FPS:** decode + pre-processing + inference + post-processing + add-ons.

Nothing has been measured on a real Pi yet. Laptop and emulator timings are not evidence.

## The reference to beat
Arm published numbers for the two 640 models in `bundle/models/`, on a Raspberry Pi 5, ExecuTorch
1.1.0, XNNPACK (their card, `bundle/models/README.md`):

| Model | p50 latency | FPS | Peak memory |
|---|---|---|---|
| YOLO11n FP32, 640 | 169 ms | 5.9 | 78 MB |
| YOLO11n INT8, 640 | 134 ms | 7.5 | 59 MB |

If our Pi 5 numbers for the same files are clearly worse than these, something in the setup is
wrong (step 1) and no tuning should start until it is found. On a Pi 4 expect much less; Arm gives
no figure for it.

## Step 0: measure where the time goes (tomorrow, first run, 10 min)
`main.py` reports inference and end-to-end time. Add per-stage timers (decode, pre-process,
inference, post-process, add-ons, image save) so the table shows which stage to attack.
Expectation, to be checked: inference is 85–95% of the time, so steps 2 and 3 matter most.

## Step 1: make the board run at full speed (tomorrow, 10 min, free)
| Check | Command | Why |
|---|---|---|
| Not throttled | `vcgencmd get_throttled` (want `0x0`), `vcgencmd measure_temp` | a hot or under-powered Pi silently drops its clock |
| Power supply | Pi 5 needs the 5 V 5 A supply | a phone charger causes under-voltage throttling |
| Cooling | fan or heatsink on, or point any fan at it | a bare Pi 5 throttles within minutes under 4-core load |
| CPU governor | `echo performance \| sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` | stops the clock ramping down between frames |
| Nothing else running | close the browser and desktop apps, or run over SSH | 4 cores, and the model wants all of them |
| All 4 cores used | watch `htop` during a run | if only 1–2 cores are busy, the thread count is the problem (see open question) |

Record temperature and throttle state next to every benchmark. It also answers "is that number real?".

## Step 2: pick the right model file (already built, measured by `run_all.sh`)
| Lever | Expected effect | Cost in counts (two sample photos, people found) |
|---|---|---|
| INT8 instead of FP32, 640 | about 1.26× on a Pi 5 (Arm's figure). May be slower or fail on a Pi 4, which lacks the dot-product instructions | 14 vs 16 |
| FP32 at 480 | roughly 1.8× fewer pixels than 640 | 9 vs 16 |
| FP32 at 320 | 4× fewer pixels than 640 | 7 vs 16 |

The trade-off is steep: the small inputs are fast and miss half the people. The likely answer on a
Pi 5 is INT8 at 640. Decide from `out/benchmark.md`, not from this table.

**Missing variant worth building tonight (1–2 hours, uncertain): INT8 at 480.** It would combine
both gains. It needs our own post-training quantization (calibration images, XNNPACK quantizer,
export) and a check that counts survive. Arm kept the first layer and the detection head in FP32 for
accuracy; do the same. Drop it if it is not working in two hours.

## Step 3: do less work per clip (tonight, 1 hour, no accuracy cost)
| Change | Why | Where |
|---|---|---|
| Skip unused frames with `cap.grab()` instead of `cap.read()` | with `--every 3`, two of three frames are fully decoded and converted, then thrown away | `main.py` |
| Resize first, then convert BGR→RGB | today the colour conversion runs on the full 1080p frame, then it is shrunk to 640 | `runner.py` |
| `--deploy` for the timed run | writing an annotated JPEG per frame is slow on an SD card; it is not in the FPS figure but it is in the wall time | already exists |
| Run the tiling pass only when people are small | tiling costs 4 extra inferences per frame | already exists (`--tile auto`) |

These help end-to-end FPS, not inference FPS.

## Step 4: skip frames where nothing changed (tonight, 1 hour, fixed camera only)
Compare each frame with the last one that was run (mean absolute difference on a small grey copy).
If it is below a threshold, reuse the previous counts. On a quiet road this skips most frames.
- Report it as what it is: "N of M frames needed inference".
- Risk: a slow-moving crowd looks like no change. Keep the threshold conservative, and switch it off
  for handheld clips.

## Step 5: two profiles, said out loud
Accuracy and speed pull in opposite directions (see `ACCURACY_PLAN.md`). Do not average them:
- **Fast profile:** INT8 640 (or INT8 480 if it works), no tiling, `--deploy`. This is the FPS number.
- **Accurate profile:** best FP32 setting, tiling on. This is where the submitted counts come from.
State both in the benchmark table with the count difference between them.

## Benchmark hygiene (so the number survives a judge's question)
- At least 100 timed frames after a warm-up; the same frames for every model.
- Report mean, median and p95, not the best run.
- Note board, OS, Python, temperature, throttle state, power supply.
- Run each model twice. If the two runs differ by more than 10%, the board is throttling.

## Open question to look up before relying on it
Whether the ExecuTorch 1.1.0 Python runtime exposes a thread-count setting, and what it defaults to
on a 4-core Pi. I have not verified the API. If `htop` shows all four cores busy, it does not matter.

## Not tonight
| Idea | Why it would help | Why not now |
|---|---|---|
| C++ runner instead of Python | Arm measured 59–78 MB peak memory; our Python process uses 500–700 MB, mostly the `torch` import, and starts in seconds rather than milliseconds | a rewrite; inference time itself would change little |
| ExecuTorch built from source for the Pi's CPU | the pip wheel is a generic ARM64 build | hours of build time on unknown hardware |
| Pruning or distilling a smaller model | fewer operations at the same input size | needs training and validation |
| Hailo or other accelerator HAT | 10× and more | not ExecuTorch, not the given hardware |

## Order
Tomorrow: 1 → 0 → 2 (runs itself) → 5. Tonight, if there is time: 3, then 4, then INT8 at 480.
If only 30 minutes are available: step 3's `grab()` change, and read step 1 so the checks happen
before the first benchmark.
