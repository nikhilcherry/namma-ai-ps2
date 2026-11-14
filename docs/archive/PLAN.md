> **Tomorrow, follow `docs/DAY.md`.** This file is the background plan; `TODO.md` lists what is still open.

# PS2 plan: edge people & vehicle counter (Namma AI Hackathon, 14 Nov 2026)

Offline counting of people and vehicles on a Raspberry Pi with ExecuTorch. Judged on a working PoC,
societal relevance and the five mandatory deliverables. Not required (official page): live camera,
real-time surveillance, tracking, complex dashboards, cloud, mobile apps.

## Confirmed facts
- Live hackathon Sat 14 Nov, NASSCOM office, 6–8 h of build time, teams of 1–3, finalists attend in person.
- Organisers confirmed by reply: YOLO11n `.pte` is allowed; the board is a **Pi 4 or Pi 5**; we **may carry** a pre-exported `.pte` and scripts.
- Still unknown: which Pi, OS and Python version, internet on the day, how the clip is delivered, scoring weights.
- Submission: project on GitHub + the challenge submission form. Pitch: max 2 minutes.

## The five mandatory deliverables (the gate)
| # | Deliverable | Pipeline part |
|---|---|---|
| 1 | Process the organiser's 10–20 s clip or 10–20 images on ExecuTorch + Pi | P1, P2 |
| 2 | Detect ≥1 category, show counts per frame | P3 |
| 3 | Submit categories, counts, ≥2 sample outputs | P3, P8 |
| 4 | Report avg inference time, model, throughput (measured on the Pi) | P6 |
| 5 | Societal issue, how counting helps, one application, ≤2 min | P9 |

Rule: **nothing beyond these gets built on the day until all five are saved to a safe folder.**

## Pipeline parts
Status: DONE = built and tested on the laptop, TODO = not built, PI = needs a test on the Pi.

| Part | What | Files | Status |
|---|---|---|---|
| **P0 Repo + bundle** | git repo, 7 `.pte` (Arm INT8 + FP32 at 640; ours: FP32 at 960, 480, 320, INT8 480 with FP32 head, YOLO11s FP32), offline aarch64 wheels for Py 3.11/3.12/3.13, install script | `bundle/`, `bundle/install_pi.sh` | DONE · rehearsed on emulated ARM64 (see below) · real Pi untested |
| **P1 Ingest** | Image folder or video, frame sampling (`--every`), warm-up frame | `main.py` | DONE |
| **P2 Inference** | Thin ExecuTorch runner (no ultralytics), Arm pre/post-processing, INT8 and FP32 | `runner.py` | DONE · PI |
| **P3 Count + CSV** | Per-frame counts for 6 classes, merged `vehicle`, CSV, annotated frames | `main.py` | DONE |
| **P4 Insights** | Congestion score, crowd level, pedestrians on road (riders excluded), 2×2 tiling when people are small, browser zone picker | `insights.py`, `tools/zones.py` | DONE |
| **P5 Privacy** | Head blur in saved frames, `--deploy` writes CSV only | `insights.py`, `main.py` | DONE |
| **P6 Benchmark** | mean / median / p95 / std ms, FPS (inference and end to end), model MB, peak RAM, four default models side by side with count differences, plus fast vs accurate profiles | `main.py`, `run_all.sh`, `tools/bench_table.py` | DONE · **numbers must be measured on the Pi** |
| **P7 Integrity** | Hash chain over the CSV, model + input SHA-256, `stamp.py verify` | `stamp.py` | DONE |
| **P8 Accuracy check** | 10 clean frames, a human fills the CSV, model-vs-human table | `tools/handcount.py` | DONE (tool) · counting happens on the day |
| **P9 Story + outputs** | One-page site report (gate or junction story) with heatmap, two pitch scripts, live photo page | `tools/report.py`, `tools/live.py`, `docs/pitch.md` | DONE · **approach PDF fixes still TODO** |
| **P10 Submission** | README (tables auto-filled by `run_all.sh`), form answers drafted, GitHub | `README.md`, `docs/submission.md` | DONE (drafts) · numbers filled on the day |

## Team (max 3 on the floor): Nikhil, Shreyas, Shaun
| Person | Owns |
|---|---|
| **A: Pi + measurements** | P0 install on the Pi, P2 on-device run, P6 numbers, INT8 vs FP32 on the actual board |
| **B: Pipeline + insights** | P1–P5, zone calibration on the organiser's clip, P7 |
| **C: Evidence + pitch** | P8 hand-count, P9 sample outputs, P10 GitHub + form, delivers the 2-minute pitch |

## Before the event (what is left)
See `TODO.md`. In short: team names, the two unsourced pitch lines, the approach PDF, and one read of `docs/DAY.md`.

## Rehearsal on emulated ARM64 (13 Nov)
`bundle/install_pi.sh` + `run_all.sh` were run inside `linux/arm64` Docker containers under QEMU.
Results are recorded at the end of this file. QEMU is not a Pi: it proves the wheels install and the
models load and run on aarch64, not how fast they are, and not that a Pi 4's older CPU is supported.

## Day plan
The commands, in order, are in **`docs/DAY.md`**. Shape of the day:

| Time | Step | Done when |
|---|---|---|
| 0:00–0:15 | Ask a mentor what judges weigh; get the clip; C starts rehearsing | answers noted |
| 0:15–0:45 | `bash bundle/install_pi.sh` | "imports OK" |
| 0:45–1:30 | `bash run_all.sh <clip> gate|junction --every 3` → deliverables 1–4 | `deliverables/` exists |
| 1:30–1:45 | **GATE:** commit, push, submit the form with what exists | pushed |
| 1:45–2:15 | Pick the story; zone (30 min cap, skip if handheld); re-run | report verdict reads right |
| 2:15–3:15 | Hand count 10 frames (someone who has not seen the boxes) | `handcount/accuracy.md` |
| 3:15–3:45 | Choose the working point from `out/benchmark.md`; rehearse the live photo if allowed | one sentence written |
| 3:45–4:30 | Fill pitch blanks, README accuracy, final push, update the form | pushed |
| Rest | Rehearse, timed. Stop building 45 min before judging | ≤2:00 twice |

## What makes this entry different
0. **A decision, not a count:** a one-page report that says whether this spot needs action, with the rule printed on it.
1. **Accuracy against a hand count** of 10 frames on the organiser's own clip.
2. **Four model variants measured on the Pi** (INT8/FP32 at 640, FP32 at 480 and 320) with count differences, and a stated reason for the one we chose.
3. **Limits stated first** (autos read as trucks, per-site calibration, Pi numbers only).
4. **Hash-chained results** that `verify` can re-check.
5. **The story is picked after seeing the clip:** gate (marshal) or junction (signal timing). Both are ready.
6. **Live photo** from a phone, counted on the Pi, if the format allows it.

## Pitch
Two ready scripts with blanks in **`docs/pitch.md`**. Say only numbers measured on the Pi.

## Risks and fallbacks
| Risk | Fallback |
|---|---|
| Pi Python version | offline wheels cover 3.11, 3.12, 3.13; anything else: the install script fetches 3.13 with `uv` (internet) |
| 32-bit Pi OS | nothing installs; ask for a 64-bit image before building anything |
| Pi 4: INT8 slow or will not load | `run_all.sh` skips a failing model and falls back to FP32; report both |
| Venue Wi-Fi is slow | wheels ship on the USB (546 MB); nothing in the pipeline calls out |
| An add-on gives a wrong-looking number on the clip | switch it off (`--no-addons`), do not debug live |
| Pi dies during the demo | pre-recorded run saved from the laptop, clearly labelled as a recording |
| Handheld clip | zones and stopped-object logic are unreliable; counts are not affected |

## Cut list (in order, when time is short)
Live photo → zones → tiling → hand count reduced from 10 frames to 5 → the 320/480 models (keep INT8 vs FP32).
Never cut: P1–P3, the Pi benchmark, the 2-minute pitch, the GitHub submission.

## Not building
Tracking, direction/dwell, any second model, cloud, live camera, dashboards, LoRa (only if radios already work and everything above is done).

## Rehearsal results (13-14 Nov, emulated ARM64 under QEMU, not a Pi)
| Pass | Container | Install | Code | 4 models + 2 profiles | Report | Tests |
|---|---|---|---|---|---|---|
| 1 | `python:3.11-slim-bookworm` (Pi OS Bookworm) | offline: pass | `00c24c2` | pass | `report.html` written | all ok except `test_samples_golden` (see below) |
| 2 | `python:3.13-slim-trixie` | offline: pass | `4020b80` | pass | `report.html` written | same single golden failure |
| 3 | `python:3.11-slim-bookworm`, empty `bundle/wheels/` (plain `git clone`) | **online: pass** (pinned wheels downloaded, `imports OK, XNNPACK: True`) | `659ef75`+ | pass | written | ok |

- PDF skipped in all passes: no browser in the container.
- Failure found: `test_samples_golden` pins x86 INT8 counts; on ARM64 one borderline box flips (brigade_03 12 vs 13 people, junction_01 4/11 vs 5/10). Not a pipeline bug. Fixed in `b622b7d` (the test runs on x86 only); pass 3 ran after it.
- An earlier attempt at pass 3 printed nothing (the filtered output hid the cause); it was re-run unfiltered and passed.
- Python 3.12 was not rehearsed. Timings under QEMU mean nothing and are not recorded here.
- Counts on two sample photos, people found: 16 (640 FP32), 14 (640 INT8), 9 (480), 30 (960). See `EFFICIENCY_PLAN.md` (same folder).
