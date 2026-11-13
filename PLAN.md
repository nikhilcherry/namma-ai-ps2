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
| **P0 Repo + bundle** | git repo, Arm INT8 + FP32 `.pte`, 18 offline aarch64 wheels (Py 3.13), install script | `bundle/`, `install_pi.sh` | DONE (laptop) · PI untested · not pushed to GitHub |
| **P1 Ingest** | Image folder or video, frame sampling (`--every`), warm-up frame | `main.py` | DONE |
| **P2 Inference** | Thin ExecuTorch runner (no ultralytics), Arm pre/post-processing, INT8 and FP32 | `runner.py` | DONE · PI |
| **P3 Count + CSV** | Per-frame counts for 6 classes, merged `vehicle`, CSV, annotated frames | `main.py` | DONE |
| **P4 Insights** | Congestion score, crowd level, pedestrians on road (riders excluded), 2×2 tiling when people are small | `insights.py` | DONE · **zone helper TODO** |
| **P5 Privacy** | Head blur in saved frames, `--deploy` writes CSV only | `insights.py`, `main.py` | DONE |
| **P6 Benchmark** | mean / median / p95 / std ms, FPS, model MB, peak RAM, INT8 vs FP32 table | `main.py` | DONE · **numbers must be measured on the Pi** |
| **P7 Integrity** | Hash chain over the CSV, model + input SHA-256, `stamp.py verify` | `stamp.py` | DONE |
| **P8 Accuracy check** | Show 5 frames, enter hand counts, write a model-vs-human table | `tools/handcount.py` | TODO |
| **P9 Story + outputs** | Fix the approach PDF, ward-report one-pager from the CSV, foot-point heatmap image, 2-min pitch | `docs/` | TODO |
| **P10 Submission** | README with Pi tables, push to GitHub, submission form | `README.md` | TODO |

## Team (max 3 on the floor; confirm the names on the registration)
| Person | Owns |
|---|---|
| **A: Pi + measurements** | P0 install on the Pi, P2 on-device run, P6 numbers, INT8 vs FP32 on the actual board |
| **B: Pipeline + insights** | P1–P5, zone calibration on the organiser's clip, P7 |
| **C: Evidence + pitch** | P8 hand-count, P9 sample outputs, P10 GitHub + form, delivers the 2-minute pitch |

## Before the event (tonight and Saturday morning)
1. Confirm the 3 attendees and the venue/time from the finalist email.
2. Zone-drawing helper: click 4 points on a frame, save `zones.json` (P4).
3. Hand-count tool (P8).
4. Fix the approach PDF, add team names (P9). Fixes: section order, "children" → "people", the "6 riders" line, bicycle in the vehicle total, board price, end-to-end FPS next to inference FPS, an accuracy line, and say faces are blurred.
5. Push the repo to GitHub (public or private: decide).
6. USB stick: repo, `bundle/wheels/`, both `.pte`, `SHA256SUMS`, this plan, the six sample photos. Repo on a phone too.
7. Rehearse the offline install once on any ARM64 Linux if available; time the 2-minute pitch.
8. Send the follow-up to the organisers: which Pi and OS/Python, internet on the day, how the clip is delivered.

## Day plan (6–8 hours)
| Time | Step | Done when |
|---|---|---|
| 0:00–0:30 | `install_pi.sh`; run both models on one frame | prints counts on the Pi |
| 0:30–1:15 | **Deliverables 1–2:** organiser clip → counts, CSV, annotated frames | CSV + frames exist |
| 1:15–1:45 | **Deliverable 4:** measure on the Pi: INT8 and FP32, mean/median/p95, FPS, MB, RAM | numbers saved |
| 1:45–2:00 | **GATE:** copy outputs to a safe folder, commit | all five deliverables safe |
| 2:00–3:00 | Draw zones on the clip, turn on P4/P5, re-run, check 3 frames by eye | numbers look right |
| 3:00–3:45 | Hand-count 5 frames (P8) | accuracy table done |
| 3:45–4:30 | Pick the 2 sample outputs, ward report, heatmap | outputs chosen |
| 4:30–5:30 | README tables filled, push to GitHub, submission form | submitted |
| Last hour | Stop building 45 min before judging; rehearse the pitch twice | timed at ≤2:00 |

## What makes this entry different
1. **Accuracy against a hand count** on the organiser's own clip.
2. **INT8 vs FP32 measured on the Pi**, with the count differences.
3. **Limits stated first** (autos read as trucks, per-site calibration, Pi numbers only).
4. **Hash-chained results** that `verify` can re-check.
5. A **school-gate story**: people on the road, peak crowd, congestion → "does this gate need a marshal?"

## Pitch (2:00, one problem, one application)
0:00 problem (no one knows how many cross the gate at 8:40) · 0:20 what the Pi does, offline, faces blurred ·
0:50 beyond counting: on-road count, congestion, peak · 1:20 Pi numbers (X ms, Y FPS, Z MB, RAM) ·
1:40 one-week CSV tells the ward engineer if a marshal is needed · 1:55 same box, different model file.
Say only numbers measured on the Pi. State the limits before a judge asks.

## Risks and fallbacks
| Risk | Fallback |
|---|---|
| Pi Python is not 3.13 | wheels will not install: ask for a 3.13 env, or build wheels for its version (add 3.11/3.12 sets before the event) |
| Pi 4: INT8 slow or will not load, stock wheel too new | run FP32, report both; fall back to a v8.0 ExecuTorch build if the wheel fails |
| No internet | everything ships on the USB; nothing in the pipeline calls out |
| An add-on gives a wrong-looking number on the clip | switch it off (`--no-addons`), do not debug live |
| Pi dies during the demo | pre-recorded run saved from the laptop, clearly labelled as a recording |
| Handheld clip | zones and stopped-object logic are unreliable; counts are not affected |

## Cut list (in order, when time is short)
Heatmap → ward-report PDF → hand-count of 5 frames reduced to 3 → tiling → INT8 comparison (keep FP32 only).
Never cut: P1–P3, the Pi benchmark, the 2-minute pitch, the GitHub submission.

## Not building
Tracking, direction/dwell, any second model, cloud, live camera, dashboards, LoRa (only if radios already work and everything above is done).
