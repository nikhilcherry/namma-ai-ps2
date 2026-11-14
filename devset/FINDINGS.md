# Findings so far (2026-11-13, laptop, NO human counts, NO Pi)

**Read this first.** `devset/truth.csv` does not exist (T1 skipped: it needs a person). Every accuracy number below is
measured against a PROXY reference (yolo11m FP32 + 5-pass tiling, conf 0.35, grouped NMS 0.5). That says how far a cheap
setting is from an expensive one, not how far it is from reality, and the reference shares the detector family's blind
spots (auto-rickshaws, tiny people). Also: `c03/c04/c06/c07/c09/c13` repeat the six `s_*` sample photos, so the 30 frames
hold about 24 distinct scenes, and 10 `v_*` clip frames are mostly empty. Do not quote any of this as accuracy.

## Verified (not proxy-dependent)
| Claim | Evidence |
|---|---|
| Grouped NMS removes the double count | `signal_06` vehicles 7 -> 6; the other five samples are unchanged |
| `--ungrouped` reproduces the old behaviour exactly | 6 sample frames, all counts identical to the pre-change run |
| Resize-before-colour-convert changes no counts | same run |
| Static-frame skip: 2.0 and 1.0 changed one frame's count by 2 (breaks the plan's rule); 0.5 changed none by more than 1 | `traffic.mp4`, every 3rd frame, 216 frames: 174 needed inference, 11.2 FPS vs 9.2 FPS unskipped (laptop) |
| Tiling is ~4x the cost of one pass | yolo11n FP32 640: ~450 ms/frame tiled vs ~90 ms single (laptop) |

## Against the proxy (sum of |model - reference| over 30 frames, people / vehicles, conf 0.30, grouped, IoU 0.5)
| setup | people | vehicles | ms/frame (laptop) |
|---|---|---|---|
| n 640, full frame | 110 | 78 | ~90 |
| n 640, full + 4 tiles | **28** | 61 | ~450 |
| n 960, full frame | 49 | 59 | ~128 |
| n 960, full + 4 tiles | 101 (over-counts crowds: 45 vs 24 on Majestic) | 45 | not run |

* T4 answer for people: 640 with tiling is closest to the proxy; **960 full frame gets most of the way (110 -> 49) at ~1.4x cost instead of ~5x**. Tiling at 960 over-counts.
* Neither fixes vehicles (all four setups miss 45-78 of the reference's 149): vehicles are large, so the gap is the model, not pixels. That is where s/m models or fine-tuning matter.
* Model sizes on the laptop, full frame, mean ms: n INT8 ~100, n FP32 ~87, s ~115, m ~214 (noisy; laptop is not a Pi).

## Not decided, by design
Defaults were NOT changed from the proxy sweep (conf 0.30/0.30, IoU 0.70 kept). The proxy pushes thresholds to 0.15 only
because the reference is more permissive; that is circular. Run `python tools/sweep.py` as soon as `devset/truth.csv` exists.

## Open
* T1 (human counts, rule is in EFFICIENCY_PLAN.md) unlocks the sweep.
* T9 (INT8 at 480) not attempted: 480 already loses counts and the plan ranks it last.
* Everything in section 4 of the plan needs the Pi.
* `bundle/models/yolo11m_fp32_640.pte` is 80 MB (GitHub warns above 50 MB); s is 38 MB.

## Update (later 2026-11-13)
* `tools/devset_init.py` deduped the dev set to 24 distinct frames (16 tune / 8 check, fixed seed) and wrote a blank `devset/truth.csv` with the counting rule on top, plus numbered sheets in `devset/count_sheets/`. **Fill `person` and `vehicle` for each frame from the full-size files, then run `python tools/sweep.py`.** A blank truth.csv is ignored (proxy mode).
* `--profile balanced` = one 960 pass + smoothing. Careful: on `signal_06` it counts 0 people where the 640 run counts 1.
* Pi day: `bash tools/pi_check.sh --perf` (M1), `python tools/bench_hygiene.py --frames devset/frames --all` (M5, two runs per model, flags >10% drift).
* x86 laptop: INT8 (~115 ms) is slower than FP32 (~79 ms) here, so INT8 gains must be measured on the Pi, not assumed.

## T9: own INT8 at 480 (`tools/quantize.py`, PT2E + XNNPACKQuantizer, calibrated on the 24 dev frames)
Proxy-reference error on the 8 check frames, legacy setting (conf 0.30, IoU 0.7), mean abs people / vehicles:
| model | people | vehicles | note |
|---|---|---|---|
| yolo11n_fp32_480 | 1.00 | 0.75 | reference point |
| yolo11n_int8_480 (everything quantized) | **2.25** | 1.25 | scores collapse; the sweep can only pick conf 0.5 |
| yolo11n_int8h_480 (Detect head left FP32) | 1.12 | 0.62 | `--fp32-head`; strips quantizer annotations from the head nodes (XNNPACKQuantizer refuses a `None` config) |

* The FP32 head is what makes it work, as Arm's model card says. Both files load and run on the 1.1.0 twin runtime although exported with 1.6.
* Verdict by the plan's rule ("keep only if clearly better than FP32 at 480"): **not clearly better on accuracy** (about equal). Whether it is faster is a Pi question: laptop timings were swamped by other processes (run-to-run 36 vs 85 ms for the same file), and x86 INT8 was slower than FP32 earlier. `yolo11n_int8h_480.pte` stays in bundle/models so `run_all.sh` and `bench_hygiene.py` time it on the board; drop it if it is not faster there.
* `yolo11n_int8_480.pte` (no FP32 head) is kept only as the negative result; delete before the submission bundle.
* 960 + tiling hybrid: not built. 960 with tiles over-counted crowds (45 vs 24 on Majestic) because tile and full-frame boxes of different scale survive NMS; the cheaper 960 single pass already captures most of the gain.
* Vehicles: yolo11s full frame had the lowest vehicle error of the small models in the proxy sweep (0.6 vs 1.1-1.4 for n); a real answer needs truth.csv.

## Hybrid and vehicle study (`tools/hybrid_eval.py`, vs PROXY reference m640 full+tiles; 24 frames, 145 people / 93 vehicles in the reference)
| recipe | sum abs diff people | vehicles | ms/frame (laptop) |
|---|---|---|---|
| s640 full + s640 tiles | 21 | 25 | 515 |
| n640 full + tiles (current `accurate`) | 25 | 42 | 390 |
| s640 full + n640 tiles | 26 | 35 | 423 |
| n960 full + n640 tiles (the "hybrid") | 28 | 36 | 499 |
| n960 full + s640 full | 37 | 34 | 215 |
| n960 full | 52 | 46 | 110 |
| n640 full | 86 | 55 | 59 |

* **The 960 + tiling hybrid does not earn its place**: slightly worse than current `accurate` for people (28 vs 25), better for vehicles (36 vs 42), 28% slower. Not adopted.
* **Cheapest useful upgrade** (now built: `--profile ensemble`, or `--model A.pte --also B.pte`): n960 full + s640 full, two passes, ~215-230 ms, beats any single full-frame pass on both counts (37/34 vs proxy). Candidates are merged before NMS and a containment filter removes the other model's duplicates. Still only proven against the proxy.
* **Vehicles are undercounted by every recipe at the same 0.35 threshold** (bias -15 to -55 of 93): vehicle scores run lower than person scores. Dropping the vehicle threshold to ~0.20-0.25 closes most of the gap for the tiled recipes, but the reference runs at 0.35 too, so this shows "more boxes", not "more correct". It is the first thing to check with `truth.csv`: sweep `--conf-vehicle` alone. Use `--conf-vehicle 0.25` meanwhile only knowingly.
* s-model tiling over-counts vehicles once cv < 0.25 (bias turns positive), so lowering the threshold and adding capacity do not stack.

## M1 open point answered: thread count (ExecuTorch 1.1.0 Python)
`executorch.extension.pybindings._portable_lib._threadpool_get_thread_count()` exists and reports the XNNPACK pthreadpool size (16 on the 16-core laptop, i.e. all cores). `main.py` now prints it and writes `threadpool_threads` / `cpu_count` to `summary.json`; `tools/pi_check.sh` prints it. On the Pi it should say 4 of 4; anything lower is the "only 1-2 cores busy" case. A setter is not exposed in Python (only `_unsafe_reset_threadpool`), so a low value cannot be fixed from our code.

## Adaptive tiling (`--tile adaptive`)
Run the full frame first; add the four tiles only if `runner.needs_tiles` fires (>= 40 borderline candidates scoring 0.15-0.30, or median person height < 10% of the frame). yolo11n FP32 640, 24 dev frames, laptop:
| | tiled frames | mean ms | vs `--tile all` |
|---|---|---|---|
| `--tile all` | 24 of 24 | 457 | reference |
| `--tile adaptive` | 10 of 24 | **232** | differs on 4 frames: sum abs diff 8 people, 5 vehicles |
The four differing frames: c02 and c05 (near-empty scenes where tiling adds a person: probably tile false positives), c01 (4 vs 3 people, 6 vs 3 vehicles), c08 (tiles find 4 small people the full pass misses). The trigger was read off these same 24 frames, so it is a starting point, not a validated rule; the dev set is also crowd-heavy, so on sparse real footage the saving should be larger. Accuracy of either mode is still unmeasured without truth.csv.

## Threshold choice without truth: cross-model agreement (`tools/consensus.py`)
Five independent detectors (n INT8, n FP32, n 960, s, m), full frame, 24 frames. Metric: mean pairwise |count difference| per frame, divided by the mean count (lower = the models agree more). Consistency, not accuracy: the models can share blind spots.
| threshold | people | vehicles |
|---|---|---|
| 0.15 | 0.35 | 0.36 |
| 0.20 | 0.35 | 0.42 |
| 0.25 | 0.32 | 0.39 |
| **0.30** | **0.32** | **0.36** |
| 0.35 | 0.36 | 0.39 |
| 0.40 | 0.34 | 0.40 |
| 0.45 | 0.40 | 0.40 |
| 0.50 | 0.42 | 0.41 |
* **People: 0.25-0.30 is the agreement minimum, so the current 0.30 default stands.** Above 0.45 the models start disagreeing more.
* **Vehicles: flat (0.36-0.42), no threshold stands out.** This does *not* support lowering `--conf-vehicle` to 0.20-0.25, which the proxy bias table suggested. The proxy's "undercount" at 0.35 partly reflects the reference running at the same threshold. Default stays 0.30/0.30.
* Absolute counts swing 3x across thresholds (e.g. vehicles 69 -> 24 for n INT8 from 0.15 to 0.50) and the models differ in total at every threshold (n 69 vs m 103 vehicles at 0.15): the choice of *model* moves the count more than the choice of threshold in 0.25-0.35. Only truth.csv can say which is right.

## Decisions of record (2026-11-13) and when to revisit
| Decision | Value | Basis | Revisit when |
|---|---|---|---|
| Person threshold | 0.30 | cross-model agreement minimum 0.25-0.30 | truth.csv exists: `python tools/sweep.py` |
| Vehicle threshold | 0.30 | agreement flat 0.15-0.50, no case for a change; proxy bias table is circular | same |
| NMS | grouped, IoU 0.70 full-frame, 0.50 when tiled | verified: removes the car/truck double count, nothing else changes | same |
| Submitted-counts profile | `accurate` (n FP32 640, full + 4 tiles); `lite` if time per frame matters | n is the only organiser-approved model | s/m approved: re-run `hybrid_eval.py`, consider `ensemble` |
| Fast profile | Arm INT8 640 + `--skip-static 0.5` | measured count-diff rule | Pi timings in `bench_hygiene.py` |
Closing step once counts exist: fill `devset/truth.csv`, run `python tools/sweep.py`; it prints ready-to-paste `main.py` flags per model (`devset/sweep.md`, section "Flags for main.py").

## Paper ideas tried (14 Nov, laptop, six sample photos, YOLO11n FP32 640, conf 0.30; counts are people / vehicles)
No human counts exist, so "better" below means "more boxes that look right when the image is opened", not a measured error.

| Photo | 1 pass | 5 passes (`accurate`) | 17 passes (`crowd`) | guided, 5–9 passes (`crowd-lite`) |
|---|---|---|---|---|
| majestic_04 | 12 / 0 | 25 / 0 | 58 / 2 | 51 / 2 |
| brigade_03 | 13 / 2 | 23 / 3 | 36 / 12 | 30 / 12 |
| junction_01 | 4 / 7 | 12 / 12 | 19 / 17 | 19 / 17 |
| arterial_05 | – | 8 / 8 | 8 / 10 | 8 / 10 |
| egipura_02 | – | 5 / 5 | 6 / 7 | 6 / 7 |
| signal_06 | – | 1 / 6 | 2 / 6 | 2 / 6 |

| Idea (paper) | Result | Decision |
|---|---|---|
| Finer slicing, 4×4 tiles (SAHI, Akyon et al. 2022) | Majestic 25 → 58 people. Adds a few false boxes on close-up traffic (a billboard face, one oversized rider box) | **in**: `--tile-grid 4`, `--profile crowd` |
| Cluster-guided tile selection (ClusDet / DMNet idea, using our own full-frame pass as the guide) | 51 of 102 passes over the six photos; same counts on four photos, 58 → 51 and 36 → 30 on the two crowds. Mean 1.16 s vs 2.70 s per image on the laptop | **in**: `--tile guided`, `--profile crowd-lite` |
| SAHI's merge step (GreedyNMM, intersection over smaller, union box) | counts within 1–3 of our NMS + containment filter; union boxes grow large on overlapping vehicles | not adopted, no gain |
| Soft-NMS (Bodla et al. 2017), Gaussian, σ 0.5 and 0.3 | Majestic 58 → 70 but with visible double boxes on single people; junction vehicles 17 → 29 (duplicates) | **rejected** |
| Crowd-trained detector (CrowdHuman; third-party YOLO11s person + head ONNX, `Sharath33/Person`, openrail++) | Majestic people: 39 in one pass, 68 with 5 passes, 97 with 17 (61 heads). Brigade 56, junction 22. Opened and checked: it covers the packed crowd our model misses, with some overlap duplicates | **evidence only**: s-size, ONNX (not ExecuTorch), not ours, not organiser-approved. It shows the real fix is crowd-trained weights, not more tiling |
| Density-map counters (pretrained DM-Count and CSRNet via the `lwcc` library, run in an isolated container) | DM-Count (ShanghaiTech A weights): Majestic 58, Brigade 57, junction 20. CSRNet (ShanghaiTech B weights): 90, 129, 61. The two disagree by up to 3×, and 61 people at the junction is plainly wrong | **rejected**: untrustworthy without calibration on our own scenes, heavy (VGG backbones), and a second model outside the YOLO11n approval |

What this means for the day: `crowd` or `crowd-lite` for small or packed people, `accurate` for ordinary traffic. Say plainly
that a COCO detector undercounts dense crowds and that a crowd-trained YOLO11n is the next step.

## ARM64 vs x86: counts can differ by one on INT8
In the emulated ARM64 rehearsal the Arm INT8 model counted 4 people on `junction_01`; on the laptop it counts 5. Quantized
kernels differ per architecture, so a borderline box can flip. `tests/test_samples_golden.py` is therefore x86-only. Expect
Pi counts to differ from laptop counts by about one per frame on INT8; only Pi counts go in the submission.

## Group scoring for vehicles (`--group-score`, `tools/group_eval.py`; 14 Nov, laptop, vs PROXY yolo11s + tiling, 24 frames, 78 reference vehicles)
Idea: sum car+bus+truck (and bicycle+motorcycle) scores per anchor before the threshold, so an auto-rickshaw scoring car 0.20 / truck 0.18 is kept.
Sum of |model - reference| vehicles, yolo11n FP32 640 full frame, grouped NMS 0.5:

| vehicle threshold | plain | group-score |
|---|---|---|
| 0.15 | **17** (bias -13) | 19 (bias -7) |
| 0.20 | 24 | 22 |
| 0.25 | 31 | 29 |
| 0.30 | 32 | 30 |
| 0.40 | 46 | 39 |

* At the same threshold group scoring is a little closer, but the best plain setting (lower the vehicle threshold to 0.15) beats the best group-score setting (17 vs 19). **Not adopted; the flag stays off.** Re-test on `devset/truth.csv` when it exists: a proxy is not truth.
* Both still undercount at every threshold >= 0.20, consistent with the earlier finding that vehicle scores run lower than person scores.

## Rectangular inputs (`tools/export_rect.py`, `tools/rect_eval.py`; 14 Nov, laptop, vs PROXY yolo11s + tiling)
Exported yolo11n FP32 at 384x640 and 480x864 (`bundle/models/yolo11n_fp32_384x640.pte`, `..._480x864.pte`); `runner.py` reads H and W from the file. Windows export note: ExecuTorch needs `flatc`, which ships in the wheel (`executorch/data/bin/flatc.exe`); put its folder on PATH or set `FLATC_EXECUTABLE`.
Sum of |model - reference| over the frames, single full-frame pass, conf 0.30, grouped NMS 0.5. ms are from THIS machine, for comparing rows only (not Pi numbers).

| frames | model | people | vehicles | mean ms |
|---|---|---|---|---|
| wide only (>= 1.5, 14 frames; ref 43 p / 31 v) | 640x640 | 23 | 13 | 232 |
| | 384x640 | 27 | 16 | 171 |
| | 480x864 | **20** | **12** | 243 |
| | 960x960 | 11 | 10 | 410 |
| all 24 (ref 140 p / 78 v) | 640x640 | 73 | 32 | 239 |
| | 384x640 | 83 | 42 | 173 |
| | 480x864 | 67 | 34 | 232 |
| | 960x960 | 34 | 24 | 406 |

* **480x864 costs about the same as 640x640 and is slightly closer to the proxy on wide frames** (people 20 vs 23, vehicles 12 vs 13): a small gain on 14 frames, within noise. Candidate to replace 640 for 16:9 clips; decide on Pi timings.
* **384x640 is about 26% faster and loses accuracy** (people +4, vehicles +3 on the wide frames). A speed option, not a free win; not adopted for the accurate profile.
* 960x960 is still the most accurate at about 1.7x the time (matches the earlier 960 finding).
* Sample is small, mixed aspect ratios, proxy reference. Repeat on the organiser's clip with `tools/rect_eval.py --frames <dir>`.
* **Portrait `yolo11n_fp32_640x384.pte` (phone clips)**: three upright portrait crops of dev frames, 640x640 vs 640x384: c13 13 vs 8 people (crowd, loses 5), c00 2p 1v vs 2p 1v, c06 3p 2v vs 4p 2v; about 30% faster (e.g. 257 vs 171 ms on this machine). Same trade as 384x640: faster, loses people in crowds. A rotated landscape photo is not a valid test (people lie sideways, both found 0).

## Sweep with the INT8 and 480x864 models (`devset/sweep_int8_rect.md`; 14 Nov, PROXY yolo11s + tiling, 8 check frames)
* INT8 full frame at conf 0.15/0.15, NMS IoU 0.6 (grouped): MAE people 0.50, vehicles 0.12. FP32 `original`, same setting: 0.62 / 0.12. On this small proxy test INT8 is not worse than FP32; Arm's COCO figure says -0.4 mAP. Needs `truth.csv` to say more.
* Winners agree across models: low thresholds (0.15 full frame), IoU 0.45-0.6. Same direction as the earlier sweep. Still PROXY: not adopted as defaults.
* 480x864 full frame at 0.15/0.15 over-counts slightly against the proxy (+1.00 people, +0.88 vehicles); its tiled variant is closest (MAE 0.50 / 0.38). Proxy artefacts are likely at this sample size.

## Correction: timing ratios for the rectangular models (`tools/bench_models.py`, 14 Nov, laptop)
The ms columns in the two sections above came from one pass through the dev set and were noisy: the same 640x640 model measured about 240 ms there and 41 ms in a later run. `tools/bench_models.py` times every model on one frame in interleaved rounds (40 rounds, median), which cancels drift. Ratio to 640x640 FP32, same frame (768x432), all cores / 4 threads:

| model | all cores | 4 threads |
|---|---|---|
| 640x640 FP32 | 1.00x | 1.00x |
| 384x640 | **0.63x** | 0.59x |
| 480x864 | 0.98x | 1.04x |
| 960x960 | 1.92x | 2.21x |
| 640x640 INT8 (Arm) | 1.11x | 1.14x |

* 384x640 saves about 40% of the time (not 26%); 960 costs about 2x (not 1.7x); 480x864 costs the same as 640x640. The accuracy findings above are counts, not ms, and stand.
* INT8 is slower than FP32 on this x86 laptop, as expected (Arm's INT8 kernels need an Arm CPU). It says nothing about the Pi, where Arm measured 1.26x faster for INT8.
* Absolute ms here are laptop numbers. Only these ratios are usable; the Pi has to be measured.
