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
