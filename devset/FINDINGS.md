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
