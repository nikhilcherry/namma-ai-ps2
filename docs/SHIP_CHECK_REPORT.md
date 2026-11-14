# Ship check report

Result of going through `docs/SHIP_CHECK.md` top to bottom, on the Raspberry Pi 5 that produced the deliverables.

- **Checked:** 14 Nov 2026, about 06:30 UTC
- **Against:** `origin/master` at `21fef4f` (clean checkout), plus the untracked `deliverables/` and `out/` folders on the Pi
- **Verdict: NOT ready to call "submitted".** Sections 0 to 4 mostly hold up, but three blockers fail and sections 5 and 6 are far from done. See "Blockers" first.

Legend: PASS = command run, result matches. FAIL = command run, result does not match. NOT CHECKED = needs a human or something outside this machine.

## Blockers (fix these before anyone says "submitted")

1. **The run was throttled.** `deliverables/summary.json` has `"throttled_during_run": true`. The checklist says to cool the board and run again. The Pi's CPU governor is `ondemand`, not `performance`, which also fits the slightly slow timings in section 4. Run `bash tools/pi_check.sh --perf` and re-run.
2. **The source is not the organisers' media.** `summary.json` says `"source": "img/raw"`, 14 stills plus a `meta.json`. That file credits every image to Wikimedia Commons (for example "Vehicles stuck in traffic at Egipura signal, Bangalore.jpg", CC BY-SA 3.0), so these are stand-in photos, not the organisers' clip or images. No clip or video exists anywhere on this Pi. The submitted numbers must be re-run on the organisers' media, and any use of the stand-in photos needs their CC attribution.
3. **Nothing is committed.** `deliverables/` is untracked (`git ls-files deliverables` is empty), so the repo that others can see has no deliverables and no benchmark numbers.
4. **The docs are still templates.** `docs/pitch.md` has 15 lines with `[N]`, `[10]` or `[Pi model]` blanks. `docs/submission.md` has 9. Team names are missing from both `docs/submission.md` and `PLAN.md`.

Also worth settling: `docs/submission.md` says the submitted counts come from the *accurate* profile (FP32 640, full frame plus tiles). The deliverables were made by the *INT8 640* model with tiling off (`"profile": "custom"`, `"tile": "off"`). One of the two must change.

## Section 0: the run happened on the Pi, on the organisers' clip

| Check | Result | Evidence |
|---|---|---|
| Board is a Pi | PASS | `Raspberry Pi 5 Model B Rev 1.1`, Debian 13, kernel 6.18.50+rpt-rpi-2712 |
| Source is the organisers' clip | FAIL | `"source": "img/raw"`: 14 Wikimedia Commons stand-in photos (see `img/raw/meta.json`). No organisers' media is on this Pi. |
| Only YOLO11n for submitted counts | PASS | model `yolo11n_raspberry_executorch_optimized.pte`, `"yolo11n_only": true` |
| Pi was not throttled | FAIL | `"throttled_during_run": true`. Right now `vcgencmd get_throttled` is `0x0` at 64.8 C, governor `ondemand` |

## Section 1: clip processed on ExecuTorch on the Pi

| Check | Result | Evidence |
|---|---|---|
| `counts.csv` has one row per frame, more than 10 lines | PASS | `wc -l` gives 15: a header plus 14 frames, matching `"frames": 14` |
| Main model not skipped | PASS (inferred) | The `run_all.sh` log is not kept, but `benchmark.md` has all four models with numbers, and a failed model has its `summary.json` deleted |

## Section 2: counts per frame

| Check | Result | Evidence |
|---|---|---|
| Columns `person, car, bus, truck, motorcycle, bicycle, vehicle` with numbers | PASS | header and first rows checked, e.g. `c00,0.0,4,3,1,0,0,0,4,...` |
| Counts not all zero | PASS | `people_total` 64, `vehicle_total` 43 |
| Boxes sit on real people and vehicles | PASS with a caveat | Both sample images opened and inspected by eye. Boxes are on real objects. In the dense crowd frame (`c04`) many people in the back are not boxed, so the count of 12 is an undercount. |

## Section 3: categories, counts, two samples

| Check | Result | Evidence |
|---|---|---|
| At least two `sample_*.jpg`, from this run | PASS | `sample_1_c04.jpg`, `sample_2_c06.jpg`, both 11:15 today, same time as the rest of `deliverables/` |
| Samples show boxes, count text, blurred heads | PASS | Both show boxes, the count text overlay, and blurred faces on detected people. People the model misses are not blurred. |
| `python stamp.py verify deliverables` is `OK` | PASS | `OK: 14 rows, head 0fe531bc..., model 154c54b5..., unsigned (chain only)`. The stamp is a hash chain, not a signature. |

## Section 4: speed measured on the Pi

| Check | Result | Evidence |
|---|---|---|
| `benchmark.md`: four YOLO11n rows with all columns | PASS | table below |
| `profiles.md`: a `fast` and an `accurate` row | PASS | table below |
| Pi 5 sanity: INT8 640 near 134 ms, FP32 640 near 169 ms | Borderline | 148.1 ms and 186.8 ms, about 10 to 11% slower than Arm's figures. Not "much slower", but consistent with the throttle flag and `ondemand` governor. Re-run before quoting. |
| Quoted numbers (form, pitch, README) copied from these files | FAIL | The pitch and form are still blanks, and the README on `master` has no Pi numbers table |

`deliverables/benchmark.md` (14 frames each):

| Model | Input | Mean ms | Median | p95 | FPS (infer) | FPS (end to end) | Model MB | Peak RAM MB | People | Vehicles |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n_fp32_480 | 480 | 80.9 | 80.5 | 82.5 | 12.37 | 11.56 | 10.13 | 477 | 54 | 35 |
| yolo11n_fp32_960 | 960 | 382.0 | 379.1 | 399.8 | 2.62 | 2.51 | 10.29 | 603 | 113 | 61 |
| yolo11n_raspberry_executorch_optimized (INT8) | 640 | 148.1 | 147.7 | 150.7 | 6.75 | 6.36 | 4.73 | 486 | 64 | 43 |
| yolo11n_raspberry_executorch_original (FP32) | 640 | 186.8 | 186.6 | 187.9 | 5.35 | 5.11 | 10.23 | 505 | 69 | 46 |

`deliverables/profiles.md`:

| Profile | Model | Tile | Inference ms (mean/median/p95) | End-to-end FPS | Peak RAM MB |
|---|---|---|---|---|---|
| fast | INT8 640 | off | 146.8 / 145.6 / 154.8 | 6.2 | 463 |
| accurate | FP32 640 | all | 998.5 / 1008.0 / 1034.1 | 0.96 | 506 |

## Section 5: the 2-minute pitch

| Check | Result | Evidence |
|---|---|---|
| One story chosen, the other deleted | FAIL (partly) | `report.pdf` is titled "Does this gate need a traffic marshal?", so gate. But `docs/pitch.md` still holds both Script A (gate) and Script B (junction). |
| No blanks in the script | FAIL | `grep -c '\[N\]\|\[Pi model\]\|\[10\]' docs/pitch.md` gives 15, not 0 |
| Two opening lines sourced or reworded | NOT CHECKED | needs a human |
| Read aloud, under 2:00 | NOT CHECKED | needs a human with a timer |
| `report.pdf` opens and shows this run's numbers | PASS | 14 frames, peak 12 pedestrians on the carriageway at `c04`, BUSY or JAMMED in 36% of frames, mean 3.1 vehicles per frame. It is labelled "Provisional" because the road zone is not calibrated. |

## Section 6: it is actually submitted

| Check | Result | Evidence |
|---|---|---|
| `docs/submission.md` has no blanks | FAIL | 9 lines with `[N]`, `[names]`, `[Pi model` or `[10]` |
| Team names in `docs/submission.md` and `PLAN.md` | FAIL | `submission.md` still says `[names]`. `PLAN.md` says "confirm the names on the registration". |
| `deliverables/` committed and pushed | FAIL | `git status --short deliverables` shows `?? deliverables/`, and no commit on `origin/master` touches it |
| Not ahead of or behind `origin/master` | FAIL | the Pi's clone was 2 commits behind and had uncommitted edits to `main.py` and `README.md` |
| Repo is public | PASS | https://github.com/nikhilcherry/namma-ai-ps2 returns HTTP 200 with no login. The `gh` CLI is not installed here, so this was checked over plain HTTP. |
| Open the repo in a private browser window | NOT CHECKED | needs a human |
| Organisers' form submitted, with a confirmation | NOT CHECKED | needs a human. Write who and when: ______ |

## Section 7: worth doing

Not done.

- Hand count (`handcount/handcount.csv`): not done. Until it is, the accuracy lines in the pitch and form should be removed.
- Privacy line in `docs/submission.md`: not updated for track mode.
- Track mode demo: not run.
- Backup copy of `deliverables/`: not made.

## Code health (extra)

All five test files in `tests/` pass on `origin/master` (`21fef4f`), run with the project's `.venv-pi` Python, which has no pytest:

- `test_postprocess.py`, `test_samples_golden.py`, `test_track.py`: pass as scripts.
- `test_group_score.py` (4 tests) and `test_sweep_logic.py` (3 tests) have no `__main__` block, so running them as scripts does nothing. Calling the 7 test functions directly: all 7 pass.

## Caveats on this report

- The deliverables were produced by the Pi's local checkout, which was 2 commits behind `origin/master` and had uncommitted edits to `main.py` and `README.md`. The code that made them may not be exactly what is on `master`. The stamp in `deliverables/stamp.json` pins the model and source hashes, not the code.
- Section 6 repo checks (behind, dirty, untracked) describe the Pi's clone at the time of checking. This report does not change them.
- Nothing in this report edits `SHIP_CHECK.md`, so no boxes are ticked there.

## What to do next, in order

1. On the Pi: `bash tools/pi_check.sh --perf`, check `vcgencmd get_throttled` is `0x0`, then `git pull --rebase` and `bash run_all.sh <organisers-clip> gate --every 3`.
2. Decide which profile feeds the submitted counts, and make `docs/submission.md` match.
3. Fill every `[N]` and `[names]` from the new `deliverables/` files. Delete the unused pitch script.
4. Commit and push `deliverables/`.
5. A person does the timed read, the private-window check, and the form.
