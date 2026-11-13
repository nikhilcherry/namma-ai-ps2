# What is still open (updated 14 Nov, 02:30)

Tomorrow the team follows **`docs/DAY.md`** and nothing else. Agent work for tonight is finished.

## Done overnight
| Item | Result |
|---|---|
| Default run shortened | four YOLO11n models + `fast` and `accurate`; wall time printed. Laptop: 264 s for the 54 s test clip at `--every 3` |
| Report from a profile | `MAIN_MODEL=profile_accurate` (or `profile_crowd`, `profile_lite`) |
| README protected | Pi table filled only on a real Pi |
| Bundle slimmed | failed INT8 480 deleted, 77 MB medium model laptop-only. Tracked models 86 MB |
| Docs in line | `docs/DAY.md` is the single runbook |
| Crowd problem | `--profile crowd` (17 passes): Majestic sample 25 → 58 people. `--profile crowd-lite` (guided tiles): 51 people at under half the time |
| Paper ideas | five tried; two in, three rejected or evidence-only. Table in `devset/FINDINGS.md` |
| ARM64 rehearsal | offline install + full run pass on Python 3.11 and 3.13. Online install: same 18 files as the offline bundle (checked by hash). Details at the end of `PLAN.md` |

## Known and accepted
- **Dense crowds are still undercounted.** A crowd-trained model found 97 people on the Majestic photo where ours finds 58. That model is third-party, s-size, ONNX and not approved, so it is evidence, not something to ship. Say this limit out loud.
- **Buses seen from above are mostly missed** by YOLO11n (2 found in a depot).
- **INT8 counts differ by about one per frame between laptop and ARM.** Only Pi counts go in the submission.
- `crowd` and `crowd-lite` were added after the ARM rehearsal started and have not been run on ARM. They use the same model file and code path as `accurate`, with more tiles.
- Nothing has been compared with a human count, and nothing has run on a real Pi.

## Needs a person (tomorrow morning)
| # | What |
|---|---|
| 1 | Team names in `PLAN.md` and `docs/submission.md` |
| 2 | The two opening lines in `docs/pitch.md` are unsourced (marked in the file) |
| 3 | Read `docs/DAY.md` once; read one pitch script aloud with a timer |
| 4 | Approach PDF: section order, board price, end-to-end FPS, an accuracy line |
| 5 | Test `tools/live.py` from a real phone if the live demo is wanted |

## At the venue
All in `docs/DAY.md`. Ask first: what judges weigh, whether a live demo is allowed, whether YOLO11s/m are allowed, whether the Pi's OS is 64-bit.

## If an agent continues
No new features. The only useful agent job left is walking through `docs/DAY.md` on the laptop against `test_media/traffic.mp4`
(`PY=.venv-twin/bin/python OUT=out_test/day`) and fixing the doc wherever a command does not work.
