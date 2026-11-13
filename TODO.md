# What is still open (updated 14 Nov, 00:20, at commit `2bec6c0`)

Tomorrow the team follows **`docs/DAY.md`** and nothing else. This file is for whoever (agent or
person) continues tonight.

**Rule: no new features, flags, profiles or tools.** Only the items below.

## Done tonight
| Item | Result |
|---|---|
| Default run shortened | `run_all.sh` runs four YOLO11n models + `fast` and `accurate` profiles; prints wall time. Laptop: 264 s for the 54 s test clip at `--every 3` |
| Report from a profile | `MAIN_MODEL=profile_accurate bash run_all.sh ...` (tested on the sample photos) |
| README protected | the Pi table is filled only on a real Pi (or `FILL_README=1`) |
| Bundle slimmed | `yolo11n_int8_480.pte` deleted; `yolo11m_fp32_640.pte` untracked (laptop only, still used by `tools/sweep.py --proxy`). Tracked models: 86 MB |
| Docs in line | `docs/DAY.md` rewritten as the single runbook; status table in `EFFICIENCY_PLAN.md`; pitch and form answers match the profiles |
| Dev-set hand count | **dropped on purpose.** Accuracy comes from 10 frames of the organiser's clip, default thresholds, no tuning (`docs/DAY.md` step 7) |

## 1. In progress: ARM64 rehearsal on the new code (machine only)
Running in the background since about 00:05. Three passes under QEMU, 10–15 minutes each:

| Pass | Container | Install path |
|---|---|---|
| 1 | `python:3.11-slim-bookworm` | offline (`bundle/wheels/` copied in) |
| 2 | `python:3.13-slim-trixie` | offline |
| 3 | `python:3.11-slim-bookworm` | online (empty `bundle/wheels/`, what a plain `git clone` at the venue does; never run before) |

- Script: `/tmp/claude-1000/-home-nikhil/8b970b16-761c-4eb3-aeee-a9d8d7907681/scratchpad/rehearse.sh <tag> <offline|online>`
- Log: `/tmp/claude-1000/-home-nikhil/8b970b16-761c-4eb3-aeee-a9d8d7907681/scratchpad/arm3.log` (ends with the line `finished`)
- Each pass runs `bundle/install_pi.sh`, `run_all.sh` on two sample photos (four models + two profiles), then the tests.
- Docker needs `--context default` on this laptop.

**When it finishes:**
1. A pass is good when the log shows `imports OK`, six `===` blocks with no `!!` line, a `DONE.` line, and the tests print `ok`.
2. Replace the "Rehearsal results" block at the end of `PLAN.md` with the three results and the commit they ran on. Say that QEMU timings mean nothing.
3. If a pass fails: fix only that failure, re-run only that pass, record both the failure and the fix.
4. Pass 1 copied the code at commit `00c24c2`; passes 2 and 3 copy the working tree when they start, so they include the later `run_all.sh` change (`MAIN_MODEL` as a profile). If pass 1 is the only one on the old code and it passed, that is fine: note it.
5. Commit and push (`git pull --rebase` first; a second agent has been committing to `master`).

## 2. Small checks an agent can still do tonight
| # | What | Done when |
|---|---|---|
| 2a | Read `docs/DAY.md` top to bottom and run every command in it on the laptop against `test_media/traffic.mp4` (use `PY=.venv-twin/bin/python OUT=out_test/day`) | every command runs; any that does not is fixed in `docs/DAY.md` (the doc, not the code, unless the code is wrong) |
| 2b | `git status` is clean after 2a | no stray `deliverables/`, no modified `README.md` |
| 2c | CI is green on the last commit | `gh run list -L 1` |

Do not go beyond these.

## 3. Needs a person (tomorrow morning is fine)
| # | What |
|---|---|
| 3a | Team names in `PLAN.md` and `docs/submission.md` |
| 3b | The two opening lines in `docs/pitch.md` are unsourced (marked in the file): find a source or ask a mentor |
| 3c | Read `docs/DAY.md` once; read one pitch script aloud with a timer |
| 3d | Approach PDF: section order, board price, end-to-end FPS, an accuracy line (3 of the listed fixes were done in `63496e7`) |
| 3e | Test `tools/live.py` from a real phone if the live demo is wanted |

## 4. At the venue
All in `docs/DAY.md`. The questions to ask first: what judges weigh, whether a live demo is allowed,
whether YOLO11s/m are allowed, and whether the Pi's OS is 64-bit.

## Still unknown
- Any timing on a real Pi, and which Pi it is.
- Real accuracy: nothing has been compared with a human count yet.
- Whether a phone can reach the Pi on the venue network.
- Whether Arm's reference used one thread (a repo note says so; not verified). `tools/pi_check.sh` prints the thread count.
