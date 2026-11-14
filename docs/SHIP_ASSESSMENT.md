# Ship assessment: where we stand against the five deliverables

Summary of `docs/SHIP_CHECK_REPORT.md` (the command-by-command results), checked on the Pi 5 on 14 Nov 2026.

**Overall: the pipeline works, the submission is not ready.** The technical deliverables run and reproduce on the Pi, but they were produced from stand-in photos on a board that throttled, and the written parts (pitch, form) are still templates.

## Deliverables

| # | Deliverable | Status | Why |
|---|---|---|---|
| 1 | Clip processed on ExecuTorch on the Pi | **Partly met** | ExecuTorch ran on a Pi 5 and a clean `master` checkout reruns end to end in 79 s. But the input was 14 Wikimedia stand-in photos (`img/raw`), not the organisers' media. |
| 2 | At least one category, counts per frame | **Met on stand-in data** | `counts.csv` has 14 frames with person, car, bus, truck, motorcycle, bicycle and vehicle columns. Totals are 64 people and 43 vehicles, and they reproduced exactly on a second run. Must be redone on the organisers' media. |
| 3 | Categories, counts, two sample outputs | **Met on stand-in data** | Two annotated samples show boxes, count text and blurred faces. `stamp.py verify` gives `OK` (a hash chain, not a signature). Same caveat as above. |
| 4 | Speed measured on the Pi | **Partly met** | `benchmark.md` has four YOLO11n rows and `profiles.md` has fast and accurate. But the run was throttled (`throttled_during_run: true`, later `0xe0000`) with the `ondemand` governor, and INT8 640 and FP32 640 are 7 to 11% above Arm's figures. Not safe to quote. |
| 5 | Societal issue, how counting helps, one application, under 2 minutes | **Not met** | `docs/pitch.md` has 15 blank lines and still holds both the gate and junction scripts. Two opening lines are unsourced and nobody has timed it. `report.pdf` does exist and shows this run's numbers. |
| - | Actually submitted | **Not done** | `docs/submission.md` has 9 blanks, team names are missing, `deliverables/` is untracked and not in the repo, and the organisers' form is not confirmed submitted. |

## Problems, in order of risk

1. **Wrong input.** `img/raw` is 14 Wikimedia Commons photos (CC BY-SA and CC BY, see `img/raw/meta.json`). No organisers' clip or images are on the Pi. Every number depends on this.
2. **Throttled, slow-governor measurements.** The CPU governor is `ondemand`, the board hit 75 C, and a rehearsal showed FP32 960 at 544 ms against 382 ms in the earlier run. The speed numbers are not trustworthy yet.
3. **The submission contradicts itself.** `docs/submission.md` says the submitted counts come from the accurate profile (FP32 640, with tiling). The deliverables were made with the fast INT8 640 model, tiling off.
4. **Nothing is committed or filled in.** About 24 placeholders across the pitch and form, no team names, and `deliverables/` is not in the repo.
5. **No accuracy evidence.** There is no hand count. In the dense crowd sample (`c04`) many people go undetected, so its count of 12 is an undercount. Without a hand count the pitch has no "can you trust it" claim, and the accuracy lines should be removed rather than left blank.

## What is solid

- The pipeline is reproducible: same code and input give the same counts.
- All five test files pass (two of them have no `__main__` block, so run them through pytest or call the functions directly).
- The outputs are well formed: right columns, blurred faces, valid stamp.
- The repo is public.
- `SHIP_CHECK.md` itself is well built: one command per box with the expected output.

## To finish

1. Put the organisers' media on the Pi.
2. Set the performance governor with `bash tools/pi_check.sh --perf` and let the board cool until `vcgencmd get_throttled` reads `0x0`.
3. `git pull --rebase`, then `bash run_all.sh <media> gate --every 3`.
4. Decide which profile feeds the submitted counts and make `docs/submission.md` match.
5. Fill every `[N]` and `[names]` from the new `deliverables/` files, choose gate or junction, and delete the other script.
6. Hand count 10 frames, or remove the accuracy lines.
7. Commit and push `deliverables/`.
8. A person times the pitch, checks the repo in a private window, and submits the form. Record who and when.

## Helper added after this assessment

`python tools/fill_docs.py --write --story=gate` (or `junction`), run on the Pi after the real run, fills the numeric
blanks in `docs/submission.md` and `docs/pitch.md` from `deliverables/`, deletes the other script, and removes the
accuracy lines when there is no `handcount/accuracy.md`. It refuses non-Pi or throttled runs. Left for a person:
team names, the hand count, and the timed read-aloud.
