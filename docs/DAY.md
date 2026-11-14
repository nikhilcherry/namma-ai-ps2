# Day runbook (14 Nov): the only file you need

Everything here has been run on the Raspberry Pi 5. What is left: the organisers' media, a cool
unthrottled run, the hand count, and the numbers in the docs.

## Decisions already made (change only with a reason)
| Question | Answer |
|---|---|
| Which models run by default | four YOLO11n files: INT8 640, FP32 640, FP32 960, FP32 480 |
| Speed number (deliverable 4) | `--profile fast`: Arm INT8 640, no tiling, no images saved, skips unchanged frames |
| Submitted counts, report, samples | `--profile accurate`: YOLO11n FP32 640, full frame + 4 tiles, smoothing. If it is too slow on the board: `lite` |
| Thresholds | defaults (0.30 / 0.30, NMS IoU 0.70). Not tuned: no human counts existed when this was written |
| Crowded or far-away people | `--profile crowd`: same model, full frame + 16 tiles (17 passes per frame). Majestic sample: 12 people in one pass, 25 with `accurate`, 58 with `crowd`. Adds a few false boxes on close-up traffic |
| YOLO11s / 11m | not used unless the organisers say yes. `ensemble` uses s: comparison only |

## Shortcut: everything measured on the Pi, one command
```bash
bash tools/pi_day.sh /path/to/clip.mp4 gate --every 3     # after install_pi.sh
```
Runs pi_check, parity (FP32 + INT8), keygen, run_all, thread sweep (INT8 and FP32), stamp verify, hand-count sheet; writes `out/pi_day/SUMMARY.md`. A failing step is recorded and the rest still run. The accuracy number still needs a human to fill `handcount/handcount.csv`.

## 0. First 15 minutes (no building yet)
- Ask a mentor: **what do the judges weigh?** **Is a live demo allowed in the 2 minutes?** **Are YOLO11s/m allowed?**
- Get the clip onto the Pi. Note the Pi model and the OS.
- Person C starts reading `docs/pitch.md` aloud now, with blanks.

## 1. Install (target: 15 min)
```bash
git clone --depth 1 https://github.com/nikhilcherry/namma-ai-ps2 && cd namma-ai-ps2
bash bundle/install_pi.sh       # no wheels folder -> downloads the pinned versions (about 200 MB)
bash tools/pi_check.sh --perf   # board, 64-bit OS, power, temperature, throttling, governor, thread count
```
- Offline instead: copy the laptop's `bundle/wheels/` into `bundle/wheels/` before `install_pi.sh`.
- If the install fails, read the ERROR line: it names the cause (32-bit OS, no `python3-venv`, Python version).
- `pi_check.sh` wants: throttle `0x0`, governor `performance`, load near 0, threads == cores. A hot or
  under-powered Pi gives slow numbers that look like a model problem.

## 2. First run: deliverables 1–4 (one command)
```bash
bash run_all.sh /path/to/clip.mp4 gate --every 10      # junction if the clip is mostly vehicles
```
- Start with `--every 10` to see how long the board takes (it prints the wall time per model and in
  total). Then choose `--every` so the whole run takes under 10 minutes, and run again.
- For 10–20 images: `bash run_all.sh /path/to/folder gate`. Timing from few images is weak: add `--repeat 10`.
- It writes `out/benchmark.md` (four models), `out/profiles.md` (fast vs accurate), the report, and `deliverables/`.
- Open two `*_det.jpg` frames by eye. Wrong-looking add-on numbers: add `--no-addons`, do not debug.

**Sanity check on a Pi 5:** Arm's published figures for the same two files are 169 ms (FP32 640) and
134 ms (INT8 640). Clearly slower means a setup problem: go back to `pi_check.sh` before anything else.
On a Pi 4 there is no reference; INT8 may be slower than FP32 or fail, and the script carries on with FP32.

## 3. GATE: save, push, submit
```bash
git add -A && git commit -m "Pi run: counts, benchmark, samples" && git push
python tools/fill_docs.py --write --story=gate      # or junction: fills the blanks in docs/submission.md and docs/pitch.md
```
The repo is already public.
Submit the form now with what you have (`docs/submission.md`). Update it later if the form allows.
Nothing else gets built until this is done.

## 4. Pick the story (2 minutes, by looking at the clip)
- People near or on a road → **gate** (marshal).
- Mostly vehicles → **junction** (signal timing).

## 5. Zone (30 minutes max; skip if the clip is handheld or moving)
```bash
python tools/zones.py --source /path/to/clip.mp4       # open the printed URL, click the road corners, Save
```
Without a zone the report says "Provisional" on its verdict. That is correct, leave it.

## 6. Final run: outputs from the accurate profile
```bash
bash run_all.sh /path/to/clip.mp4 gate --every N --zones zones.json      # outputs come from profile_accurate by default
```
`MAIN_MODEL` decides where the report, the two sample frames and `deliverables/` come from.
**Look at the boxes before choosing.** If people in the clip are small, far away or packed together and
many have no box, use `crowd` instead: `python main.py --source <clip> --profile crowd --out out/profile_crowd --every N --zones zones.json`,
then `MAIN_MODEL=profile_crowd`. It is about 3.5 times slower than `accurate`, so raise `--every`.
`--profile crowd-lite` runs only the tiles that need it: about half the time of `crowd`, a few people fewer in dense crowds.
If `accurate` takes too long on this board, edit nothing: run
`python main.py --source <clip> --profile lite --out out/profile_lite --every N --zones zones.json`
and use `MAIN_MODEL=profile_lite`.

## 7. Hand count: the headline accuracy number (45 min, one person who has not seen the boxes)
```bash
python tools/handcount.py sheet --source /path/to/clip.mp4 --run out/profile_accurate
```
- Open `handcount/f*.jpg` (clean frames), fill `handcount/handcount.csv`.
- Rule: every person and every vehicle (car, bus, truck, auto, two-wheeler, bicycle) at least half
  visible. A rider is a person **and** a vehicle.
```bash
python tools/handcount.py table --run out/profile_accurate
```
- Report the result as it is, on all 10 frames, with the default thresholds. No tuning means no
  question about tuning.
- Paste `handcount/accuracy.md` into the README between the ACCURACY markers. The `handcount/`
  folder is not committed (it holds unblurred frames), so the README is where the table lives.
- Re-run step 6 so the report picks up the accuracy section.

## 8. The three sentences for the pitch (from `out/benchmark.md`, `out/profiles.md`, `handcount/accuracy.md`)
1. Speed: "[fast] runs at [N] FPS on this [Pi model]."
2. Accuracy: "against a human count of 10 frames we are within [N] people and [N] vehicles per frame."
3. The trade: "the fast setting differs by [N] people per frame, so our counts come from the accurate
   one, which takes [N] ms per frame."
Fill the blanks in `docs/pitch.md` and `docs/submission.md`.

## 9. Optional, only if everything above is pushed
| What | Command | When |
|---|---|---|
| Live photo from a phone | `python tools/live.py` → open the printed URL | only if a mentor said a demo is allowed; phone and Pi on one hotspot; never tested with a real phone |
| Stronger benchmark | `python tools/bench_hygiene.py --frames <folder of frames> --all` | two runs per model, flags throttling |
| Thread count | `bash tools/thread_sweep.sh <clip>` | if `pi_check.sh` shows fewer threads than cores |
| Signed results | `python stamp.py keygen` before a run | only if a judge cares about tamper evidence |

## 10. Final push
```bash
git add -A && git commit -m "Final: accuracy, report" && git push
```
Stop building 45 minutes before judging. Rehearse the pitch twice, timed.

## If something breaks
| Problem | Do |
|---|---|
| INT8 model fails or is slower than FP32 (Pi 4) | nothing: the script skips failed models and uses FP32. Say so in the pitch |
| A profile fails | the four model runs still give every deliverable: the script falls back to the FP32 640 run by itself (then fix the "accurate profile" line in `docs/submission.md`) |
| Runs are too slow | raise `--every`; restrict with `MODELS="yolo11n_raspberry_executorch_optimized yolo11n_raspberry_executorch_original"` |
| An add-on number looks wrong | add `--no-addons`, do not debug |
| Report or PDF fails | not a deliverable; open `report.html` on the laptop and print to PDF |
| Pi dies | show `deliverables/` from GitHub, say it is a saved run |
