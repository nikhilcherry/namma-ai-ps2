# Ship check: are all five deliverables really submitted?

Go top to bottom. Tick a box only after its check passes. **Every box in sections 1 to 6 must be ticked
before anyone says "submitted".** Each check is a command to run from the repo root, with what you must see.

State when this file was written (14 Nov, midday): the code is ready, but no box below was ticked yet.
The only results on the laptop were from our own test clip on x86.

---

## 0. The run happened on the Pi, on the organisers' clip

Everything else depends on this. If either box fails, stop and run
`git pull && bash run_all.sh <organisers-clip> gate|junction --every 3` on the Pi.

- [ ] **Board is a Pi.**
  `head -1 deliverables/benchmark.md` → must name a Raspberry Pi. If it says `x86_64 (not a Pi)`, it is a laptop run.
- [ ] **Source is the organisers' clip.**
  `grep '"source"' deliverables/summary.json` → their file name, not `test_media/traffic.mp4` or `samples/`.
- [ ] **Only YOLO11n was used for the submitted counts.**
  `grep '"model"\|yolo11n_only' deliverables/summary.json` → a `yolo11n_*.pte` model, and `yolo11n_only` is `true`.
- [ ] **The Pi was not throttled.**
  `grep throttled_during_run deliverables/summary.json` → `false`. If `true`, cool the board and run again.

## 1. Deliverable 1: clip processed on ExecuTorch + Pi

- [ ] `deliverables/counts.csv` exists and has one row per processed frame.
  `wc -l deliverables/counts.csv` → more than 10 lines.
- [ ] The run finished without a skipped main model (read the end of the `run_all.sh` output: no `!! ... failed` line for the model named in `summary.json`).

## 2. Deliverable 2: at least one category, counts per frame

- [ ] `head -3 deliverables/counts.csv` → columns `person`, `car`, `bus`, `truck`, `motorcycle`, `bicycle`, `vehicle` with numbers in them.
- [ ] The counts are not all zero.
  `grep people_total deliverables/summary.json` and `grep vehicle_total deliverables/summary.json` → at least one is above 0.
- [ ] A person opened one annotated frame and the boxes sit on real people and vehicles.

## 3. Deliverable 3: categories, counts, two sample outputs

- [ ] `ls deliverables/sample_*.jpg` → at least two files, both from this run (no leftovers from an older run: check the times with `ls -l`).
- [ ] Both samples open, show boxes and the count text, and heads are blurred.
- [ ] `python stamp.py verify deliverables` → `OK`.

## 4. Deliverable 4: speed measured on the Pi

- [ ] `cat deliverables/benchmark.md` → four YOLO11n rows with mean ms, median, p95, FPS, model MB, peak RAM.
- [ ] `cat deliverables/profiles.md` → a `fast` row and an `accurate` row.
- [ ] Sanity on a Pi 5: INT8 640 near 134 ms and FP32 640 near 169 ms (Arm's published figures). Much slower means a setup problem: run `bash tools/pi_check.sh`.
- [ ] The numbers quoted anywhere (form, pitch, README) are copied from these two files, not from a laptop run.

## 5. Deliverable 5: societal issue, how counting helps, one application, under 2 minutes

- [ ] One story chosen: gate **or** junction. The other one is deleted from the form answer.
- [ ] `grep -c '\[N\]\|\[Pi model\]\|\[10\]' docs/pitch.md` → `0` (no blanks left in the script you will read).
- [ ] The two opening lines of the pitch have a source, or are reworded so they need none.
- [ ] Read aloud with a timer: under 2:00.
- [ ] `deliverables/report.pdf` opens and shows this run's numbers.

## 6. It is actually submitted

- [ ] `docs/submission.md` has no blanks: `grep -c '\[N\]\|\[names\]\|\[Pi model' docs/submission.md` → `0`.
- [ ] Team names are in `docs/submission.md` and `PLAN.md`.
- [ ] `deliverables/` is committed and pushed.
  `git status --short deliverables` → empty, and `git log origin/master -1 -- deliverables` shows today's commit.
- [ ] `git status -sb` → not ahead of and not behind `origin/master`. (A second agent pushes too: `git pull --rebase` before every push.)
- [ ] Repo is public: `gh repo view --json visibility -q .visibility` → `PUBLIC`.
- [ ] Open https://github.com/nikhilcherry/namma-ai-ps2 in a private browser window: `deliverables/` is there, the sample images load, and the README Pi table is filled with Pi numbers.
- [ ] The organisers' form is submitted and someone has the confirmation screen or email. Write who and when: ______

## 7. Worth doing, not mandatory

- [ ] **Hand count.** Fill `handcount/handcount.csv` for the 10 frames, then
  `python tools/handcount.py table --run out/<model>`. Without it, remove the accuracy lines from the form and the pitch instead of leaving blanks.
- [ ] **Privacy line matches the code.** `docs/submission.md` says "No tracking, no identities". Track mode is now merged, so say: "No identities. Optional tracking links boxes by overlap only."
- [ ] **Track mode, only if you demo it.** `--profile track --every 1` with a line from `tools/zones.py --line`. Call the unique counts upper estimates. Do not put them in the submitted counts.
- [ ] **Backup.** A copy of `deliverables/` on a second laptop or phone, in case the Pi or the network dies.

---

## Sign-off

| | Name | Time |
|---|---|---|
| Sections 0 to 4 checked on the Pi by | | |
| Sections 5 and 6 checked by | | |
| Form submitted by | | |
