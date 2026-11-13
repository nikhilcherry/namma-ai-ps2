# Day runbook (14 Nov): commands in order

Everything below is already built. On the day you run it, count by hand, and fill in numbers.

## 0. First 15 minutes
- Ask a mentor: **what do the judges weigh?** and **is a live demo allowed in the 2 minutes?**
- Get the clip onto the Pi. Note the Pi model and the OS.
- Person C starts rehearsing `docs/pitch.md` now, with blanks.

## 1. Install (target: 15 min)
```bash
git clone https://github.com/nikhilcherry/namma-ai-ps2 && cd namma-ai-ps2
# copy bundle/wheels/ from the USB stick into bundle/wheels/ (offline path), or skip it (online path)
bash bundle/install_pi.sh
```
If it fails: read the ERROR line, it says what is wrong (32-bit OS, no python3-venv, Python version).

## 2. Deliverables 1–4 (one command)
```bash
bash run_all.sh /path/to/clip.mp4 gate --every 3      # junction if the clip is vehicle-heavy
```
- A 20 s clip at 30 fps is 600 frames. `--every 3` keeps 200 frames per model; raise it if a run takes more than 5 minutes.
- For 10–20 images: `bash run_all.sh /path/to/folder gate`
- Check `out/benchmark.md` and open two `*_det.jpg` frames by eye.

## 3. GATE: save and push
```bash
git add -A && git commit -m "Pi run: counts, benchmark, samples" && git push
```
Submit the form now with what you have (`docs/submission.md`). Update it later if the form allows.

## 4. Pick the story (2 minutes, by looking at the clip)
- People near or on a road → **gate** (marshal).
- Mostly vehicles → **junction** (signal timing).

## 5. Zone (30 minutes max; skip if the clip is handheld or moving)
```bash
python tools/zones.py --source /path/to/clip.mp4       # open the printed URL, click the road corners, Save
bash run_all.sh /path/to/clip.mp4 gate --every 3 --zones zones.json
```

## 6. Hand count (the headline number)
```bash
python tools/handcount.py sheet --source /path/to/clip.mp4 --run out/yolo11n_raspberry_executorch_optimized
```
- The counter must **not** have seen the model's output. Open `handcount/f*.jpg` (clean frames), fill `handcount/handcount.csv`.
- Rule: every person and vehicle at least half visible. A rider is a person **and** a vehicle.
```bash
python tools/handcount.py table --run out/yolo11n_raspberry_executorch_optimized
bash run_all.sh ... (same as step 5)       # re-run so the report picks up the accuracy section
```
Paste `handcount/accuracy.md` into the README between the ACCURACY markers.

## 7. Choose the working point (one sentence for the pitch)
Read `out/benchmark.md`: pick the model with the best FPS whose Δ counts are small. Say why:
"we run 480 because it is X times faster and differs by Y people per frame".
To make that model the source of the report: `MAIN_MODEL=yolo11n_fp32_480 bash run_all.sh ...`

## 8. Live moment (only if a mentor said it is allowed)
```bash
python tools/live.py           # prints http://<pi-ip>:18080
```
Phone on the same network (a phone hotspot with mobile data off proves "offline"). Rehearse it twice.
If the network blocks device-to-device traffic, drop it. It is a bonus, not a deliverable.

## 9. Final push and form
```bash
git add -A && git commit -m "Final: accuracy, report" && git push
```
Stop building 45 minutes before judging. Rehearse the pitch twice, timed.

## If something breaks
| Problem | Do |
|---|---|
| INT8 model fails or is slower than FP32 (Pi 4) | nothing: `run_all.sh` skips failed models and falls back to FP32. Say so in the pitch. |
| An add-on number looks wrong | add `--no-addons`, do not debug |
| Runs are too slow | raise `--every` |
| Report or PDF fails | not a deliverable; open `report.html` on the laptop and print to PDF |
| Pi dies | show the saved outputs in `deliverables/` from GitHub, say it is a saved run |
