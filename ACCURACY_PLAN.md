# Accuracy plan: getting the counts as close to a human as this pipeline can

"Accuracy" here means: per-frame count of people and of vehicles, compared with a human count of
the same frame (`tools/handcount.py`). Not mAP. The steps are ordered by gain for effort.

Two facts decide the order:
- The challenge does **not** need real time. A 20 s clip may take minutes. So the counts can come
  from the slowest, most accurate setting, and the speed numbers from the fast one. Report both.
- Nothing below can be chosen honestly without ground truth. Step 0 comes first.

## What is known today (measured on the six sample photos, laptop)
| Finding | Evidence |
|---|---|
| The threshold matters a lot | 20 of 69 boxes have a confidence between 0.30 and 0.40. Moving `--conf` by 0.1 changes the count by about a third. |
| One vehicle can be counted twice | `signal_06`: the same vehicle is reported as `car` and `truck` (boxes overlap 98%). NMS is per class, the vehicle total is merged. |
| Small people are missed | Majestic crowd: 12 people at one pass, 27 with 2×2 tiling. |
| Smaller inputs lose counts | Two photos: 16 people at 640 FP32, 14 at 640 INT8, 9 at 480, 7 at 320. |
| Autos have no class | They land in car or truck. Per-class vehicle numbers are unreliable, the merged total is not. |

## Step 0: a dev set and a sweep tool (tonight, 1 hour) — everything else depends on it
1. Hand-count 30 frames: the 14 Bengaluru photos in `img/raw/` plus 16 frames from two or three
   public Indian traffic clips. Use the rule in `tools/handcount.py`. Write the rule down once.
2. Save detections at a low threshold (`--conf 0.10`) so thresholds can be swept without re-running
   the model.
3. New `tools/sweep.py`: for each setting (conf, NMS IoU, model, tiling) print the mean absolute
   count error for people and vehicles against the 30 hand counts.

Done when: one table says which setting has the lowest error. Every later step is kept only if this
table improves.

## Step 1: free fixes in post-processing (tonight, 1 hour, no speed cost)
| Change | Why | Where |
|---|---|---|
| Class-agnostic NMS across car/bus/truck, and across motorcycle/bicycle | removes the car+truck double count | `runner.py` |
| Separate thresholds for people and vehicles, chosen by the sweep | one global 0.30 is a guess | `main.py` |
| NMS IoU 0.7 → sweep 0.45–0.7 | 0.7 is the benchmark setting and keeps near-duplicates; too low merges people standing close | `runner.py` |
| Minimum box size and a counting region | far-away specks are noise, and a human would not count them either; the rule and the model must count the same things | `main.py`, `zones.json` |
| Report only `person` and merged `vehicle` as headline numbers | per-class splits carry the auto confusion | report, pitch |

## Step 2: more pixels per object (tonight to build, costs 4–5× time per frame)
Small and distant objects are the biggest source of misses in street scenes.
- Extend 2×2 tiling to vehicles (today it is people only), merge with the full-frame pass, and switch
  it on by default for the "accurate" run.
- Or export one FP32 model at 960 or 1280 input and compare it with tiling in the sweep. One pass,
  no seams where objects are cut by a tile edge. Keep whichever wins.

## Step 3: a larger model for the counts (tonight to export, ask the organisers first)
YOLO11s and YOLO11m are the same family and export the same way. Ultralytics' published COCO mAP is
39.5 for n, 47.0 for s, 51.5 for m. That is a larger jump than any tuning of n.
- Export `yolo11s` and `yolo11m` FP32 at 640, add them to `bundle/models/`. `run_all.sh` picks up
  every `.pte` automatically.
- The organisers confirmed n is allowed. Ask whether s or m is, before relying on it.
- Unknown until measured on the Pi: time per frame and RAM.

## Step 4: use time, for video only (tonight, 30 min, no speed cost)
A detection that flickers off for one frame is a miss in that frame. With a fixed camera:
- Report the rolling median of the count over 3–5 consecutive frames.
- Run with a small `--every` so neighbouring frames are close in time.
This needs no tracking and no identities. It does nothing for a folder of unrelated images.

## Step 5: calibrate on the day, honestly (tomorrow, 20 min)
1. Hand-count 10 frames of the organiser's clip.
2. Tune the two thresholds on **5** of them.
3. Report the error on the **other 5**, and say that this is what was done.
Tuning and reporting on the same frames would look better and mean nothing.

## Step 6: FP32 for counts, INT8 for speed
INT8 lost 2 of 16 people on the sample photos. Use the best FP32 setting for the submitted counts and
show INT8 in the benchmark table as the fast option, with its count difference.

## Not tonight (the real fixes, each a project of its own)
| Fix | Why it is the real one | Why not now |
|---|---|---|
| Fine-tune on Indian road data with an auto-rickshaw class | removes the car/truck confusion and the domain gap | needs a labelled dataset, hours of training, a new export and its own validation; too risky the night before |
| Crowd-density model for packed scenes | detectors undercount dense crowds by design | a second model, outside PS2's scope |
| Tracking for unique counts | per-frame counts are not unique people | explicitly not required, and unreliable at the Pi's frame rate |

Test-time flip augmentation is possible too (2× time). Expect it to add less than steps 1–3; try it
last, if at all.

## Order and stop rule
0 → 1 → 2 → 3 → 4 tonight, each kept only if the sweep table improves. 5 and 6 tomorrow.
If only one hour is available: do step 0 with 15 frames and step 1. Those two alone remove the known
double count and replace a guessed threshold with a measured one.

What "best" will look like is not known yet. The sample photos are too few to promise a number; the
sweep table from step 0 is where the first real one comes from.
