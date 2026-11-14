# What is still open (14 Nov, event day)

The pipeline has run end to end on the Raspberry Pi 5. Follow `docs/DAY.md`; check off `docs/SHIP_CHECK.md`.

## Open, in order
| # | What | Who |
|---|---|---|
| 1 | Run on the organisers' media (the first Pi run used the stand-in photos in `img/raw`) | Pi |
| 2 | Cool, unthrottled run: `bash tools/pi_check.sh --perf`, wait for `0x0`, then `bash run_all.sh <media> gate --every N` | Pi |
| 3 | `git add deliverables && git commit && git push` from the Pi | Pi |
| 4 | `python tools/fill_docs.py --write --story=gate` (or `junction`) on the Pi, commit | Pi |
| 5 | Hand count 10 frames (`tools/handcount.py`), paste the table into the README. Without it the accuracy lines stay out | person |
| 6 | Read the pitch aloud with a timer; submit the form | person |

## Known and accepted
- **Dense crowds are undercounted.** A crowd-trained model found 97 people on the Majestic photo where ours finds 58. Say this limit out loud.
- **Buses seen from above are mostly missed** by YOLO11n.
- **INT8 counts differ by about one per frame between laptop and ARM.** Only Pi counts go in the submission.
- `tools/live.py` has not been tried from a real phone.

History, rehearsal results and the original plans: `docs/archive/`.
