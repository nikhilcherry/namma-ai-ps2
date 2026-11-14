# Submission form answers (paste, then fill the [blanks])

**Project title:** Edge people & vehicle counter: from counts to a decision

**Team:** Nikhil, Shreyas, Shaun · NMIT Bengaluru

**GitHub:** https://github.com/nikhilcherry/namma-ai-ps2

**Problem statement:** PS2, edge object detection and counting with ExecuTorch on Raspberry Pi

**One-line summary:** A Raspberry Pi counts people and vehicles offline with YOLO11n on ExecuTorch and
turns the counts into a one-page report that tells a ward engineer whether a spot needs action.

**Categories detected:** person, car, bus, truck, motorcycle, bicycle (plus a merged `vehicle` total)

**Counts (deliverable 3):** per-frame counts are in `deliverables/counts.csv`. Over [N] frames:
[N] person detections, [N] vehicle detections; peak [N] people and [N] vehicles in one frame.

**Sample outputs (deliverable 3):** `deliverables/sample_1_*.jpg`, `deliverables/sample_2_*.jpg`

**Performance on the Pi (deliverable 4):** board [Pi model, RAM, OS]. Fast profile: YOLO11n INT8 640, 4.7 MB.
Average inference [N] ms (median [N], p95 [N]). Throughput [N] FPS inference only, [N] FPS end to
end. Peak RAM [N] MB. Full table for four model variants: `deliverables/benchmark.md`. Fast vs accurate: `deliverables/profiles.md`.
The submitted counts come from the accurate profile (YOLO11n FP32 640, full frame + 4 tiles): [N] ms per frame.

**Model used:** YOLO11n (Ultralytics, COCO pretrained), ExecuTorch `.pte`, XNNPACK backend, CPU only.
INT8 and FP32 640 exports from Arm; FP32 480 and 320 exported by us.

**Accuracy:** hand count of [10] frames from the organiser's clip: the model found [N]% of people and
[N]% of vehicles; within ±2 of the human count on [N] of [10] frames.

**Societal issue (deliverable 5):** [gate: decisions about where to post traffic marshals at school
gates are made from complaints, not counts] / [junction: signal timings are set once and not
revisited because manual counts are costly].

**How counting helps:** a week of offline counts from one spot shows when people and traffic share
the road, how often, and for how long, without sending any video anywhere.

**One application:** [gate: deciding whether a school gate needs a marshal and at what time] /
[junction: flagging approaches whose green time no longer matches the traffic].

**Privacy:** heads are blurred in saved frames; deploy mode writes numbers only. No identities. Optional tracking links boxes by overlap only.

**Limits we state:** counts are per frame, not unique; auto-rickshaws are read as car or truck;
small or hidden people are missed; zone and thresholds are set by hand per site.
