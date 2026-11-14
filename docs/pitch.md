# Pitch (2:00). Pick ONE script after seeing the clip.

Fill every [blank] from `out/benchmark.md`, `handcount/accuracy.md` and the report. Say only numbers
measured on the Pi. Each script is about 230 words: 1:45 at a normal pace, which leaves 15 s of slack.

## Script A: gate (clip has people near a road)

**0:00 Problem.** Children cross roads to reach school gates. Whether a gate gets a traffic marshal
should depend on a count, so we built the counter.

**0:20 What we built.** This Raspberry Pi counts people and vehicles from a camera view, fully
offline, with a YOLO11n model on ExecuTorch. No cloud, no internet, and in deploy
mode it stores no images at all (frames we do save blur the heads it detects). Only numbers leave the box.

**0:45 Result on your clip.** On the clip you gave us it found a peak of [N] people in view and
[N] pedestrians on the road itself, with traffic busy in [N]% of frames.

**1:00 Can you trust it.** We counted [10] frames by hand. The model found [N]% of the people and
[N]% of the vehicles a human found, within two of the human count on [N] of [10] frames. It misses
small and half-hidden people, and it reads autos as cars or trucks, so we report one vehicle total.

**1:20 On the Pi.** [Pi model]: [N] ms per frame, [N] frames per second, a [N] MB model, [N] MB of
RAM. That is our fast setting. It differs from our accurate setting by [N] people per frame, so the
counts in our report come from the accurate one, at [N] ms per frame.

**1:40 What it is for.** Leave this box at a gate for a week. The ward engineer gets one page:
this gate needs a marshal from 8:30 to 8:50, or it does not. Same box, any gate.

## Script B: junction (clip is mostly vehicles)

**0:00 Problem.** A signal timing is only right for the traffic it was set for.
Re-timing needs a count, and a count by hand costs a person a day, so we built the counter.

**0:20 What we built.** This Raspberry Pi counts vehicles and people from a camera view, fully
offline, with a YOLO11n model on ExecuTorch. No cloud, no internet, and in deploy
mode it stores no images at all (frames we do save blur the heads it detects). Only numbers leave the box.

**0:45 Result on your clip.** On the clip you gave us it saw a peak of [N] vehicles in view, an
average of [N], and the road was busy or jammed in [N]% of frames.

**1:00 Can you trust it.** We counted [10] frames by hand. The model found [N]% of the vehicles and
[N]% of the people a human found, within two of the human count on [N] of [10] frames. It reads
autos as cars or trucks, so we report one vehicle total and say so.

**1:20 On the Pi.** [Pi model]: [N] ms per frame, [N] frames per second, a [N] MB model, [N] MB of
RAM. That is our fast setting. It differs from our accurate setting by [N] vehicles per frame, so the
counts in our report come from the accurate one, at [N] ms per frame.

**1:40 What it is for.** Leave this box at an approach for a week. The traffic engineer gets one
page: when the queue fills the road, and whether the green time should change. One box per
approach, and no video ever leaves the pole.

## Questions to expect
| Question | Answer |
|---|---|
| Is this unique people? | No. Counts per frame. Unique counts need tracking, which we left out on purpose. |
| Why not a bigger model? | Measured: [fill from benchmark]. Nano gives [N] FPS on this board. |
| Did you train it? | No. Pretrained COCO YOLO11n; INT8 and FP32 exports from Arm, 320/480 exports are ours. |
| What about autos? | No auto class in COCO. They land in car or truck, so the merged vehicle count is the honest one. |
| How do you know nobody edited the CSV? | Every row is hash-chained with the model and input hashes; `stamp.py verify` re-checks it. |
| Night, rain? | Not tested. We only claim what we measured on this clip. |
| What did you prepare before today? | The pipeline and model files, as the organisers allowed. Every number shown was measured today on this Pi. |
