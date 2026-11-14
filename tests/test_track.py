"""Tracker: unique counts and line crossings on synthetic moving boxes. No model needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import track

LINE = [(0, 100), (400, 100)]          # drawn left -> right: moving down the image is "in"


def walk(x, y0, y1, steps, name="person", w=40, h=80):
    """Box whose foot point moves from y0 to y1 over `steps` frames."""
    return [(name, 0.9, (x, y - h, x + w, y)) for y in (y0 + (y1 - y0) * k / (steps - 1) for k in range(steps))]


def run(tr, *paths):
    n = max(len(p) for p in paths)
    for k in range(n):
        tr.update([p[k] for p in paths if k < len(p)])
    return tr.totals()


def test_one_walker_counts_once_in():
    t = run(track.Tracker(LINE), walk(100, 60, 140, 9))
    assert t["people_unique"] == 1 and t["people_in"] == 1 and t["people_out"] == 0


def test_walking_back_is_out():
    t = run(track.Tracker(LINE), walk(100, 140, 60, 9))
    assert t["people_in"] == 0 and t["people_out"] == 1


def test_two_people_and_a_car_apart():
    t = run(track.Tracker(LINE), walk(20, 60, 140, 9), walk(250, 60, 140, 9), walk(320, 140, 60, 9, "car", 70, 50))
    assert t["people_unique"] == 2 and t["vehicles_unique"] == 1
    assert t["people_in"] == 2 and t["vehicles_out"] == 1


def test_short_flicker_is_not_unique():
    t = run(track.Tracker(LINE), walk(100, 60, 62, 2))          # seen in 2 frames only
    assert t["people_unique"] == 0


def test_gap_keeps_the_same_track():
    path = walk(100, 60, 140, 9)
    tr = track.Tracker(LINE)
    for k, d in enumerate(path):
        tr.update([] if k in (3, 4) else [d])                   # missed twice, then found again
    assert tr.totals()["people_unique"] == 1 and tr.totals()["people_in"] == 1


def test_jitter_on_the_line_counts_at_most_once():
    jitter = [("person", 0.9, (100, 20 + d, 140, 100 + d)) for d in (-2, 2, -2, 2, -2, 2)]
    t = run(track.Tracker(LINE), jitter)
    assert t["people_in"] + t["people_out"] <= 1


def test_no_line_still_counts_unique():
    t = run(track.Tracker(None), walk(100, 60, 140, 9))
    assert t["people_unique"] == 1 and t["people_in"] == 0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("ok")
