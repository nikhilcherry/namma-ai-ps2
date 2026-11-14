"""decode_scores: group scoring keeps a vehicle whose score is split across car/bus/truck. No model needed."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runner

COL = {cid: i for i, cid in enumerate(runner.IDS)}


def row(**scores):
    r = np.zeros(len(runner.IDS), np.float32)
    for name, v in scores.items():
        r[COL[{"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}[name]]] = v
    return r


def test_default_drops_split_score_vehicle():
    cls = np.stack([row(car=0.20, truck=0.18)])
    keep, score, cid = runner.decode_scores(cls, floor=0.25)
    assert len(keep) == 0


def test_group_score_keeps_it_with_best_class_label():
    cls = np.stack([row(car=0.20, truck=0.18)])
    keep, score, cid = runner.decode_scores(cls, floor=0.25, group_score=True)
    assert len(keep) == 1 and abs(score[0] - 0.38) < 1e-6 and cid[0] == 2


def test_group_score_never_touches_people_and_clips_at_one():
    cls = np.stack([row(person=0.30, car=0.20), row(car=0.9, bus=0.5, truck=0.4)])
    keep, score, cid = runner.decode_scores(cls, floor=0.05, group_score=True)
    assert list(cid) == [0, 2]
    assert abs(score[0] - 0.30) < 1e-6          # person: the person score, car 0.20 not mixed in
    assert score[1] == 1.0


def test_bike_group_sums_bicycle_and_motorcycle():
    cls = np.stack([row(motorcycle=0.2, bicycle=0.15)])
    keep, score, cid = runner.decode_scores(cls, floor=0.3, group_score=True)
    assert len(keep) == 1 and cid[0] == 3


if __name__ == "__main__":
    for _n, _f in sorted(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f()
            print("ok", _n)
