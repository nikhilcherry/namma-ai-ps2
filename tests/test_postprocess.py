"""Pure-numpy tests for runner.postprocess / containment_filter / insights region. No model needed.
  .venv-twin/bin/python -m pytest tests -q     (or: python tests/test_postprocess.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import insights, runner

CAR, TRUCK, PERSON, MOTO = 2, 7, 0, 3


def cand(rows):
    """rows: (x1,y1,x2,y2,score,cid)"""
    a = np.array(rows, dtype=np.float32).reshape(-1, 6)
    return a[:, :4], a[:, 4], a[:, 5].astype(np.int64)


def names(d):
    return sorted(x[0] for x in d)


def test_same_vehicle_as_car_and_truck_counts_once_when_grouped():
    c = cand([(100, 100, 300, 200, .8, CAR), (102, 101, 301, 200, .6, TRUCK)])
    assert len(runner.postprocess(c, grouped=False)) == 2          # the old double count
    d = runner.postprocess(c, grouped=True)
    assert names(d) == ["car"]                                      # higher score keeps its label


def test_person_and_vehicle_overlap_are_not_merged():
    c = cand([(100, 100, 200, 300, .9, PERSON), (90, 150, 220, 320, .9, MOTO)])
    assert names(runner.postprocess(c, grouped=True)) == ["motorcycle", "person"]


def test_two_thresholds_filter_independently():
    c = cand([(0, 0, 50, 100, .35, PERSON), (200, 0, 300, 100, .35, CAR)])
    assert names(runner.postprocess(c, conf_person=.30, conf_vehicle=.40)) == ["person"]
    assert names(runner.postprocess(c, conf_person=.40, conf_vehicle=.30)) == ["car"]


def test_min_height_share_of_frame():
    c = cand([(0, 0, 10, 15, .9, PERSON), (50, 0, 80, 100, .9, PERSON)])
    assert len(runner.postprocess(c, min_h=0.10, frame_h=1000)) == 1


def test_empty_and_only_person():
    assert runner.postprocess(cand([])) == []
    c = cand([(0, 0, 50, 100, .9, PERSON), (200, 0, 300, 100, .9, CAR)])
    assert names(runner.postprocess(c, only_person=True)) == ["person"]


def test_containment_filter_drops_half_box_keeps_distinct_neighbour():
    big = ("car", .9, (0., 0., 100., 100.))
    half = ("car", .5, (0., 0., 100., 40.))                         # cut by a tile edge, inside the big one
    other = ("car", .5, (200., 0., 300., 100.))
    person_inside = ("person", .8, (10., 10., 50., 90.))            # different group: never suppressed
    out = runner.containment_filter([big, half, other, person_inside])
    assert half not in out and big in out and other in out and person_inside in out


def test_count_region_uses_foot_point():
    poly = np.array([[0, 50], [100, 50], [100, 100], [0, 100]], np.int32)
    inside = ("person", .9, (10., 0., 30., 80.))                    # foot (20, 80) inside
    outside = ("person", .9, (10., 0., 30., 40.))                   # foot (20, 40) above the polygon
    assert insights.in_region([inside, outside], poly) == [inside]
    assert insights.in_region([inside, outside], None) == [inside, outside]


def test_needs_tiles_trigger():
    H = 1000
    empty = cand([])
    assert not runner.needs_tiles(empty, H)
    crowd = cand([(i * 10, 0, i * 10 + 5, 50, .2, PERSON) for i in range(45)])      # 45 borderline boxes
    assert runner.needs_tiles(crowd, H)
    tiny = cand([(0, 0, 10, 40, .9, PERSON)])                                       # person 4% of frame height
    assert runner.needs_tiles(tiny, H)
    big = cand([(0, 0, 100, 500, .9, PERSON), (200, 0, 300, 400, .9, CAR)])         # large objects, sparse
    assert not runner.needs_tiles(big, H)


def test_rolling_median_removes_one_frame_flicker():
    assert insights.rolling_median([5, 5, 0, 5, 5]) == [5, 5, 5, 5, 5]          # one-frame dropout filled
    assert insights.rolling_median([0, 0, 0, 0, 0]) == [0, 0, 0, 0, 0]          # a real zero stays zero
    assert insights.rolling_median([3]) == [3] and insights.rolling_median([]) == []


def test_rolling_median_keeps_real_steps_and_ends():
    out = insights.rolling_median([0, 0, 0, 10, 10, 10, 10])
    assert out[0] == 0 and out[-1] == 10 and out[2] == 0 and out[4] == 10       # step moves by <= 1 frame
    assert all(isinstance(x, int) for x in out)


if __name__ == "__main__":
    fails = 0
    for k, f in sorted(globals().items()):
        if k.startswith("test_"):
            try:
                f(); print("ok  ", k)
            except AssertionError:
                fails += 1; print("FAIL", k)
    sys.exit(fails)
