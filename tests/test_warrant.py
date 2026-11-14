"""IRC:103 crossing check on synthetic tracks. Stdlib only, no model needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import warrant as W

# 1000 x 1000 frame, road is the band y 400..600; walkers cross it top to bottom, vehicles drive left to right
ZONES = {"road": [[0, 0.4], [1, 0.4], [1, 0.6], [0, 0.6]],
         "cross_line": [[0, 0.5], [1, 0.5]],          # along the middle of the road
         "line": [[0.5, 0.3], [0.5, 0.7]]}            # across the road


def walker(t0, x=300, wait=0.0, h=80):
    """Stands still at the kerb (y 380) for `wait` s, then walks to y 700 in 10 s. 1 sample per second."""
    pts = [(t0 + k, (x, 380), h) for k in range(int(wait) + 1)]
    tw = pts[-1][0]
    pts += [(tw + k, (x, 380 + 32 * k), h) for k in range(1, 11)]
    return {"name": "person", "pts": pts, "riding": 0}


def car(t0, y=500):
    return {"name": "car", "pts": [(t0 + k, (100 * k, y), 60) for k in range(11)], "riding": 0}


def ev(T, t1, **kw):
    return W.evaluate(T, ZONES, 1000, 1000, 0.0, t1, **kw)


def test_pv2_scaled_to_an_hour():
    # 6 minutes: 30 walkers and 60 cars -> 300 and 600 per hour -> PV2 = 300 * 600^2 = 1.08e8
    T = {i: walker(i * 10) for i in range(30)}
    T.update({100 + i: car(i * 5) for i in range(60)})
    r = ev(T, 360.0)
    i = r["criteria"]["i_pv2"]
    assert (i["P_per_h"], i["V_per_h"]) == (300, 600) and abs(i["PV2"] - 1.08e8) < 1
    assert i["met"] and r["met"] and r["provisional"] and i["basis"].startswith("screening only")
    assert not ev(T, 360.0, divided=True)["criteria"]["i_pv2"]["met"]          # divided road: limit 2e8


def test_wait_and_level_of_service():
    r = ev({1: walker(0, wait=20), 2: walker(100, wait=0)}, 3600.0)
    w = r["criteria"]["iii_wait"]
    assert w["walkers_timed"] == 2 and w["mean_s"] == 10.0 and w["level_of_service"] == "B"
    assert w["lower_bounds"] == 2 and w["met"] is False   # both tracks start already standing: lower bounds


def test_long_wait_meets_rule_iii():
    w = ev({1: walker(0, wait=100)}, 3600.0)["criteria"]["iii_wait"]
    assert w["level_of_service"] == "F" and w["met"] is True


def test_rider_is_not_a_walker():
    # a person box inside a motorcycle box, both moving down across the road: one vehicle, zero walkers
    dets, times = {}, {}
    for k in range(10):
        y = 380 + 32 * k
        dets[f"f{k}"] = {"w": 1000, "h": 1000, "boxes": [["person", 0.9, 300, y - 60, 340, y - 10, 1],
                                                          ["motorcycle", 0.9, 290, y - 40, 350, y, 2]]}
        times[f"f{k}"] = float(k)
    T = W.tracks(dets, times)
    assert T[1]["riding"] == 10
    assert ev(T, 3600.0)["criteria"]["i_pv2"]["P_per_h"] == 0


def test_peak_hour_is_the_busiest_window():
    # 2 hours: quiet first hour (2 walkers), busy second hour (20 walkers, 20 cars)
    T = {i: walker(i * 1000) for i in range(2)}
    T.update({10 + i: walker(3700 + i * 100) for i in range(20)})
    T.update({100 + i: car(3700 + i * 100) for i in range(20)})
    r = ev(T, 7200.0)
    i = r["criteria"]["i_pv2"]
    assert (i["P_per_h"], i["V_per_h"]) == (20, 20) and i["basis"].startswith("measured")
    assert r["window_s"][0] >= 3600 - 100


def test_los_bands():
    assert [W.los(s) for s in (3, 3.1, 13, 38, 64, 90, 91)] == ["A", "B", "B", "C", "D", "E", "F"]


def test_no_signal_gives_zebra_option():
    r = ev({1: walker(0)}, 3600.0, lanes=4, width_m=7.5, area="school")
    assert not r["met"] and "zebra" in r["options"][0]
    assert any("refuge island" in d for d in r["design"]) and any("marshals" in d for d in r["design"])
    assert any("12 s" in d and "at least 12 s" in d for d in r["design"])


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("ok")
