"""The sweep decides the final thresholds, so its scoring and tie-break must be right."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import sweep


def test_score_mae_bias_within2():
    truth = {"a": (10, 5), "b": (10, 5), "c": (10, 5)}
    mae, bias, w2 = sweep.score(["a", "b", "c"], truth, {"a": 8, "b": 10, "c": 15}, 0)
    assert abs(mae - 7 / 3) < 1e-9 and abs(bias - 1.0) < 1e-9 and w2 == 2     # errors -2, 0, +5


def test_best_prefers_lower_tune_error_then_higher_threshold():
    truth = {f: (5, 5) for f in "abcd"}
    tune, check = ["a", "b"], ["c", "d"]
    # people: thresholds .30 and .40 are both perfect on tune -> the higher (.40) must win; .15 is wrong
    cP = {t: {f: 5 for f in "abcd"} for t in sweep.TH}
    for f in "abcd":
        cP[0.15][f] = 9
    cV = {t: {f: 5 for f in "abcd"} for t in sweep.TH}
    res = {(True, 0.5, 0.0): (cP, cV)}
    r = sweep.best(res, truth, tune, check)[0]
    assert r["cp"] == max(sweep.TH) and r["cv"] == max(sweep.TH) and r["mae_p"] == 0 and r["mae_v"] == 0


def test_best_ranks_settings_by_tune_error():
    truth = {f: (5, 5) for f in "ab"}
    good = {t: {"a": 5, "b": 5} for t in sweep.TH}
    bad = {t: {"a": 9, "b": 1} for t in sweep.TH}
    res = {(False, 0.7, 0.0): (bad, bad), (True, 0.5, 0.0): (good, good)}
    out = sweep.best(res, truth, ["a"], ["b"])
    assert out[0]["grouped"] is True and out[0]["tune"] == 0 and out[1]["tune"] > 0
