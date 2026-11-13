"""Pinned counts on the six sample photos (INT8 Arm model, conf 0.30, IoU 0.7). Skipped without the model/runtime.
x86-64 only: the INT8 kernels differ per architecture, and on emulated ARM64 a borderline box flips
(junction_01: 4 people instead of 5). So laptop counts and Pi counts can differ by one on INT8.
--ungrouped must reproduce the pre-T3 counts exactly; grouped must differ only on signal_06 (7 -> 6 vehicles)."""
import platform, subprocess, sys, csv, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "bundle/models/yolo11n_raspberry_executorch_optimized.pte"
OLD = {"arterial_05": (7, 6), "brigade_03": (13, 2), "egipura_02": (4, 4),
       "junction_01": (5, 10), "majestic_04": (10, 0), "signal_06": (1, 7)}


def run(*flags):
    out = tempfile.mkdtemp()
    subprocess.run([sys.executable, str(ROOT / "main.py"), "--source", str(ROOT / "samples"), "--model", str(MODEL),
                    "--out", out, "--no-addons", "--deploy", *flags], check=True, capture_output=True, cwd=ROOT)
    return {r["frame"]: (int(r["person"]), int(r["vehicle"])) for r in csv.DictReader(open(Path(out) / "counts.csv"))}


def _ok():
    if platform.machine() not in ("x86_64", "AMD64"):
        return False
    try:
        import executorch  # noqa
        return MODEL.exists()
    except Exception:
        return False


def test_ungrouped_matches_old_counts():
    if not _ok():
        return
    assert run("--ungrouped") == OLD


def test_grouped_only_changes_signal_06():
    if not _ok():
        return
    new = run()
    assert {k: v for k, v in new.items() if k != "signal_06"} == {k: v for k, v in OLD.items() if k != "signal_06"}
    assert new["signal_06"] == (1, 6)


if __name__ == "__main__":
    test_ungrouped_matches_old_counts(); test_grouped_only_changes_signal_06(); print("ok")
