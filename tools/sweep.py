"""T2: choose thresholds / NMS / min-height from data, not guesses.

  python tools/sweep.py                       every bundle/models/*.pte on devset/frames, truth from devset/truth.csv
  python tools/sweep.py --proxy yolo11m_fp32_640   no human counts yet: use a PROXY reference (see below)

Raw candidates (every box scoring > 0.05, before NMS) are cached per model in devset/cache/<model>.npz, for the
full frame and for the 5-pass tiling, so re-running the sweep costs seconds.

truth.csv columns: frame,person,vehicle,split  (split = tune | check). Settings are chosen on `tune`,
the reported error is on `check`. Ties go to the higher threshold.

PROXY MODE. With no human counts the reference is the strongest model run with tiling at a fixed sane setting.
That is NOT truth: it is good for picking NMS/min-height/thresholds for one model, and says nothing honest about
which model is better (the proxy model always wins its own test). Every table is labelled accordingly.
"""
import argparse, csv, itertools, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runner

ROOT = Path(__file__).resolve().parent.parent
TH = [round(x, 2) for x in np.arange(0.15, 0.501, 0.05)]
IOUS = [0.45, 0.50, 0.60, 0.70]
MINH = [0.0, 0.02, 0.04]
VEH = {"car", "bus", "truck", "motorcycle", "bicycle"}


def cache_model(pte, frames, cdir):
    """{frame: {'full': cand, 'tiled': cand, 'h': int}} loaded from or written to the .npz cache."""
    cdir.mkdir(parents=True, exist_ok=True)
    f = cdir / f"{pte.stem}.npz"
    have = {}
    if f.exists():
        z = np.load(f, allow_pickle=False)
        names = set(z["names"].tolist())
        if names >= {p.stem for p in frames}:
            for p in frames:
                have[p.stem] = {k: (z[f"{p.stem}|{k}|xy"], z[f"{p.stem}|{k}|sc"], z[f"{p.stem}|{k}|ci"])
                                for k in ("full", "tiled")} | {"h": int(z[f"{p.stem}|h"])}
            return have
    det = runner.ExecutorchDetector(pte)
    save = {}
    for p in frames:
        img = cv2.imread(str(p))
        full, _ = det.raw(img, 0.05)
        tiled, _, _ = runner.tiled_candidates(det, img, 0.05)
        have[p.stem] = {"full": full, "tiled": tiled, "h": img.shape[0]}
        for k, c in (("full", full), ("tiled", tiled)):
            save[f"{p.stem}|{k}|xy"], save[f"{p.stem}|{k}|sc"], save[f"{p.stem}|{k}|ci"] = c
        save[f"{p.stem}|h"] = np.array(img.shape[0])
        print(f"  cached {pte.stem} {p.name}", flush=True)
    np.savez_compressed(f, names=np.array([p.stem for p in frames]), **save)
    return have


def counts(entry, variant, cp, cv, iou, grouped, minh):
    d = runner.postprocess(entry[variant], cp, cv, iou, grouped, minh, entry["h"])
    if variant == "tiled":
        d = runner.containment_filter(d)
    n = [x[0] for x in d]
    return sum(1 for x in n if x == "person"), sum(1 for x in n if x in VEH)


def evaluate(entries, truth, variant):
    """Rows of (cp, cv, iou, grouped, minh, mae_p, mae_v, bias_p, bias_v, within2) per split."""
    frames = [f for f in truth if f in entries]
    res = {}
    # people depend on (cp, iou, grouped, minh); vehicles on (cv, iou, grouped, minh): sweep them apart
    for grouped, iou, minh in itertools.product((False, True), IOUS, MINH):
        cacheP, cacheV = {}, {}
        for t in TH:
            cacheP[t] = {f: counts(entries[f], variant, t, 1.1, iou, grouped, minh)[0] for f in frames}
            cacheV[t] = {f: counts(entries[f], variant, 1.1, t, iou, grouped, minh)[1] for f in frames}
        res[(grouped, iou, minh)] = (cacheP, cacheV)
    return res


def score(split_frames, truth, pred, which):
    err = np.array([pred[f] - truth[f][which] for f in split_frames], float)
    return float(np.abs(err).mean()), float(err.mean()), int((np.abs(err) <= 2).sum())


def best(res, truth, tune, check):
    out = []
    for key, (cP, cV) in res.items():
        # choose per class on tune; higher threshold wins ties
        bp = min(TH, key=lambda t: (score(tune, truth, cP[t], 0)[0], -t))
        bv = min(TH, key=lambda t: (score(tune, truth, cV[t], 1)[0], -t))
        mp, bip, w2p = score(check, truth, cP[bp], 0)
        mv, biv, w2v = score(check, truth, cV[bv], 1)
        tp, _, _ = score(tune, truth, cP[bp], 0)
        tv, _, _ = score(tune, truth, cV[bv], 1)
        out.append(dict(grouped=key[0], iou=key[1], minh=key[2], cp=bp, cv=bv, tune=tp + tv,
                        mae_p=mp, mae_v=mv, bias_p=bip, bias_v=biv, w2p=w2p, w2v=w2v))
    out.sort(key=lambda r: (r["tune"], -r["cp"] - r["cv"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "devset/frames"))
    ap.add_argument("--truth", default=str(ROOT / "devset/truth.csv"))
    ap.add_argument("--models", nargs="*", help="names without .pte (default: all in bundle/models)")
    ap.add_argument("--proxy", help="model name used as PROXY reference when truth.csv does not exist")
    ap.add_argument("--out", default=str(ROOT / "devset/sweep.md"))
    a = ap.parse_args()

    frames = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    ptes = sorted((ROOT / "bundle/models").glob("*.pte"))
    if a.models:
        ptes = [p for p in ptes if p.stem in a.models]
    cdir = ROOT / "devset/cache"
    entries = {p.stem: cache_model(p, frames, cdir) for p in ptes}

    names = [p.stem for p in frames]
    proxy = None
    trows = []
    if Path(a.truth).exists():
        trows = [r for r in csv.DictReader(l for l in open(a.truth) if not l.startswith("#"))
                 if r["person"].strip() != "" and r["vehicle"].strip() != ""]
        if not trows:
            print("devset/truth.csv has no counts yet -> PROXY mode")
    if trows:
        truth = {r["frame"]: (int(r["person"]), int(r["vehicle"]), r["split"]) for r in trows}
        tune = [f for f in truth if truth[f][2] == "tune"]
        check = [f for f in truth if truth[f][2] == "check"]
        truth = {f: truth[f][:2] for f in truth}
        label = "HUMAN TRUTH"
    else:
        proxy = a.proxy or max(entries, key=lambda m: Path(ROOT / f"bundle/models/{m}.pte").stat().st_size)
        label = f"PROXY reference = {proxy} + tiling @ 0.35/0.35, NMS 0.5 grouped (NOT human truth)"
        truth = {f: counts(entries[proxy][f], "tiled", 0.35, 0.35, 0.5, True, 0.02) for f in names}
        tune = [f for i, f in enumerate(names) if i % 3 != 2]       # 2 of 3 tune, 1 of 3 check
        check = [f for i, f in enumerate(names) if i % 3 == 2]
    print(f"\nreference: {label}\nframes: {len(names)}  tune {len(tune)}  check {len(check)}")

    md = [f"# Sweep ({label})\n", f"{len(tune)} tune / {len(check)} check frames. "
          "Settings chosen on tune; the numbers below are on check. Ties go to the higher threshold.\n",
          "| model | variant | grouped | iou | min_h | conf_person | conf_vehicle | MAE people | MAE vehicles | "
          "bias p/v | within ±2 (p/v) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    winners = {}
    for m, ent in entries.items():
        for variant in ("full", "tiled"):
            if proxy and m == proxy and variant == "tiled":
                continue      # that is the reference itself
            res = evaluate(ent, truth, variant)
            r = best(res, truth, tune, check)[0]
            winners[(m, variant)] = r
            md.append(f"| {m} | {variant} | {r['grouped']} | {r['iou']} | {r['minh']} | {r['cp']} | {r['cv']} | "
                      f"{r['mae_p']:.2f} | {r['mae_v']:.2f} | {r['bias_p']:+.2f}/{r['bias_v']:+.2f} | "
                      f"{r['w2p']}/{len(check)} , {r['w2v']}/{len(check)} |")
    # the untouched legacy setting for comparison
    md += ["", "## Legacy setting (per-class NMS, IoU 0.7, conf 0.30/0.30, min_h 0), same check frames", "",
           "| model | MAE people | MAE vehicles | bias p/v |", "|---|---|---|---|"]
    for m, ent in entries.items():
        pp = {f: counts(ent[f], "full", 0.30, 0.30, 0.7, False, 0.0) for f in check}
        mp, bp_, _ = score(check, truth, {f: v[0] for f, v in pp.items()}, 0)
        mv, bv_, _ = score(check, truth, {f: v[1] for f, v in pp.items()}, 1)
        md.append(f"| {m} | {mp:.2f} | {mv:.2f} | {bp_:+.2f}/{bv_:+.2f} |")
    if proxy:
        md += ["", "> Proxy mode: a model's error against a reference it did not make says how far a cheap setting "
               "is from the expensive one, not how far it is from reality. Replace with devset/truth.csv."]
    Path(a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
