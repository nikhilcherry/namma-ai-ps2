"""Crossing check against IRC:103-2012 (Guidelines for Pedestrian Facilities): does this spot need a signal?

  python main.py --source clip.mp4 --profile track --zones zones.json --out out/track
  python tools/warrant.py out/track --zones zones.json [--divided] [--lanes 2] [--width-m 7.5] [--area school] [--injuries 3]
  -> out/track/warrant.json and warrant.md (tools/report.py adds it to the site report)

zones.json needs: "road" (polygon), "cross_line" (2 points ALONG the middle of the road: walkers who cross it crossed
the road; draw with tools/zones.py --cross) and "line" (2 points ACROSS the road: vehicles that cross it passed; --line).

What the camera can check, rule by rule (IRC:103 clause 6.7.4.1.2, "control measures at mid-block crossings"):
  (i)   peak-hour P x V^2 > 1e8 on an undivided road, > 2e8 on a divided one       -> measured here
  (ii)  approach speeds above 65 km/h                                              -> NOT measured (needs calibration)
  (iii) "waiting time for pedestrians becomes too long"                            -> measured; the rule gives no number,
        we read "too long" as Annex I Table 1.2 level of service E or F (> 64 s)   (our reading, printed as such)
  (iv)  5 or more pedestrian injuries in a year                                    -> from police records (--injuries)
Stdlib only, so it runs on any laptop or the Pi without the model environment.
"""
import argparse, csv, json, math
from pathlib import Path

VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}
BIKES = {"motorcycle", "bicycle"}
LOS = [(3, "A"), (13, "B"), (38, "C"), (64, "D"), (90, "E")]        # IRC:103 Annex I Table 1.2, waiting time in s; above 90 = F
SPEED = {"normal": 0.95, "elderly": 0.79, "school": 0.98}             # m/s, IRC:103 6.7.4.1.2 design crossing speeds


def los(wait_s):
    return next((g for lim, g in LOS if wait_s <= lim), "F")


def side(line, p):
    """+1 / -1 side of the line for point p, 0 on it, None when p is past the line's ends (same rule as track.py)."""
    (x1, y1), (x2, y2) = line
    dx, dy = x2 - x1, y2 - y1
    t = ((p[0] - x1) * dx + (p[1] - y1) * dy) / max(1e-9, dx * dx + dy * dy)
    if not 0.0 <= t <= 1.0:
        return None
    c = dx * (p[1] - y1) - dy * (p[0] - x1)
    return (c > 0) - (c < 0)


def inside(poly, p):
    """Ray-casting point-in-polygon."""
    x, y, ins = p[0], p[1], False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            ins = not ins
    return ins


def overlap_frac(a, b):
    """Intersection area / area of a (boxes x1,y1,x2,y2)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def tracks(dets, times):
    """dets: detections.json; times: frame -> t_s. Returns {tid: {"name", "pts": [(t, foot, h)], "rider": share}}."""
    T = {}
    for f in sorted(dets, key=lambda k: times[k]):
        d = dets[f]
        bikes = [b[2:6] for b in d["boxes"] if b[0] in BIKES]
        for b in d["boxes"]:
            if len(b) < 7 or b[6] is None:
                continue                                    # untracked run, or a track not yet confirmed
            name, box = b[0], b[2:6]
            tr = T.setdefault(b[6], {"name": name, "pts": [], "riding": 0})
            tr["pts"].append((times[f], ((box[0] + box[2]) / 2, box[3]), box[3] - box[1]))
            if name == "person" and any(overlap_frac(box, bk) > 0.4 for bk in bikes):
                tr["riding"] += 1                           # same rider rule as insights.pedestrians
    return T


def crossing_time(pts, line):
    """Time the track's foot point first reaches the other side of the line from where it started; None if never."""
    start = None
    for t, foot, _ in pts:
        s = side(line, foot)
        if not s:
            continue
        if start is None:
            start = s
        elif s != start:
            return t
    return None


def wait_before_road(pts, road):
    """(seconds stood still just outside the road before stepping in, truncated?) or None.
    Still = foot point within half a body height of where the walker stood last before entering (scale-aware,
    so it works near and far from the camera). Truncated = the track starts already standing, so it is a lower bound."""
    k = next((i for i, (_, foot, _) in enumerate(pts) if inside(road, foot)), None)
    if not k:
        return None                                         # never on the road, or already on it when first seen
    tk, (kx, ky), _ = pts[k - 1]
    j = k - 1
    while j > 0 and math.dist(pts[j - 1][1], (kx, ky)) <= 0.5 * pts[j - 1][2]:
        j -= 1
    return tk - pts[j][0], j == 0


def peak_hour(p_times, v_times, t0, t1):
    """Busiest 60-minute window by P*V^2 when the clip is at least an hour; otherwise the clip scaled to an hour."""
    dur = max(1e-9, t1 - t0)
    if dur < 3600:
        s = 3600 / dur
        return len(p_times) * s, len(v_times) * s, (t0, t1)
    best = (-1, 0, 0, (t0, t0 + 3600))
    for w0 in sorted({t0, t1 - 3600, *p_times, *v_times}):     # t1 - 3600: the last full hour
        if w0 + 3600 > t1:
            break
        P = sum(w0 <= t < w0 + 3600 for t in p_times); V = sum(w0 <= t < w0 + 3600 for t in v_times)
        if P * V * V > best[0]:
            best = (P * V * V, P, V, (w0, w0 + 3600))
    return best[1], best[2], best[3]


def evaluate(T, zones, w, h, t0, t1, divided=False, lanes=None, width_m=None, area="normal", injuries=None):
    px = lambda pts: [(x * w, y * h) for x, y in pts]
    road = px(zones["road"]) if zones.get("road") else None
    cross = px(zones["cross_line"]) if zones.get("cross_line") else None
    vline = px(zones["line"]) if zones.get("line") else None
    if cross is None:
        raise SystemExit("zones.json has no 'cross_line': draw one along the middle of the road with tools/zones.py --cross")
    walkers = {k: v for k, v in T.items() if v["name"] == "person" and v["riding"] <= len(v["pts"]) / 2}
    p_times = sorted(t for t in (crossing_time(v["pts"], cross) for v in walkers.values()) if t is not None)
    vehs = {k: v for k, v in T.items() if v["name"] in VEHICLES}
    if vline is not None:
        v_times = sorted(t for t in (crossing_time(v["pts"], vline) for v in vehs.values()) if t is not None)
        v_how = "vehicles crossing the counting line"
    else:
        v_times = sorted(v["pts"][0][0] for v in vehs.values())
        v_how = "unique vehicle tracks (no 'line' in zones.json: an upper estimate)"
    two_w = sum(1 for v in vehs.values() if v["name"] in BIKES) / max(1, len(vehs))
    P, V, win = peak_hour(p_times, v_times, t0, t1)
    pv2, lim = P * V * V, (2e8 if divided else 1e8)
    dur = t1 - t0
    tier = ("measured: busiest hour of the recording" if dur >= 3600 else
            f"estimate: {dur / 60:.0f} min scaled to an hour" if dur >= 900 else
            f"screening only: {dur:.0f} s scaled to an hour (the rule needs peak-hour counts)")
    waits, trunc = [], 0
    if road:
        for k, v in walkers.items():
            if crossing_time(v["pts"], cross) is None:
                continue
            r = wait_before_road(v["pts"], road)
            if r:
                waits.append(r[0]); trunc += r[1]
    mean_w = sum(waits) / len(waits) if waits else None
    grade = los(mean_w) if mean_w is not None else None
    c = {
        "i_pv2": {"met": pv2 > lim, "P_per_h": round(P), "V_per_h": round(V), "PV2": pv2, "limit": lim,
                  "times_limit": round(pv2 / lim, 2), "basis": tier},
        "ii_speed": {"met": None, "note": "not measured: needs a speed gun or a calibrated camera"},
        "iii_wait": {"met": None if grade is None else grade in ("E", "F"), "mean_s": None if mean_w is None else round(mean_w, 1),
                     "level_of_service": grade, "walkers_timed": len(waits), "lower_bounds": trunc,
                     "note": "IRC:103 gives no number for 'too long'; we read it as level of service E or F (> 64 s)"},
        "iv_injuries": {"met": None if injuries is None else injuries >= 5, "per_year": injuries},
    }
    met = any(x["met"] for x in c.values())
    provisional = dur < 3600 or not road or vline is None
    if met:
        verdict = "Controlled crossing warranted: traffic signal with a pedestrian phase"
        options = ["IRC:103 6.7.4.1.2: the usual control is a traffic signal with an exclusive pedestrian phase.",
                   "If the warrant is met only in peak hours, police control can be used instead.",
                   "Where vehicle volume is very large even with few pedestrians, a pedestrian-operated (pelican) signal."]
    else:
        verdict = "Signal not warranted on this evidence"
        options = ["Basic option: a marked zebra crossing with 'pedestrian cross' signs and a STOP line (IRC:103 6.7.3.1), "
                   "if the siting rules below hold."]
    if provisional:
        verdict = "Provisional: " + verdict
    design = []
    if width_m:
        s = SPEED[area]
        design.append(f"Walking time across {width_m} m at {s} m/s ({area} design speed, 6.7.4.1.2): {width_m / s:.1f} s; "
                      f"the 12 s per 7.5 m rule (6.7.3.2) gives {12 * width_m / 7.5:.1f} s. "
                      f"Pedestrian red for traffic: at least {max(width_m / s, 12 * width_m / 7.5):.0f} s.")
    if lanes and lanes >= 4:
        design.append("4 or more lanes: a pedestrian refuge island is mandatory, at least 2 m wide (6.7.3.3, Table 4).")
    if area == "school":
        design.append("School zone: traffic marshals are generally required (6.15).")
    design.append("Crossing at least 3 m wide (6.7.2); zebra 2-4 m wide (6.7.3.1); 80 lux lighting at the crossing (6.13).")
    site = ["No other zebra crossing within 150 m (6.7.3.1).",
            "Mid-block: junctions more than 300 m apart and a real demand, e.g. shops or a school (6.7.4.1.1).",
            "Guard-rails long enough to stop crossing at random points; kerb ramps on both sides (6.7.3.1, 6.7.2)."]
    bins = {}
    for t in p_times:
        bins.setdefault(int((t - t0) // 900), [0, 0])[0] += 1
    for t in v_times:
        bins.setdefault(int((t - t0) // 900), [0, 0])[1] += 1
    return {"verdict": verdict, "met": met, "provisional": provisional, "criteria": c, "options": options,
            "design": design, "check_on_site": site, "vehicles_counted_as": v_how, "two_wheeler_share": round(two_w, 2),
            "window_s": [round(win[0], 1), round(win[1], 1)], "duration_s": round(dur, 1),
            "per_15_min": [{"from_min": k * 15, "P": p, "V": v} for k, (p, v) in sorted(bins.items())],
            "source": "IRC:103-2012 Guidelines for Pedestrian Facilities (First Revision), Indian Roads Congress"}


def markdown(r):
    c = r["criteria"]; i, w = c["i_pv2"], c["iii_wait"]
    yn = lambda m: "not checked" if m is None else ("**met**" if m else "not met")
    L = [f"# Crossing check (IRC:103-2012)\n\n**{r['verdict']}**\n",
         "| Rule (clause 6.7.4.1.2) | Measured | Result |", "|---|---|---|",
         f"| (i) P x V² over {i['limit']:.0e} | P = {i['P_per_h']} walkers/h, V = {i['V_per_h']} vehicles/h, "
         f"PV² = {i['PV2']:.2e} ({i['times_limit']}x the limit); {i['basis']} | {yn(i['met'])} |",
         f"| (ii) speeds over 65 km/h | {c['ii_speed']['note']} | not checked |",
         f"| (iii) waiting too long | " + (f"mean {w['mean_s']} s over {w['walkers_timed']} walkers, level of service {w['level_of_service']}"
                                         f" ({w['lower_bounds']} lower bounds); {w['note']}" if w['mean_s'] is not None else "no walker timed")
         + f" | {yn(w['met'])} |",
         f"| (iv) 5+ pedestrian injuries a year | {c['iv_injuries']['per_year'] if c['iv_injuries']['per_year'] is not None else 'not given'} | {yn(c['iv_injuries']['met'])} |",
         "", f"V counts {r['vehicles_counted_as']}; {r['two_wheeler_share']:.0%} of vehicle tracks are two-wheelers, each counted as one vehicle.",
         "", "## What the rulebook offers", *[f"- {o}" for o in r["options"]],
         "", "## Design numbers", *[f"- {d}" for d in r["design"]],
         "", "## Check on site (the camera cannot)", *[f"- {s}" for s in r["check_on_site"]],
         "", "## Per 15 minutes", "| from min | walkers crossing | vehicles |", "|---|---|---|",
         *[f"| {b['from_min']} | {b['P']} | {b['V']} |" for b in r["per_15_min"]],
         "", f"_Limits: walkers crossing outside the view are missed, so P is a lower bound; V is squared, so a 10% vehicle error is a 21% PV² error. "
             f"Tracks link boxes by overlap only. Source: {r['source']}._\n"]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="output folder of a main.py --track run")
    ap.add_argument("--zones", required=True)
    ap.add_argument("--divided", action="store_true", help="divided carriageway (median): limit 2e8 instead of 1e8")
    ap.add_argument("--lanes", type=int, help="lanes to cross (4+ makes a refuge island mandatory)")
    ap.add_argument("--width-m", type=float, help="crossing length in metres, for the pedestrian time")
    ap.add_argument("--area", choices=list(SPEED), default="normal", help="design walking speed: elderly 0.79, school 0.98 m/s")
    ap.add_argument("--injuries", type=int, help="pedestrian injuries here in the last year, from police records")
    a = ap.parse_args()
    run = Path(a.run)
    dets = json.loads((run / "detections.json").read_text())
    with open(run / "counts.csv", encoding="utf-8") as f:
        times = {r["frame"]: float(r["t_s"]) for r in csv.DictReader(f)}
    if not any(len(b) >= 7 for d in dets.values() for b in d["boxes"]):
        raise SystemExit("no track ids in detections.json: re-run main.py with --profile track (or --track) on a video")
    T = tracks(dets, times)
    d0 = next(iter(dets.values()))
    ts = sorted(times.values())
    step = (ts[-1] - ts[0]) / max(1, len(ts) - 1)
    r = evaluate(T, json.loads(Path(a.zones).read_text()), d0["w"], d0["h"], ts[0], ts[-1] + step,
                 a.divided, a.lanes, a.width_m, a.area, a.injuries)
    (run / "warrant.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    (run / "warrant.md").write_text(markdown(r), encoding="utf-8")
    i = r["criteria"]["i_pv2"]
    print(f"{r['verdict']}\nP={i['P_per_h']}/h V={i['V_per_h']}/h PV2={i['PV2']:.2e} ({i['times_limit']}x limit; {i['basis']})"
          f"\nwait: {r['criteria']['iii_wait']['mean_s']} s, level of service {r['criteria']['iii_wait']['level_of_service']}"
          f"\nwrote {run / 'warrant.md'}")
