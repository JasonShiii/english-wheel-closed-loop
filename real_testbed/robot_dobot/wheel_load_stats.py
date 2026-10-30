"""The wheel load actually applied in one robot run, from the driver's own log (outputs/<run>/wheel_force_*.csv:
time_s, x_mm, z_mm, force_lbf at ~16 Hz, the load cell on AI1) and the threshold in run_info.txt.

Rolling = the samples between the first and the last robot motion while the wheel is loaded (force > half the
threshold): load ramp, pivots at the ends and the unload are outside or at the edges of that window. For a block of
several passes run back to back the window covers the whole block (the CSV does not mark the passes).
The load is a calibration variable (the threshold changed from 300 to 250 N on 2026-10-29), so every run records it.

usage: python wheel_load_stats.py <run folder> [...]   prints the stats as JSON"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

LBF_TO_N = 4.4482216153


def threshold_n(run_dir):
    p = os.path.join(run_dir, "run_info.txt")
    if not os.path.exists(p):
        return None
    m = re.search(r"Wheel Force Threshold \(lbf\):\s*([\d.]+)", open(p, encoding="utf-8", errors="replace").read())
    return round(float(m.group(1)) * LBF_TO_N, 1) if m else None


def load_stats(run_dir):
    """dict with the threshold and the force while rolling (N); None if the run folder has no wheel force log"""
    files = sorted(glob.glob(os.path.join(run_dir, "wheel_force_*.csv")))
    files = [f for f in files if not f.endswith("_path.csv")]
    if not files:
        return None
    t, x, F = [], [], []
    with open(files[0], newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                t.append(float(r["time_s"])); x.append(float(r["x_mm"]) if r.get("x_mm") else np.nan)
                F.append(float(r["force_lbf"]) * LBF_TO_N if r.get("force_lbf") else np.nan)
            except (KeyError, ValueError):
                continue
    t, x, F = np.array(t), np.array(x), np.array(F)
    thr = threshold_n(run_dir)
    out = {"source": os.path.basename(files[0]), "threshold_N": thr, "samples": int(len(t)),
           "peak_N": round(float(np.nanmax(F)), 1) if np.isfinite(F).any() else None}
    if len(t) < 3 or not np.isfinite(F).any():
        return out
    loaded = np.isfinite(F) & (F > 0.5 * (thr if thr else 250.0))
    moving = np.r_[False, np.abs(np.diff(x)) > 0.01]
    idx = np.flatnonzero(loaded & moving)
    if idx.size == 0:
        out["rolling"] = None
        return out
    i0, i1 = int(idx[0]), int(idx[-1])
    w = F[i0:i1 + 1]; w = w[np.isfinite(w)]
    before = F[max(0, i0 - 8):i0]; before = before[np.isfinite(before)]
    out["load_at_roll_start_N"] = round(float(np.median(before)), 1) if before.size else None
    out["rolling"] = {"secs": round(float(t[i1] - t[i0]), 1), "samples": int(w.size),
                      "median_N": round(float(np.median(w)), 1), "mean_N": round(float(np.mean(w)), 1),
                      "p5_N": round(float(np.percentile(w, 5)), 1), "p95_N": round(float(np.percentile(w, 95)), 1),
                      "min_N": round(float(np.min(w)), 1), "max_N": round(float(np.max(w)), 1)}
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps({"run": d, **(load_stats(d) or {"error": "no wheel_force_*.csv"})}))
