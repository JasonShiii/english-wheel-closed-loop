"""Final repeat scans of a finished sheet, in the gripper, sheet untouched (physical_runs/README.md: a_hat and rho are the
mean of 3 final scans). Each scan = the driver's baseline mode (scan pose, capture, back to pos_9; no forming, wheel open),
measured with the run's own flat reference and cleaned exactly as in the loop, with the glare check and retakes.

    python robot_dobot\final_repeat_scans.py --run loops\<OutName> --target figures/rig/targets/<T>.npz [--n 3] [--retakes 2]
            [--scripts <pinned Scripts>] [--calib "pc_post_processing/deliverable_092826/calibration results"]

Writes <run>/final_repeat_scans/: scan_k/ (robot run + point clouds), field_scan_k.npz (+_raw), scan_k.png, REPEATABILITY.txt,
final_repeat_scans.json. Run from real_testbed/ on the robot PC."""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RT = os.path.dirname(HERE)
sys.path.insert(0, RT)
from ew_loop import loop as L, measure as ME, state as ST          # noqa: E402
from ew_loop.frames import HeightField                             # noqa: E402
from ew_loop.spec import Sheet                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--target", required=True)
ap.add_argument("--n", type=int, default=3); ap.add_argument("--retakes", type=int, default=2)
ap.add_argument("--scripts", default=os.path.join(os.path.expanduser("~"), "ew_robot_code_pinned", "bbe1380", "Scripts"))
ap.add_argument("--calib", default="pc_post_processing/deliverable_092826/calibration results")
ap.add_argument("--clamp", type=float, default=12.0); ap.add_argument("--margin-top", type=float, default=12.0)
a = ap.parse_args()

log = lambda *x: print(*x, flush=True)                                         # noqa: E731
sheet = Sheet(width_mm=231.0, length_mm=174.0, thickness_mm=1.0, clamp_mm=a.clamp, margin_side_mm=43.4, margin_bottom_mm=0.0,
              margin_top_mm=a.margin_top, pad_offsets_from_left_mm=(6.5, 224.5), pad_diameter_mm=13.0)
L._SHEET_FOR_FIGS[0] = sheet
out = os.path.join(a.run, "final_repeat_scans"); os.makedirs(out, exist_ok=True)
measurer = ME.RigMeasurer(calib_dir=a.calib).load(os.path.join(a.run, "round_00", "reference"))
target = L._rig_tools()[1].robust_clip(HeightField.load(a.target), "target clipped to its robust range")
plant = L.DobotPlant(a.scripts, sheet=sheet, manual_wheel="none")
rows, fields = [], []
t0 = time.time()
for k in range(1, a.n + 1):
    rd = os.path.join(out, f"scan_{k}"); os.makedirs(rd, exist_ok=True)
    pdir = plant.retake(rd, 0)
    fld_raw, pdir, chk = L._measure_with_retakes(measurer.field, pdir, plant, rd, sheet, a.retakes, log, f"repeat scan {k}")
    fld = L._clean(fld_raw, sheet)
    name = os.path.join(out, f"field_scan_{k}")
    L._save_field(fld, name); L._save_field(fld_raw, name + "_raw")
    try:
        L._scan_png(os.path.join(out, f"scan_{k}.png"), measurer.full_view(pdir), fld, sheet, f"final repeat scan {k}")
    except Exception as ex:                                                     # figures never stop the scans
        log(f"  (scan figure not written: {ex})")
    mt = ST.compare(fld, target, True)
    hole = L.scan_holes(fld_raw, sheet)
    rows.append({"scan": k, "field": name + ".npz", "takes": len(chk["takes"]), "glare_ok": chk["ok"], "largest_hole_mm2": hole["largest_hole_mm2"],
                 "coverage": fld.meta.get("coverage_observed"), **{kk: (None if isinstance(v, float) and np.isnan(v) else v) for kk, v in mt.items()}})
    fields.append(fld.z_mm)
    log(f"repeat scan {k}: corr {mt['pattern_corr']:.3f} amp {mt['amp_ratio']:.3f} span {mt['span_cur']:.2f} mm | hole {hole['largest_hole_mm2']:.0f} mm2 "
        f"({len(chk['takes'])} take(s))")
corr = np.array([r["pattern_corr"] for r in rows], float); amp = np.array([r["amp_ratio"] for r in rows], float)
span = np.array([r["span_cur"] for r in rows], float)
Z = np.stack(fields); pw = float(np.nanmean(np.nanstd(Z, axis=0, ddof=1))) if len(rows) > 1 else float("nan")
sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")                                  # noqa: E731
summ = {"n_scans": len(rows), "corr_mean": float(corr.mean()), "corr_sd": sd(corr), "amp_mean": float(amp.mean()), "amp_sd": sd(amp),
        "span_mean_mm": float(span.mean()), "span_sd_mm": sd(span), "pointwise_std_mm": pw, "glare_free": int(sum(r["glare_ok"] for r in rows)),
        "secs": round(time.time() - t0, 1), "target": os.path.abspath(a.target), "run": os.path.abspath(a.run)}
json.dump({"scans": rows, "summary": summ}, open(os.path.join(out, "final_repeat_scans.json"), "w"), indent=1, default=str)
with open(os.path.join(out, "REPEATABILITY.txt"), "w") as f:
    f.write("Final repeat scans of the finished sheet in the gripper (sheet untouched); scans with a glare hole > "
            f"{L.GLARE_HOLE_MM2:g} mm2 were retaken (max {a.retakes}).\n")
    for r in rows:
        f.write(f"repeat scan {r['scan']}   corr {r['pattern_corr']:.3f} amp {r['amp_ratio']:.3f} span {r['span_cur']:.2f} mm | glare hole "
                f"{r['largest_hole_mm2']:5.0f} mm2 | coverage {r['coverage']} | takes {r['takes']}\n")
    f.write(f"glare-free scans: {summ['glare_free']} of {summ['n_scans']}\n")
    f.write(f"mean +- sample SD: corr {summ['corr_mean']:.3f} +- {summ['corr_sd']:.3f}, amp {summ['amp_mean']:.3f} +- {summ['amp_sd']:.3f}, "
            f"span {summ['span_mean_mm']:.2f} +- {summ['span_sd_mm']:.2f} mm\nscan-to-scan pointwise std {pw:.2f} mm; {summ['secs']} s\n")
log(open(os.path.join(out, "REPEATABILITY.txt")).read())
