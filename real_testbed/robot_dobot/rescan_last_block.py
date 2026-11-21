"""Re-measure the state after the last executed block of a server-final run (2026-11-21: aluminium S7, the wheel scraped the
matte coat off the rolled strips, the block scan had glare holes inside the ridge and a spurious bump after cleaning; the sheet
was re-sprayed in the gripper). One scan in the driver's baseline mode (scan pose, capture, back to pos_9; no forming), measured
with the run's own flat reference and cleaned as in the loop, with the glare check and retakes. Writes
round_NN/field_<tag>.npz/.png (+_raw), appends the measurement to round_NN/round.json["measured"] with a note, and points
state.json["last_field"] at the new field so `-Resume` re-asks the controller for round NN+1 (the service restarts that round
from its state_before_round file). The original scan and its files are kept.

    python robot_dobot\rescan_last_block.py --run loops\<OutName> --target figures/rig/targets/<T>.npz --round 1 --tag TP0_capture1_recoated
"""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RT = os.path.dirname(HERE); sys.path.insert(0, RT)
from ew_loop import loop as L, measure as ME, state as ST          # noqa: E402
from ew_loop.frames import HeightField                             # noqa: E402
from ew_loop.spec import Sheet                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--target", required=True); ap.add_argument("--round", type=int, required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--retakes", type=int, default=2); ap.add_argument("--note", default="re-scan after re-spraying the rolled strips (matte coat scraped off by the wheel)")
ap.add_argument("--scripts", default=os.path.join(os.path.expanduser("~"), "ew_robot_code_pinned", "bbe1380", "Scripts"))
ap.add_argument("--calib", default="pc_post_processing/deliverable_092826/calibration results")
a = ap.parse_args()
log = lambda *x: print(*x, flush=True)                                          # noqa: E731
sheet = Sheet(width_mm=231.0, length_mm=174.0, thickness_mm=1.0, clamp_mm=12.0, margin_side_mm=43.4, margin_bottom_mm=0.0,
              margin_top_mm=12.0, pad_offsets_from_left_mm=(6.5, 224.5), pad_diameter_mm=13.0)
L._SHEET_FOR_FIGS[0] = sheet
rd = os.path.join(a.run, f"round_{a.round:02d}")
measurer = ME.RigMeasurer(calib_dir=a.calib).load(os.path.join(a.run, "round_00", "reference"))
target = L._rig_tools()[1].robust_clip(HeightField.load(a.target), "target clipped to its robust range")
plant = L.DobotPlant(a.scripts, sheet=sheet, manual_wheel="none")
pdir = plant.retake(rd, 0) if not os.path.exists(os.path.join(rd, "robot_retake0")) else plant._run("baseline", rd, out_name=f"robot_rescan_{a.tag}")["captures"]["0"]
fld_raw, pdir, chk = L._measure_with_retakes(measurer.field, pdir, plant, rd, sheet, a.retakes, log, f"round {a.round} re-scan ({a.tag})")
fld = L._clean(fld_raw, sheet)
name = os.path.join(rd, f"field_{a.tag}")
L._save_field(fld, name); L._save_field(fld_raw, name + "_raw")
try:
    L._scan_png(os.path.join(rd, f"scan_{a.tag}.png"), measurer.full_view(pdir), fld, sheet, f"round {a.round}: re-scan {a.tag}")
except Exception as ex:
    log(f"  (scan figure not written: {ex})")
mt = ST.compare(fld, target, True); hole = L.scan_holes(fld_raw, sheet)
log(f"re-scan: corr {mt['pattern_corr']:.3f} amp {mt['amp_ratio']:.3f} span {mt['span_cur']:.2f} mm | hole {hole['largest_hole_mm2']:.0f} mm2 ({len(chk['takes'])} take(s)) | coverage {fld.meta.get('coverage_observed')}")
rj = os.path.join(rd, "round.json"); rec = json.load(open(rj))
rec.setdefault("measured", []).append({"rescan": True, "note": a.note, "field": name + ".npz", "scan_check": chk, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                       "z_min": round(float(fld.zrange()[0]), 2), "z_max": round(float(fld.zrange()[1]), 2), **mt})
rec["measurement_of_record"] = name + ".npz"
json.dump(rec, open(rj, "w"), indent=1, default=str)
sp = os.path.join(a.run, "state.json"); st = json.load(open(sp))
st["last_field_before_rescan"] = st.get("last_field"); st["last_field"] = name + ".npz"; st["rescan_note"] = a.note
json.dump(st, open(sp, "w"), indent=1, default=str)
with open(os.path.join(rd, "OPERATOR_NOTE.txt"), "a") as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {a.note}. The original block scan is kept (field_*_capture*.npz); the measurement of record "
            f"for the controller's next round is {os.path.basename(name)}.npz (corr {mt['pattern_corr']:.3f}, amp {mt['amp_ratio']:.3f}).\n")
log("state.json last_field ->", name + ".npz", "| resume the loop with -Resume to re-ask the controller for round", a.round + 1)
