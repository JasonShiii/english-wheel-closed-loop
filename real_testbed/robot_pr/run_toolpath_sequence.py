"""
run_toolpath_sequence.py
------------------------
Robot side of the closed loop. Executes a toolpath.json (an ordered list of zigzags with
pass counts, written by the model side, ew_loop) on the UR5e with the demo's forming routine
(zigzag_forming.run_zigzag_forming: passes back and forth without lifting the wheel, a point
cloud after every pass, wheel unloaded after the last pass of each zigzag), and takes the flat
reference capture the measurement needs.

  python run_toolpath_sequence.py baseline --out DIR
      sheet gripped, wheel open: HOME -> centering -> pos_9, capture DIR/baseline/
      (the reference for locating the sheet in the cameras; once per sheet, before forming)
  python run_toolpath_sequence.py run TOOLPATH.json --out DIR [--yes]
      execute; asks before each zigzag unless --yes
  python run_toolpath_sequence.py plan TOOLPATH.json --out DIR
      no robot: safe-zone check, preview PNGs and the commanded-path CSV of every pass

toolpath.json (schema 1)
  sheet        width_mm, length_mm, thickness_mm, clamp_mm (= SHEET_BOTTOM_TO_WHEEL_LINE_MM),
               margin_side_mm, margin_bottom_mm -> applied to zigzag_forming's constants
  toolpaths[]  id, passes, pivots_xy_mm (zigzag corner points in plan_zigzag()'s frame:
               origin = middle of the wheel contact line, X across the sheet, Y along the
               feed; these are what gets executed), robot {zigzag_lv_mm, zigzag_alpha_deg,
               zigzag_N, rotation_deg, center_x_mm, start_mm} (equivalent plan_zigzag()
               parameters, for the preview title and run_info.txt only)

Output of `run`: DIR/toolpath.json (copy) and one folder per zigzag, DIR/tp<i>_<id>/, with
zigzag_preview.png, run_info.txt, zigzag_path_pass<k>.csv, pointcloud_pass<k>/ and done.flag.

Needs the small change to zigzag_forming.run_zigzag_forming() in
zigzag_forming_toolpath_api.patch (optional pivots_2d / num_passes arguments).
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import zigzag_forming as zf
from zigzag_planner import plot_zigzag, interpolate_path, point_in_safe_zone, safe_zone_bounds
import run_output

SCHEMA_VERSION = 1

# Same pose as part2_english_wheel_forming.py's centering_position (approach waypoint
# between HOME and pos_9).
CENTERING_POSITION = [0.5407457298775096, -0.051720560566184584, 0.40054418028270794,
                      1.1609291717861003, 1.2104751792264261, 1.2701928927639679]


def load_toolpath(path):
    doc = json.load(open(path))
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {doc.get('schema_version')!r} != {SCHEMA_VERSION}")
    for k in ("width_mm", "length_mm", "clamp_mm"):
        if k not in doc["sheet"]:
            raise ValueError(f"{path}: sheet.{k} missing")
    if not doc["toolpaths"]:
        raise ValueError(f"{path}: no toolpaths")
    for tp in doc["toolpaths"]:
        if int(tp["passes"]) < 1 or len(tp["pivots_xy_mm"]) < 2:
            raise ValueError(f"{path}: toolpath {tp.get('id')} needs passes >= 1 and >= 2 pivots")
    return doc


def apply_sheet(sheet):
    """Sheet block -> zigzag_forming's geometry constants (read as module globals at call time)."""
    zf.X_DIM_MM = float(sheet["width_mm"])
    zf.Y_DIM_MM = float(sheet["length_mm"])
    zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM = float(sheet["clamp_mm"])
    zf.Y_DIM_EFFECTIVE_MM = zf.Y_DIM_MM - zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM
    zf.MARGIN_SIDE_MM = float(sheet.get("margin_side_mm", zf.MARGIN_SIDE_MM))
    zf.MARGIN_BOTTOM_MM = float(sheet.get("margin_bottom_mm", zf.MARGIN_BOTTOM_MM))


def apply_zigzag_params(tp):
    """robot block -> zigzag_forming's zigzag constants. Cosmetic (preview title, run_info):
    the executed geometry is the pivot list."""
    r = tp.get("robot", {})
    for src, dst in (("zigzag_lv_mm", "ZIGZAG_LV_MM"), ("zigzag_alpha_deg", "ZIGZAG_ALPHA_DEG"),
                     ("zigzag_N", "ZIGZAG_N"), ("rotation_deg", "ZIGZAG_ROTATION_DEG"),
                     ("start_mm", "ZIGZAG_START_MM")):
        if src in r:
            setattr(zf, dst, r[src])
    zf.NUM_PASSES = int(tp["passes"])


def oob_pivots(pivots):
    return [p for p in pivots
            if not point_in_safe_zone(p, zf.X_DIM_MM, zf.Y_DIM_EFFECTIVE_MM, zf.MARGIN_SIDE_MM, zf.MARGIN_BOTTOM_MM)]


def write_preview(tp, pivots, path):
    dense = interpolate_path(pivots, zf.ZIGZAG_RESOLUTION_MM)
    r = tp.get("robot", {})
    plot_zigzag(
        x_dim=zf.X_DIM_MM, y_dim=zf.Y_DIM_MM,
        waypoints_2d=dense, waypoints_3d=[(x, y, 0.0) for x, y in dense],
        zigzag_lv_mm=r.get("zigzag_lv_mm", 0), zigzag_alpha_deg=r.get("zigzag_alpha_deg", 0),
        zigzag_N=r.get("zigzag_N", 0), rotation_deg=r.get("rotation_deg", 0), arc_radius_mm=0,
        pivots_2d=pivots, wheel_z_mm=zf.WHEEL_Z_OFFSET * 500,
        margin_side=zf.MARGIN_SIDE_MM, margin_bottom=zf.MARGIN_BOTTOM_MM, save_path=str(path),
        sheet_bottom_offset_mm=zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM, safe_zone_y_dim=zf.Y_DIM_EFFECTIVE_MM,
    )


def plan_only(tp, pivots):
    """Build the base-frame motion plan of every pass (forward/reverse alternating, as
    run_zigzag_forming does) and export the commanded-path CSVs — pure geometry, no robot."""
    n_legs = n_pivots = 0
    for pass_num in range(1, int(tp["passes"]) + 1):
        curr = pivots[::-1] if pass_num % 2 == 0 else pivots
        plan, _thetas = zf._build_leveled_motion_plan(curr, csv_path=f"zigzag_path_pass{pass_num}.csv")
        n_legs += sum(1 for s in plan if s[0] == "leg")
        n_pivots += sum(1 for s in plan if s[0] == "pivot")
    return n_legs, n_pivots


def _go_to_pos9(env, ROBOT_HOST, ACCELERATION, VELOCITY, target_start_0):
    print("Moving to home position...")
    env.startup_movel(ROBOT_HOST, target_start_0, a=ACCELERATION, v=VELOCITY)
    env.wait_until_stopped(settle=0.5)
    print("Moving to centering position...")
    env.startup_movel(ROBOT_HOST, CENTERING_POSITION, a=ACCELERATION, v=VELOCITY)
    env.wait_until_stopped(settle=0.25)
    print("Moving to forming start position (pos_9)...")
    env.startup_movel(ROBOT_HOST, zf.TARGET_POS_9, a=ACCELERATION, v=VELOCITY)
    env.wait_until_stopped(settle=0.5)


def cmd_baseline(a):
    from robot_setup import (setup_robot_environment, teardown_robot_environment,
                             ROBOT_HOST, ACCELERATION, VELOCITY, target_start_0)
    env = setup_robot_environment(output_dir=a.out, tag="toolpath_baseline", enable_apriltag=False)
    try:
        _go_to_pos9(env, ROBOT_HOST, ACCELERATION, VELOCITY, target_start_0)
        zf._output_dir = str(a.out)
        if zf._capture_pointcloud_snapshot("baseline") is None:
            raise RuntimeError("baseline capture produced no point cloud")
        print(f"Flat reference capture -> {Path(a.out) / 'baseline'} (robot at pos_9, wheel open)")
    finally:
        teardown_robot_environment(env)


def cmd_run_or_plan(a, robot):
    doc = load_toolpath(a.toolpath)
    apply_sheet(doc["sheet"])
    toolpaths = doc["toolpaths"]
    s = doc["sheet"]
    print(f"Sheet {s['width_mm']} x {s['length_mm']} x {s.get('thickness_mm', '?')} mm, "
          f"wheel line {s['clamp_mm']} mm from the clamped edge -> usable Y 0..{zf.Y_DIM_EFFECTIVE_MM:.1f} mm; "
          f"{len(toolpaths)} toolpath(s), {sum(int(t['passes']) for t in toolpaths)} passes in total")
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    env = None
    if robot:
        from robot_setup import (setup_robot_environment, teardown_robot_environment,
                                 ROBOT_HOST, ACCELERATION, VELOCITY, target_start_0)
    if Path(a.toolpath).resolve() != (out_root / "toolpath.json").resolve():
        shutil.copy(a.toolpath, out_root / "toolpath.json")

    # ---- pre-flight: every toolpath is checked and previewed BEFORE any motion ----------
    subdirs, pivot_lists = [], []
    for i, tp in enumerate(toolpaths):
        pivots = [(float(x), float(y)) for x, y in tp["pivots_xy_mm"]]
        sub = out_root / f"tp{i}_{tp.get('id', '')}"
        sub.mkdir(parents=True, exist_ok=True)
        apply_zigzag_params(tp)
        write_preview(tp, pivots, sub / "zigzag_preview.png")
        oob = oob_pivots(pivots)
        if oob:
            x0, x1, y0, y1 = safe_zone_bounds(zf.X_DIM_MM, zf.Y_DIM_EFFECTIVE_MM, zf.MARGIN_SIDE_MM, zf.MARGIN_BOTTOM_MM)
            raise RuntimeError(
                f"toolpath {tp.get('id')}: {len(oob)}/{len(pivots)} pivots outside the safe zone "
                f"X[{x0:.1f},{x1:.1f}] Y[{y0:.1f},{y1:.1f}] — nothing executed. Preview: {sub / 'zigzag_preview.png'}")
        run_output.write_run_info(
            sub, **{"Toolpath": tp.get("id", ""), "Passes": int(tp["passes"]), "Pivots": len(pivots),
                    **{f"robot.{k}": v for k, v in tp.get("robot", {}).items()},
                    **{f"ours.{k}": v for k, v in tp.get("ours", {}).items()},
                    "Sheet": json.dumps(s)})
        subdirs.append(sub); pivot_lists.append(pivots)
        print(f"  {tp.get('id')}: {len(pivots)} pivots, {int(tp['passes'])} pass(es) — inside the safe zone; "
              f"preview {sub / 'zigzag_preview.png'}")

    # ---- execute (or plan) one toolpath after another ---------------------------------
    try:
        if robot:
            env = setup_robot_environment(output_dir=out_root, tag="toolpath_seq", enable_apriltag=False)
            _go_to_pos9(env, ROBOT_HOST, ACCELERATION, VELOCITY, target_start_0)
        for i, (tp, sub, pivots) in enumerate(zip(toolpaths, subdirs, pivot_lists)):
            apply_zigzag_params(tp)
            zf._output_dir = str(sub)       # per-toolpath folder for CSVs / point clouds / preview
            print(f"\n=== Toolpath {i + 1}/{len(toolpaths)}: {tp.get('id')} — {int(tp['passes'])} pass(es) ===")
            if not robot:
                n_legs, n_piv = plan_only(tp, pivots)
                print(f"  planned: {n_legs} legs, {n_piv} pivot rotations over {int(tp['passes'])} pass(es); CSVs in {sub}")
                continue
            if not a.yes:
                ans = input(f"  Execute {tp.get('id')} ({int(tp['passes'])} pass(es))? [y/N] ").strip().lower()
                if ans != "y":
                    print("  Stopped by operator before this toolpath.")
                    break
            zf.run_zigzag_forming(pivots_2d=pivots, num_passes=int(tp["passes"]))
            (sub / "done.flag").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            print(f"  Toolpath {tp.get('id')} done; wheel unloaded.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if env is not None:
            teardown_robot_environment(env)
    print(f"\nOutput folder: {out_root}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Robot side of the closed loop (see the module docstring).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline"); b.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("toolpath"); r.add_argument("--out", required=True)
    r.add_argument("--yes", action="store_true", help="do not ask before each zigzag")
    p = sub.add_parser("plan"); p.add_argument("toolpath"); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "baseline":
        cmd_baseline(a)
    else:
        cmd_run_or_plan(a, robot=(a.cmd == "run"))


if __name__ == "__main__":
    main()
