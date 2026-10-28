"""Execute one toolpath.json on the Dobot CR5A English-wheel cell through Danish's own driver
(English-Wheel---DoBot-Files/Scripts/run_zigzag_oldclamps.py), without editing any of his files.

    python run_toolpath_dobot.py check    toolpath.json --scripts <Scripts> --out <dir>   # no robot: constants, his safety check, preview
    python run_toolpath_dobot.py baseline               --scripts <Scripts> --out <dir>   # scan pose, flat scan (pointcloud_pass0), no forming
    python run_toolpath_dobot.py run      toolpath.json --scripts <Scripts> --out <dir>   # the block: its passes, a scan after every pass

Before his driver runs, in memory only (zigzag_forming_oldclamps = zf):
  zf.plan_zigzag          -> returns the file's pivots (his planner frame; our geometry equals his plan_zigzag point
                             for point, ew_loop/tests/test_spec.py), interpolated with his own interpolate_path for the preview
  zf.X_DIM_MM, Y_DIM_MM, SHEET_BOTTOM_TO_WHEEL_LINE_MM, MARGIN_SIDE/BOTTOM_MM  <- the file's sheet block; MARGIN_TOP_MM <- --margin-top
                             (his per-pivot safety check then runs on OUR sheet and zone)
  zf.ZIGZAG_*             <- the file's robot block (run_info.txt stays truthful); zf.NUM_PASSES <- the toolpath's passes
  zf._capture_pointcloud_snapshot("pointcloud_pass0") skipped in `run` (the loop measured that state already)
  zf.run_zigzag_forming   -> no-op in `baseline` (his run_step_7_5 then only scans and parks at pos_9)
  zf._prompt_pass_radii   -> kept (the operator enters Rx/Rz between passes as today) unless --auto-radii
  os.startfile            -> no image viewer; wheel_force.plot_last_wheel_force_path -> non-blocking
Then his driver runs as __main__ (connects, outputs/<timestamp>/, scans, forms, disables the robot) and this script
writes <out>/robot_run.json: {"mode", "output_dir", "captures": {"<pass>": "<folder>"}, "passes_done", "ok"}.
"""
import argparse
import glob
import json
import os
import runpy
import sys
import time

DRIVER = "run_zigzag_oldclamps.py"


def load_toolpath(path):
    doc = json.load(open(path))
    if doc.get("schema_version") != 1:
        sys.exit(f"{path}: schema_version {doc.get('schema_version')} != 1")
    if len(doc["toolpaths"]) != 1:
        sys.exit(f"{path}: {len(doc['toolpaths'])} toolpaths; this driver runs exactly one zigzag per call")
    return doc, doc["toolpaths"][0]


def configure(zf, zp, doc, tp, mode, margin_top, auto_radii):
    """set his module constants from the file and replace his planner by the file's pivots"""
    sheet, rb = doc["sheet"], tp["robot"]
    piv = [(float(x), float(y)) for x, y in tp["pivots_xy_mm"]]
    zf.X_DIM_MM = float(sheet["width_mm"]); zf.Y_DIM_MM = float(sheet["length_mm"])
    zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM = float(sheet["clamp_mm"])
    zf.Y_DIM_EFFECTIVE_MM = zf.Y_DIM_MM - zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM
    zf.MARGIN_SIDE_MM = float(sheet["margin_side_mm"]); zf.MARGIN_BOTTOM_MM = float(sheet["margin_bottom_mm"])
    zf.MARGIN_TOP_MM = float(sheet["margin_top_mm"] if sheet.get("margin_top_mm") is not None else margin_top)
    zf.ZIGZAG_LV_MM = float(rb["zigzag_lv_mm"]); zf.ZIGZAG_ALPHA_DEG = float(rb["zigzag_alpha_deg"]); zf.ZIGZAG_N = int(rb["zigzag_N"])
    zf.ZIGZAG_ROTATION_DEG = float(rb["rotation_deg"]); zf.ZIGZAG_START_MM = float(rb["start_mm"])
    zf.ZIGZAG_WIDTH_MM = round(max(x for x, _ in piv) - min(x for x, _ in piv), 1)
    zf.LEG_REPEAT = 1
    zf.NUM_PASSES = int(tp["passes"]) if mode == "run" else 0

    def plan_zigzag(x_dim, y_dim, zigzag_lv_mm, zigzag_alpha_deg, zigzag_N, rotation_deg, resolution_mm=None, start_mm=None):
        return zp.interpolate_path(list(piv), resolution_mm) if resolution_mm else list(piv)
    zf.plan_zigzag = plan_zigzag
    if auto_radii:
        zf._prompt_pass_radii = lambda pass_num, rx, rz, dir_str="": (rx, rz, False)
    return piv


def safety(zf, piv):
    args = (zf.X_DIM_MM, zf.Y_DIM_MM, zf.MARGIN_SIDE_MM, zf.MARGIN_BOTTOM_MM, zf.MARGIN_TOP_MM, zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM)
    oob = [p for p in piv if not zf.point_in_safe_zone(p, *args)]
    return zf.safe_zone_bounds(*args), oob


def quiet(zf, wf, mode):
    os.startfile = lambda *a, **k: None                                   # his code opens the preview in a viewer
    if wf is not None and hasattr(wf, "plot_last_wheel_force_path"):
        _f = wf.plot_last_wheel_force_path
        wf.plot_last_wheel_force_path = lambda interactive=True: _f(interactive=False)   # the driver's last call blocks otherwise
    if mode == "run":
        _cap = zf._capture_pointcloud_snapshot

        def _capture(subdir_name):
            if subdir_name == "pointcloud_pass0":
                print("[run_toolpath_dobot] pointcloud_pass0 skipped (the loop measured this state already)", flush=True)
                return os.path.join(zf._output_dir or ".", subdir_name)
            return _cap(subdir_name)
        zf._capture_pointcloud_snapshot = _capture
    elif mode == "baseline":
        zf.run_zigzag_forming = lambda: print("[run_toolpath_dobot] baseline only: no forming", flush=True)


def captures_of(run_dir):
    out = {}
    for d in glob.glob(os.path.join(run_dir, "pointcloud_pass*")):
        if glob.glob(os.path.join(d, "device_*_pointcloud_*.ply")):
            out[os.path.basename(d).replace("pointcloud_pass", "")] = os.path.abspath(d)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "baseline", "run"])
    ap.add_argument("toolpath", nargs="?")
    ap.add_argument("--scripts", required=True, help="Danish's Scripts folder (run_zigzag_oldclamps.py, zigzag_forming_oldclamps.py, ...)")
    ap.add_argument("--out", required=True, help="where robot_run.json and the preview go")
    ap.add_argument("--margin-top", type=float, default=12.0, help="his MARGIN_TOP_MM (far-edge strip the wheel cannot reach)")
    ap.add_argument("--auto-radii", action="store_true", help="no Rx/Rz prompt between passes (keep the current radii)")
    a = ap.parse_args()
    if a.mode != "baseline" and not a.toolpath:
        ap.error(f"{a.mode} needs a toolpath.json")
    scripts = os.path.abspath(a.scripts); out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    os.chdir(scripts); sys.path.insert(0, scripts)
    if a.mode == "check":
        import matplotlib; matplotlib.use("Agg")
    import zigzag_planner as zp                      # noqa: E402  his planner
    import zigzag_forming_oldclamps as zf            # noqa: E402  his forming module (imports dobot_sdk; no connection yet)
    try:
        import wheel_force as wf                     # noqa: E402
    except Exception:
        wf = None
    t0 = time.time()
    res = {"mode": a.mode, "toolpath": os.path.abspath(a.toolpath) if a.toolpath else None, "scripts": scripts, "ok": False}

    if a.mode == "baseline":
        doc = tp = None; piv = []
    else:
        doc, tp = load_toolpath(a.toolpath)
        piv = configure(zf, zp, doc, tp, a.mode, a.margin_top, a.auto_radii)
        bounds, oob = safety(zf, piv)
        res["constants"] = {k: getattr(zf, k) for k in ("X_DIM_MM", "Y_DIM_MM", "SHEET_BOTTOM_TO_WHEEL_LINE_MM", "MARGIN_SIDE_MM",
                                                       "MARGIN_BOTTOM_MM", "MARGIN_TOP_MM", "ZIGZAG_LV_MM", "ZIGZAG_ALPHA_DEG",
                                                       "ZIGZAG_N", "ZIGZAG_ROTATION_DEG", "ZIGZAG_START_MM", "ZIGZAG_WIDTH_MM",
                                                       "NUM_PASSES", "LEG_REPEAT", "ARC_RADIUS_X_MM", "ARC_RADIUS_Z_MM")}
        res["safe_zone"] = {"x_min": bounds[0], "x_max": bounds[1], "y_min": bounds[2], "y_max": bounds[3]}
        res["pivots"] = len(piv); res["out_of_zone"] = oob
        print(f"[run_toolpath_dobot] sheet {zf.X_DIM_MM} x {zf.Y_DIM_MM}, wheel line {zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM} mm, "
              f"safe zone X [{bounds[0]:.1f}, {bounds[1]:.1f}] Y [{bounds[2]:.1f}, {bounds[3]:.1f}]; {len(piv)} pivots "
              f"X [{min(x for x, _ in piv):.1f}, {max(x for x, _ in piv):.1f}] Y [{min(y for _, y in piv):.1f}, {max(y for _, y in piv):.1f}]; "
              f"N {zf.ZIGZAG_N}, lv {zf.ZIGZAG_LV_MM}, alpha {zf.ZIGZAG_ALPHA_DEG}, passes {zf.NUM_PASSES}; out of zone: {len(oob)}", flush=True)
        if oob:
            res["error"] = f"{len(oob)} pivot(s) outside his safe zone"
            json.dump(res, open(os.path.join(out, "robot_run.json"), "w"), indent=1)
            sys.exit(f"[run_toolpath_dobot] REFUSED: {res['error']}: {oob[:4]}")
        if a.mode == "check":
            dense = zf.plan_zigzag(0, 0, 0, 0, 0, 0, resolution_mm=zf.ZIGZAG_RESOLUTION_MM, start_mm=0)
            prev = os.path.join(out, "zigzag_preview_dobot.png")
            zf.plot_zigzag(x_dim=zf.X_DIM_MM, y_dim=zf.Y_DIM_MM, waypoints_2d=dense, waypoints_3d=[(x, y, 0.0) for x, y in dense],
                           zigzag_lv_mm=zf.ZIGZAG_LV_MM, zigzag_alpha_deg=zf.ZIGZAG_ALPHA_DEG, zigzag_N=zf.ZIGZAG_N,
                           rotation_deg=zf.ZIGZAG_ROTATION_DEG, arc_radius_mm=None, arc_radius_z_mm=None, pivots_2d=piv[::-1],
                           wheel_z_mm=zf.WHEEL_Z_OFFSET * 500, margin_side=zf.MARGIN_SIDE_MM, margin_bottom=zf.MARGIN_BOTTOM_MM,
                           margin_top=zf.MARGIN_TOP_MM, save_path=prev, sheet_bottom_offset_mm=zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM)
            res["preview"] = prev; res["ok"] = True
            json.dump(res, open(os.path.join(out, "robot_run.json"), "w"), indent=1)
            print(f"[run_toolpath_dobot] check OK; his preview: {prev}", flush=True)
            return 0

    quiet(zf, wf, a.mode)
    print(f"[run_toolpath_dobot] {a.mode}: running {DRIVER} ...", flush=True)
    err = None
    try:
        runpy.run_path(os.path.join(scripts, DRIVER), run_name="__main__")
    except SystemExit as e:
        err = f"driver exited {e.code}" if e.code not in (0, None) else None
    except BaseException as e:                       # noqa: BLE001  his driver already stopped the robot in its own handlers
        err = f"{type(e).__name__}: {e}"
    runs = [d for d in glob.glob(os.path.join(scripts, "outputs", "*")) if os.path.isdir(d) and os.path.getmtime(d) >= t0 - 5]
    run_dir = max(runs, key=os.path.getmtime) if runs else None
    caps = captures_of(run_dir) if run_dir else {}
    want = ["0"] if a.mode == "baseline" else [str(i) for i in range(1, zf.NUM_PASSES + 1)]
    res.update({"output_dir": run_dir, "captures": caps, "passes_done": len([k for k in caps if k != "0"]),
                "ok": err is None and all(k in caps for k in want), "error": err, "secs": round(time.time() - t0)})
    json.dump(res, open(os.path.join(out, "robot_run.json"), "w"), indent=1)
    print(f"[run_toolpath_dobot] done: ok={res['ok']} error={err} run folder {run_dir} captures {sorted(caps)}", flush=True)
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
