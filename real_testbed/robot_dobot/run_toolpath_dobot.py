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
  the wheel rule (user, 2026-10-29), decided per TRANSITION between two consecutive passes:
                             - the next pass is the SAME zigzag (reversed) and starts where the last ended: the wheel stays
                               loaded, no unload / reposition / reload, the next pass starts from the end point;
                             - a DIFFERENT zigzag next, or ends that do not coincide: unload -> reposition -> reload (his flow);
                             - going to the scan pose: always unload. One scan per block, after its last pass.
                             The loop sends one zigzag per block (the plan's first entry x the controller's passes), so inside
                             a block every transition is the first case (checked on his own motion plans before any motion);
                             between blocks there is always the scan.
  zf.run_zigzag_forming   -> no-op in `baseline` (his run_step_7_5 then only scans and parks at pos_9)
  zf._prompt_pass_radii   -> kept (the operator enters Rx/Rz between passes as today) unless --auto-radii
  os.startfile            -> no image viewer unless --show-preview; wheel_force.plot_last_wheel_force_path -> non-blocking
  run_output.write_run_info -> his run_info.txt gains the sheet geometry (X_DIM across the feed, Y_DIM along it,
                             thickness, wheel-line distance, margins), the toolpath file, the zigzag centre and the pivots
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


def configure(zf, zp, doc, tp, mode, margin_top, auto_radii, start_direction="l2r"):
    """set his module constants from the file and replace his planner by the file's pivots. His driver runs
    pass 1 over the REVERSED pivot list ('left -> right'), pass 2 over the list as given, ...; for a block that
    must start right -> left the list is handed over reversed."""
    sheet, rb = doc["sheet"], tp["robot"]
    piv = [(float(x), float(y)) for x, y in tp["pivots_xy_mm"]]
    if start_direction == "r2l":
        piv = piv[::-1]
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


def run_info_extra(sheet, tp=None, toolpath=None):
    """the lines added to his run_info.txt (sheet block of toolpath.json, or the loop's sheet for the baseline)"""
    e = {"Sheet X_DIM (mm, across the feed)": sheet["width_mm"], "Sheet Y_DIM (mm, along the feed)": sheet["length_mm"],
         "Sheet Thickness (mm)": sheet["thickness_mm"], "Sheet Bottom To Wheel Line (mm)": sheet["clamp_mm"],
         "Safe-zone Margins side / bottom / top (mm)": f"{sheet['margin_side_mm']} / {sheet['margin_bottom_mm']} / {sheet.get('margin_top_mm')}"}
    if tp is not None:
        rb = tp["robot"]
        e.update({"Toolpath": toolpath, "Toolpath ID": tp["id"], "Zigzag Centre X (mm, planner frame)": rb.get("center_x_mm"),
                  "Zigzag Start Y (mm)": rb.get("start_mm"), "Pivots (planner frame, mm)": json.dumps(tp["pivots_xy_mm"])})
    return e


def speed_info(zf):
    keys = ("ACCELERATION", "VELOCITY", "LEG_V_MULTIPLIER", "LEG_A_MULTIPLIER", "LEG_V", "LEG_A", "PIVOT_V", "PIVOT_A")
    return {"Speeds (% of CR5A max)": ", ".join(f"{k} {getattr(zf, k)}" for k in keys if hasattr(zf, k))}


def patch_run_info(extra):
    import run_output as ro
    _w = ro.write_run_info

    def write_run_info(output_dir, **params):
        return _w(output_dir, **params, **extra)
    ro.write_run_info = write_run_info


def _pose_gap(zf, a, b):
    """(position mm, rotation deg) between two of his poses [x, y, z (m), rotation vector]"""
    import math
    dp = 1000.0 * math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))
    Ra, Rb = zf._rotation_matrix(a[3:6]), zf._rotation_matrix(b[3:6])
    tr = sum(Ra[k][i] * Rb[k][i] for i in range(3) for k in range(3))          # trace(Ra^T Rb)
    return dp, math.degrees(math.acos(max(-1.0, min(1.0, (tr - 1.0) / 2.0))))


def turnaround_coincides(zf, piv, tol_mm=0.5, tol_deg=0.5):
    """does the last pose of a pass end where the next (reversed) pass starts? his own _build_motion_plan on both
    directions, no robot needed"""
    a, _, _ = zf._build_motion_plan(list(piv)[::-1])          # pass 1 as his driver runs it (pivots reversed = his 'l2r')
    b, _, _ = zf._build_motion_plan(list(piv))                # pass 2 (his 'r2l')
    end_a, start_b = a[-1][2][-1], b[0][2][0]
    d1 = _pose_gap(zf, end_a, start_b)
    end_b, start_a = b[-1][2][-1], a[0][2][0]
    d2 = _pose_gap(zf, end_b, start_a)
    ok = max(d1[0], d2[0]) <= tol_mm and max(d1[1], d2[1]) <= tol_deg
    return ok, {"end1_to_start2_mm_deg": [round(d1[0], 3), round(d1[1], 3)], "end2_to_start1_mm_deg": [round(d2[0], 3), round(d2[1], 3)]}


def continuous_block(zf, n_passes, coincide, log=print):
    """wrap his pass loop so that a block of n_passes is: load once, the passes back to back (or with his normal
    unload/reposition/reload where the ends do not coincide), unload, scan once. Returns the bookkeeping dict."""
    st = {"done": 0, "between": False, "expect_start": False, "last_pose": None, "skipped": [], "n": n_passes, "coincide": coincide}
    pc_pose = list(zf.pointcloud_position)
    o_exec, o_unload, o_movel, o_arc, o_seq = zf._execute_pass, zf._unload_wheel_by_recorded_pulses, zf._movel_single, zf._arc_single, zf._send_sequence
    o_cap, o_reload, o_prompt = zf._capture_pointcloud_snapshot, zf._reload_wheel, zf._prompt_pass_radii

    def intermediate():
        return st["done"] < n_passes

    def _execute_pass(*a, **k):
        st["between"] = False
        r = o_exec(*a, **k)
        st["done"] += 1
        if intermediate():
            st["between"] = True
        return r

    def _unload(pulse_count):
        if st["between"] and coincide:
            st["skipped"].append(f"unload after pass {st['done']}"); log(f"[continuous] pass {st['done']} ends where pass {st['done'] + 1} starts: wheel stays loaded")
            return None
        return o_unload(pulse_count)

    def _movel(pose, *a, **k):
        if st["between"] and list(pose) == pc_pose:
            st["skipped"].append(f"scan-pose move after pass {st['done']}"); log(f"[continuous] no scan between passes {st['done']} and {st['done'] + 1} (one scan per block)")
            return None
        if st["between"] and st["expect_start"]:
            st["expect_start"] = False
            if coincide:
                gap = _pose_gap(zf, st["last_pose"], pose) if st["last_pose"] is not None else (float("inf"), float("inf"))
                if gap[0] > 1.0 or gap[1] > 1.0:
                    raise RuntimeError(f"continuous pass: next start is {gap[0]:.2f} mm / {gap[1]:.2f} deg from the last pose (expected 0)")
                st["skipped"].append(f"reposition after pass {st['done']} ({gap[0]:.3f} mm, {gap[1]:.3f} deg)")
                log(f"[continuous] next pass starts at the end point ({gap[0]:.3f} mm, {gap[1]:.3f} deg): no reposition")
                return None
        st["last_pose"] = list(pose)
        return o_movel(pose, *a, **k)

    def _arc(pose_mid, pose_end, *a, **k):
        st["last_pose"] = list(pose_end)
        return o_arc(pose_mid, pose_end, *a, **k)

    def _seq(poses, *a, **k):
        st["last_pose"] = list(poses[-1])
        return o_seq(poses, *a, **k)

    def _capture(subdir_name):
        if st["between"]:
            st["skipped"].append(f"scan after pass {st['done']}")
            return None
        return o_cap(subdir_name)

    def _prompt(pass_num, rx, rz, dir_str=""):
        st["expect_start"] = True
        return o_prompt(pass_num, rx, rz, dir_str)

    def _reload():
        if st["between"] and coincide:
            st["skipped"].append(f"reload before pass {st['done'] + 1}")
            return None
        return o_reload()

    zf._execute_pass, zf._unload_wheel_by_recorded_pulses, zf._movel_single = _execute_pass, _unload, _movel
    zf._arc_single, zf._send_sequence, zf._capture_pointcloud_snapshot = _arc, _seq, _capture
    zf._prompt_pass_radii, zf._reload_wheel = _prompt, _reload
    return st


def safety(zf, piv):
    args = (zf.X_DIM_MM, zf.Y_DIM_MM, zf.MARGIN_SIDE_MM, zf.MARGIN_BOTTOM_MM, zf.MARGIN_TOP_MM, zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM)
    oob = [p for p in piv if not zf.point_in_safe_zone(p, *args)]
    return zf.safe_zone_bounds(*args), oob


def quiet(zf, wf, mode, show_preview=False):
    if not show_preview:
        os.startfile = lambda *a, **k: None                               # his code opens the preview in a viewer
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
    # his driver prints non-ASCII (theta, arrows); a console takes it, a pipe (ssh, the loop's subprocess) defaults to
    # cp1252 on Windows and raises UnicodeEncodeError mid-pass (2026-10-29 dry run: his handler unloaded and stopped)
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "baseline", "run"])
    ap.add_argument("toolpath", nargs="?")
    ap.add_argument("--scripts", required=True, help="Danish's Scripts folder (run_zigzag_oldclamps.py, zigzag_forming_oldclamps.py, ...)")
    ap.add_argument("--out", required=True, help="where robot_run.json and the preview go")
    ap.add_argument("--margin-top", type=float, default=12.0, help="his MARGIN_TOP_MM (far-edge strip the wheel cannot reach)")
    ap.add_argument("--auto-radii", action="store_true", help="no Rx/Rz prompt between passes (keep the current radii)")
    ap.add_argument("--show-preview", action="store_true", help="let his driver open its zigzag preview in the image viewer (desktop session)")
    ap.add_argument("--sheet-json", help="(baseline) the loop's sheet (frames.Sheet.to_dict) for run_info.txt")
    ap.add_argument("--start-direction", choices=["l2r", "r2l"], default="l2r",
                    help="direction of the block's first pass (his driver always starts left->right: r2l hands it the pivots reversed)")
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
        zf.NUM_PASSES = 0                                               # his run_info.txt then says 0 passes
        extra = {"Loop Mode": "baseline: flat scan only, no forming"}
        if a.sheet_json:
            sh = json.load(open(a.sheet_json))
            zf.X_DIM_MM, zf.Y_DIM_MM = float(sh["width_mm"]), float(sh["length_mm"])
            zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM = float(sh["clamp_mm"]); zf.Y_DIM_EFFECTIVE_MM = zf.Y_DIM_MM - zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM
            extra.update(run_info_extra(sh))
        patch_run_info({**extra, **speed_info(zf)})
    else:
        doc, tp = load_toolpath(a.toolpath)
        piv = configure(zf, zp, doc, tp, a.mode, a.margin_top, a.auto_radii, a.start_direction)
        dirs = []
        d = a.start_direction
        for _ in range(int(tp["passes"]) if a.mode == "run" else 0):
            dirs.append(d); d = "r2l" if d == "l2r" else "l2r"
        res["pass_directions"] = dirs
        patch_run_info({**speed_info(zf), "Loop Mode": f"run: {tp['passes']} pass(es) of the toolpath below",
                        "Pass Directions": ", ".join(f"{i + 1}: {'left -> right' if x == 'l2r' else 'right -> left'}" for i, x in enumerate(dirs)) or "none",
                        **run_info_extra(doc["sheet"], tp, os.path.abspath(a.toolpath))})
        if a.start_direction == "r2l":
            print("[run_toolpath_dobot] NOTE: this block starts RIGHT -> LEFT (pivots handed over reversed); his driver's "
                  "'left -> right' / 'right -> left' labels are therefore swapped for this run", flush=True)
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

    quiet(zf, wf, a.mode, a.show_preview)
    block = None
    if a.mode == "run":
        if not a.auto_radii and zf.NUM_PASSES > 1:
            print("[run_toolpath_dobot] Rx/Rz prompt kept: the passes of this block are NOT run back to back "
                  "(changed radii change the plan); still one scan per block", flush=True)
            coincide, gaps = False, {}
        else:
            coincide, gaps = turnaround_coincides(zf, piv)
        block = continuous_block(zf, zf.NUM_PASSES, coincide)
        res["continuous"] = {"coincide": coincide, **gaps}
        print(f"[run_toolpath_dobot] block: {zf.NUM_PASSES} pass(es), "
              + ("back to back, wheel loaded throughout" if coincide else "his unload/reposition/reload between passes")
              + f", ONE scan after the last pass; turnaround check {gaps}", flush=True)
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
    want = ["0"] if a.mode == "baseline" else [str(zf.NUM_PASSES)]
    missing = [k for k in want if k not in caps]
    if err is None and missing:                      # his driver catches its own errors and prints them: report the symptom
        err = f"driver finished without capture(s) pointcloud_pass{','.join(missing)} (see its output above; robot alarm / control mode / camera)"
    done = block["done"] if block is not None else 0
    res.update({"output_dir": run_dir, "captures": caps, "passes_done": done, "skipped_between_passes": block["skipped"] if block else [],
                "ok": not missing and err is None, "error": err, "secs": round(time.time() - t0)})
    json.dump(res, open(os.path.join(out, "robot_run.json"), "w"), indent=1)
    print(f"[run_toolpath_dobot] done: ok={res['ok']} error={err} run folder {run_dir} captures {sorted(caps)}", flush=True)
    return 0 if res["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
