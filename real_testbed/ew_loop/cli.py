"""Operator entry points (run from the folder that contains ew_loop/). CLOSED_LOOP.md explains the loop.

  python -m ew_loop.cli loop --sheet 195x190 --clamp 100 --thickness 1.0 --target tgt.npz --out RUN
         (--server URL | --ckpt CKPT | --fixed-plan plan.json)
         --plant robot --robot-code DIR --robot-python PY --feed-dir X,Y,Z [--camera 3] [--yes] [--resume]
      the closed loop on the machine (or --plant sim [--fake-camera] on the training sheet 300x300)

  python -m ew_loop.cli serve --ckpt CKPT [--host 0.0.0.0 --port 8765]
      the planning model as an HTTP service on a GPU machine (CUDA_VISIBLE_DEVICES selects the GPU)

  python -m ew_loop.cli target --from-sim cycles.npz --cycle K --sheet WxL --clamp C --out tgt.npz
      height-field target from a simulator trajectory

  python -m ew_loop.cli reference --capture baseline/ --sheet ... --feed-dir X,Y,Z --out REF
      locate the flat sheet in one camera; writes REF/reference_preview.png to check the setup

  python -m ew_loop.cli measure --capture pointcloud_pass3/ --reference REF [--contact-planner X,Y] --out f
      one capture -> height field (f.npz, f.png)

  python -m ew_loop.cli rigloop --target T.npz --out RUN --plant file (--exchange DIR | --captures DIR) [--ckpt CK | --server URL]
      the automatic loop on the 2026-10-28 rig setup; --plant replay --replay-run DIR rehearses it on a recording
  python -m ew_loop.cli serve-fields --ckpt CK --port 8765     model service for rigloop --server

  python -m ew_loop.cli plan ... --out DIR       one re-plan only (model -> toolpath.json + preview)
  python -m ew_loop.cli check toolpath.json      re-check a toolpath.json against its sheet and draw it
"""
import argparse
import json
import os
import sys

import numpy as np

from . import frames, spec, predict as P
from .frames import Sheet, HeightField


def _sheet_from_args(a):
    return Sheet.parse(a.sheet, clamp_mm=a.clamp, thickness_mm=a.thickness, margin_side_mm=a.margin_side)


def cmd_serve(a):
    P.serve(a.ckpt, host=a.host, port=a.port, schema=a.schema)


def cmd_target(a):
    sheet = _sheet_from_args(a)
    xyz = np.load(a.from_sim)["xyz"]          # (T+1, n_vertices, 3) metres, simulator mesh per cycle
    k = a.cycle if a.cycle >= 0 else xyz.shape[0] + a.cycle
    Pm = xyz[k] * 1e3
    fld = frames.field_from_points(Pm[:, :2], Pm[:, 2], sheet, step_mm=a.step,
                                   meta={"source": "sim", "file": os.path.abspath(a.from_sim), "cycle": int(k)})
    fld = frames.mask_clamp_band(fld, sheet)
    fld.save(a.out)
    lo, hi = fld.zrange()
    print(f"target: cycle {k} of {a.from_sim} -> {a.out}  z in [{lo:.2f}, {hi:.2f}] mm, "
          f"{int(fld.valid.sum())}/{fld.z_mm.size} valid cells")


def cmd_plan(a):
    sheet = _sheet_from_args(a)
    tgt = HeightField.load(a.target)
    cur = HeightField.load(a.cur) if a.cur else frames.flat_field(sheet, step_mm=a.step)
    history = json.load(open(a.history)) if a.history else []
    if a.server:
        planner = P.HttpPlanner(a.server)
        print("server:", planner.health())
    else:
        planner = P.LocalPlanner(a.ckpt, schema=a.schema)
        print(f"model {a.ckpt} loaded in {planner.load_secs:.0f}s")
    os.makedirs(a.out, exist_ok=True)
    res = P.predict(cur, tgt, history, planner, a.out, schema=a.schema)
    print(f"model answered in {res['secs']}s{' (sampled retry)' if res['sampled'] else ''}: {res['raw'][:300]}")
    problems = []
    toolpaths = []
    if res["was"] is None:
        problems.append((-1, "no parseable plan"))
    else:
        block = P.plan_take_cycles(res["was"], a.n) if a.n > 0 else res["was"]
        toolpaths, problems = spec.plan_to_toolpaths(block, sheet, allow_rot90=a.allow_rot90)
    doc = spec.toolpath_doc(sheet, toolpaths, note=f"first {a.n} cycle(s) of the model plan; source {a.out}")
    spec.write_toolpath_json(os.path.join(a.out, "toolpath.json"), doc)
    spec.preview_png(doc, os.path.join(a.out, "preview.png"))
    with open(os.path.join(a.out, "problems.txt"), "w") as f:
        for i, msg in problems:
            f.write(f"entry {i}: {msg}\n")
    for tp in toolpaths:
        o = tp["ours"]
        print(f"  {tp['id']}: {tp['passes']} pass(es)  lv {o['lv_mm']:.1f} alpha {o['alpha_deg']:.2f} N {o['N']} "
              f"rot {o['rot']}  WA centre ({o['cx']:.1f},{o['cy']:.1f}) size {o['wx']:.0f}x{o['wy']:.0f}")
    for i, msg in problems:
        print(f"  entry {i} NOT executable: {msg}")
    print(f"wrote {a.out}/toolpath.json ({len(toolpaths)} toolpath(s)), preview.png, problems.txt")
    return 0 if toolpaths else 2


def cmd_check(a):
    doc = spec.read_toolpath_json(a.toolpath)
    sheet = Sheet.from_dict(doc["sheet"])
    ok = True
    for tp in doc["toolpaths"]:
        bad = spec.check_pivots_planner(np.array(tp["pivots_xy_mm"], dtype=float), sheet)
        print(f"  {tp['id']}: {len(tp['pivots_xy_mm'])} pivots, {tp['passes']} pass(es): "
              f"{'OK' if not bad else f'{len(bad)} pivot(s) OUTSIDE the safe zone'}")
        ok &= not bad
    if a.preview:
        spec.preview_png(doc, a.preview); print("preview ->", a.preview)
    return 0 if ok else 2


def _vec(s):
    return np.array([float(v) for v in s.split(",")], float)


def _camera(a):
    cam = a.camera if a.camera == "merged" else int(a.camera)
    if cam == "merged" and not a.calib:
        raise SystemExit("--camera merged needs --calib (rig calibration folder)")
    return cam


def cmd_reference(a):
    from . import scan_to_field as S
    sheet = _sheet_from_args(a)
    cam = _camera(a)
    ids = None
    if cam == "merged":
        P, ids = S.merge_capture(a.capture, a.calib, with_ids=True)
    else:
        P = S.load_capture(a.capture, cam)
    ref = S.make_reference(P, sheet, _vec(a.feed_dir), cam, not a.up_away, step=a.step,
                           window=None if cam == "merged" else S.DEPTH_WINDOW_MM)
    if ids is not None:
        ref.outline["camera_agreement"] = S.camera_agreement(P, ids, ref)
        if not ref.outline["camera_agreement"]["ok"]:
            print(f"WARNING: the cameras disagree on the flat sheet by {ref.outline['camera_agreement']['spread_mm']} mm "
                  f"(limit {S.CAMERA_AGREEMENT_MM}); recalibrate the rig or use one camera (--camera 3)")
    ref.save(a.out)
    S.reference_preview(P, ref, os.path.join(a.out, "reference_preview.png"))
    frames.field_png(ref.baseline, os.path.join(a.out, "baseline_field.png"))
    print(json.dumps(ref.outline, indent=1))
    print(f"reference -> {a.out} (check reference_preview.png: +y must point from the gripped edge to the free edge)")


def cmd_measure(a):
    from . import scan_to_field as S
    ref = S.Reference.load(a.reference)
    contact = ref.sheet.from_planner(_vec(a.contact_planner)) if a.contact_planner else None
    fld = S.scan_to_field(S.load_capture(a.capture, ref.camera, a.calib), ref, contact_xy=contact, step=a.step)
    fld.save(a.out + ".npz"); frames.field_png(fld, a.out + ".png")
    print(json.dumps({k: v for k, v in fld.meta.items() if k != "sheet"}, indent=1))
    print(f"field -> {a.out}.npz / .png, z in {fld.zrange()} mm")


def cmd_loop(a):
    from . import loop as L, scan_to_field as S
    sheet = _sheet_from_args(a)
    if a.fixed_plan:
        planner = P.FixedPlanner(a.fixed_plan)
    elif a.server:
        planner = P.HttpPlanner(a.server); print("server:", planner.health())
    else:
        planner = P.LocalPlanner(a.ckpt, schema=a.schema)
    feed = _vec(a.feed_dir) if a.feed_dir else None
    if a.plant == "sim":
        rig = None
        if a.fake_camera:
            from .synthetic import FakeRig
            rig = FakeRig(sheet)
            feed = rig.feed_dir_cam(error_deg=20.0)
        plant = L.SimPlant(os.path.join(a.out, "sim"), sheet, rig=rig, cores=(a.core0, a.cores), step=a.step)
    else:
        plant = L.RobotPlant(sheet, a.robot_code, a.robot_python, camera=_camera(a), calib_dir=a.calib, yes=a.yes, script=a.robot_script)
    st = L.run_loop(a.out, sheet, a.target, planner, plant, n=a.n, max_passes=a.max_passes, schema=a.schema,
                    feed_dir_cam=feed, camera=_camera(a) if a.plant == "robot" else 3, up_toward_camera=not a.up_away, allow_rot90=a.allow_rot90,
                    confirm=not a.yes, step=a.step, robust=not a.nonrobust, resume=a.resume)
    return 0 if st.reason and st.reason.startswith(("converged",)) else 3


def cmd_rigloop(a):
    """the automatic loop on the rig setup of 2026-10-28 (231 x 174 x 0.6 sheet, scan-pose captures,
    new-sheet model): measurement = measure.RigMeasurer, model profile = model_newsheet.NewsheetModel."""
    from . import loop as L, measure as ME, model_newsheet as MN
    profile = MN.NewsheetModel(pairs=a.pairs or MN.DEFAULT_PAIRS, data=a.data or MN.DEFAULT_DATA)
    sheet = Sheet(width_mm=231.0, length_mm=174.0, thickness_mm=a.thickness, clamp_mm=a.clamp, margin_side_mm=a.margin_side,
                  margin_bottom_mm=0.0, margin_top_mm=a.margin_top, pad_offsets_from_left_mm=(6.5, 224.5), pad_diameter_mm=13.0)
    script = None
    if a.script:
        script = json.load(open(a.script))
        planner = None; print(f"scripted blocks: {[(s['wa'], s['passes']) for s in script]}")
    elif a.fixed_plan:
        planner = P.FixedFieldsPlanner(a.fixed_plan, profile)
    elif a.server:
        planner = P.HttpFieldsPlanner(a.server); print("server:", planner.health())
    else:
        planner = P.ModelPlanner(a.ckpt or MN.DEFAULT_CKPT, profile); print(f"model loaded in {planner.load_secs:.0f}s")
    if a.plant == "replay":
        plant = L.ReplayPlant(a.replay_run)
    elif a.plant == "dobot":
        plant = L.DobotPlant(a.robot_scripts, python=a.robot_python, margin_top=a.margin_top, auto_radii=a.auto_radii,
                             show_preview=a.show_preview, sheet=sheet)
    else:
        plant = L.FilePlant(a.exchange, a.captures)
    measurer = ME.RigMeasurer(calib_dir=a.calib)
    st = L.run_loop(a.out, sheet, a.target, planner, plant, n=a.n, max_passes=a.max_passes, allow_rot90=False,
                    confirm=not a.yes, robust=True, resume=a.resume, measurer=measurer, profile=profile,
                    script=script, allow_no_sheet=a.allow_no_sheet)
    return 0 if st.reason and st.reason.startswith(("converged", "script_done")) else 3


def cmd_serve_fields(a):
    from . import model_newsheet as MN
    profile = MN.NewsheetModel(pairs=a.pairs or MN.DEFAULT_PAIRS, data=a.data or MN.DEFAULT_DATA)
    P.serve_fields(a.ckpt or MN.DEFAULT_CKPT, profile, host=a.host, port=a.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ew_loop", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_sheet(p):
        p.add_argument("--sheet", default="195x190", help="width x length mm (width = across the feed)")
        p.add_argument("--clamp", type=float, default=100.0,
                       help="gripped edge -> wheel contact line at the robot start pose, mm (robot "
                            "SHEET_BOTTOM_TO_WHEEL_LINE_MM; not the gripper size; confirm on the machine)")
        p.add_argument("--thickness", type=float, default=1.0)
        p.add_argument("--margin-side", type=float, default=12.0,
                       help="corner points stay this far from the side and free edges (robot MARGIN_SIDE_MM)")
        p.add_argument("--step", type=float, default=2.0, help="height-field grid step, mm")

    s = sub.add_parser("serve"); s.add_argument("--ckpt", required=True); s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8765); s.add_argument("--schema", default=P.DEFAULT_SCHEMA); s.set_defaults(f=cmd_serve)

    t = sub.add_parser("target"); add_sheet(t); t.add_argument("--from-sim", required=True)
    t.add_argument("--cycle", type=int, default=-1); t.add_argument("--out", required=True); t.set_defaults(f=cmd_target)

    p = sub.add_parser("plan"); add_sheet(p); p.add_argument("--target", required=True); p.add_argument("--cur")
    p.add_argument("--history"); p.add_argument("--ckpt"); p.add_argument("--server"); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=3, help="cycles to execute before the next scan (0 = whole plan)")
    p.add_argument("--schema", default=P.DEFAULT_SCHEMA); p.add_argument("--allow-rot90", action="store_true"); p.set_defaults(f=cmd_plan)

    c = sub.add_parser("check"); c.add_argument("toolpath"); c.add_argument("--preview"); c.set_defaults(f=cmd_check)

    r = sub.add_parser("reference"); add_sheet(r); r.add_argument("--capture", required=True, help="flat-sheet capture folder")
    r.add_argument("--feed-dir", required=True, help="rough gripped->free edge direction in the camera frame, 'x,y,z'")
    r.add_argument("--camera", default="merged", help="'merged' (needs --calib) or a rig camera number 1-3")
    r.add_argument("--calib", help="rig calibration folder (robot repo code/mvs_recorder/'calibration results')")
    r.add_argument("--up-away", action="store_true",
                   help="the sheet's +z (the model's up) faces AWAY from the camera")
    r.add_argument("--out", required=True); r.set_defaults(f=cmd_reference)

    m = sub.add_parser("measure"); m.add_argument("--capture", required=True); m.add_argument("--reference", required=True)
    m.add_argument("--calib", help="rig calibration folder, needed when the reference uses merged cameras")
    m.add_argument("--contact-planner", help="planner point 'X,Y' under the wheel at capture time (omit: outline fit)")
    m.add_argument("--step", type=float, default=2.0); m.add_argument("--out", required=True); m.set_defaults(f=cmd_measure)

    lp = sub.add_parser("loop"); add_sheet(lp); lp.add_argument("--target", required=True); lp.add_argument("--out", required=True)
    lp.add_argument("--server"); lp.add_argument("--ckpt"); lp.add_argument("--fixed-plan", help="plan JSON instead of a model")
    lp.add_argument("--schema", default=P.DEFAULT_SCHEMA)
    lp.add_argument("--plant", choices=["robot", "sim"], default="robot")
    lp.add_argument("--robot-code", help="robot code folder (contains run_toolpath_sequence.py)")
    lp.add_argument("--robot-python", default=sys.executable, help="python of the robot environment")
    lp.add_argument("--robot-script", default="run_toolpath_sequence.py")
    lp.add_argument("--camera", default="merged", help="'merged' (needs --calib) or a rig camera number 1-3")
    lp.add_argument("--calib", help="rig calibration folder (robot repo code/mvs_recorder/'calibration results')")
    lp.add_argument("--feed-dir"); lp.add_argument("--up-away", action="store_true")
    lp.add_argument("--fake-camera", action="store_true", help="(sim) render every pass through the fake camera rig")
    lp.add_argument("--core0", type=int, default=0); lp.add_argument("--cores", type=int, default=8)
    lp.add_argument("--n", type=int, default=3, help="most passes per block before re-planning")
    lp.add_argument("--max-passes", type=int, default=30); lp.add_argument("--allow-rot90", action="store_true")
    lp.add_argument("--nonrobust", action="store_true", help="max-min span instead of percentiles (simulator)")
    lp.add_argument("--yes", action="store_true", help="no operator confirmation"); lp.add_argument("--resume", action="store_true")
    lp.set_defaults(f=cmd_loop)

    rl = sub.add_parser("rigloop", help="the automatic loop on the 2026-10-28 rig setup (new-sheet model, scan-pose captures)")
    rl.add_argument("--target", required=True, help="HeightField .npz on the 231x174 grid (figures/rig/targets/)")
    rl.add_argument("--out", required=True)
    rl.add_argument("--ckpt", help="new-sheet checkpoint (default: free_aug_e8/checkpoint-1832)"); rl.add_argument("--server", help="URL of `serve-fields`")
    rl.add_argument("--fixed-plan", help="plan JSON instead of a model")
    rl.add_argument("--script", help="JSON list of {wa, passes}: run these blocks as listed, no planner, no controller decisions (dry runs)")
    rl.add_argument("--allow-no-sheet", action="store_true", help="continue when the flat scan holds no sheet (dry runs without a sheet)")
    rl.add_argument("--plant", choices=["file", "replay", "dobot"], default="file")
    rl.add_argument("--robot-scripts", help="(dobot) Danish's Scripts folder (English-Wheel---DoBot-Files/Scripts)")
    rl.add_argument("--robot-python", help="(dobot) python for his driver (default: this one)")
    rl.add_argument("--auto-radii", action="store_true", help="(dobot) skip his Rx/Rz prompt between passes")
    rl.add_argument("--show-preview", action="store_true", help="(dobot) let his driver open its zigzag preview in the image viewer (desktop)")
    rl.add_argument("--margin-top", type=float, default=12.0, help="robot MARGIN_TOP_MM (far-edge strip the wheel cannot reach)")
    rl.add_argument("--exchange", help="(file) folder shared with the robot side: round_NN/toolpath.json out, captures/pointcloud_pass<k> in")
    rl.add_argument("--captures", help="(file) folder where the capture folders pointcloud_pass<k> appear, e.g. the robot software's "
                                       "run folder when the loop runs on the robot PC (default <exchange>/captures)")
    rl.add_argument("--replay-run", help="(replay) recorded run folder with pointcloud_pass0..k")
    rl.add_argument("--calib", help="rig calibration folder (default: the 09-28 deliverable)")
    rl.add_argument("--clamp", type=float, default=12.0, help="robot SHEET_BOTTOM_TO_WHEEL_LINE_MM for this sheet (assumed 12; confirm)")
    rl.add_argument("--thickness", type=float, default=1.0, help="sheet thickness in mm, for the records (the real sheet: 1.0, user 2026-10-29; "
                                                                 "the model's prompt keeps the text it was trained with)")
    rl.add_argument("--margin-side", type=float, default=43.4, help="robot MARGIN_SIDE_MM (43.4 = the model's reachable zone, x within +-72.1)")
    rl.add_argument("--pairs"); rl.add_argument("--data")
    rl.add_argument("--n", type=int, default=3, help="most passes per block before re-planning")
    rl.add_argument("--max-passes", type=int, default=12); rl.add_argument("--yes", action="store_true"); rl.add_argument("--resume", action="store_true")
    rl.set_defaults(f=cmd_rigloop)

    sf = sub.add_parser("serve-fields", help="model service for rigloop --server: takes height fields, returns the plan")
    sf.add_argument("--ckpt"); sf.add_argument("--host", default="0.0.0.0"); sf.add_argument("--port", type=int, default=8765)
    sf.add_argument("--pairs"); sf.add_argument("--data"); sf.set_defaults(f=cmd_serve_fields)

    a = ap.parse_args(argv)
    if a.cmd == "plan" and not (a.ckpt or a.server):
        ap.error("plan needs --ckpt or --server")
    if a.cmd == "loop" and not (a.ckpt or a.server or a.fixed_plan):
        ap.error("loop needs --ckpt, --server or --fixed-plan")
    if a.cmd == "loop" and a.plant == "robot" and not (a.robot_code and a.feed_dir):
        ap.error("--plant robot needs --robot-code and --feed-dir")
    if a.cmd == "rigloop":
        if a.plant == "file" and not (a.exchange or a.captures):
            ap.error("--plant file needs --exchange DIR and/or --captures DIR")
        if a.plant == "replay" and not a.replay_run:
            ap.error("--plant replay needs --replay-run DIR")
        if a.plant == "dobot" and not a.robot_scripts:
            ap.error("--plant dobot needs --robot-scripts DIR")
    return a.f(a) or 0


if __name__ == "__main__":
    sys.exit(main())
