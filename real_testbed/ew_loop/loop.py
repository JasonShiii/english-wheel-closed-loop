"""The closed loop: measure -> plan -> choose passes -> convert -> execute -> measure ...

Interchangeable plants execute a toolpath.json and return captures:
  RobotPlant  (first setup) runs robot_pr/run_toolpath_sequence.py in the robot's Python
              environment and reads the point clouds it saved after every pass
  SimPlant    runs the C++ simulator with the physics the model was trained on (training
              sheet only: 300 x 300 x 0.5 mm); optionally renders every simulated pass through
              the fake camera rig, so the measurement code runs exactly as on the machine
  FilePlant   (rig, 2026-10-28) the robot side runs the block and captures ONCE at the scan pose;
              the loop waits for the capture folder (MPC-like cadence: one measurement per block)
  ReplayPlant (rehearsal) returns the recorded capture of a run folder after the block's passes
  DobotPlant  (rig, 2026-10-28) Danish's Dobot driver through robot_dobot/run_toolpath_dobot.py: the whole
              block without an operator copying anything; his driver scans after every pass

Every round is written to OUT/round_NN/ (model inputs and answer, toolpath.json and preview,
the measured field after every pass); OUT/state.json holds the bookkeeping, so a run can be
resumed after an interruption.
"""
import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import frames, predict as P, scan_to_field as S, spec, state as ST
from .frames import Sheet, HeightField


@dataclass
class Capture:
    toolpath_id: Optional[str]
    pass_index: int                       # 0 = the flat reference capture
    contact_xy: np.ndarray                # sheet point under the wheel at capture time
    points: Optional[np.ndarray] = None   # camera points (robot or fake rig)
    field: Optional[HeightField] = None   # height field directly (simulator without fake rig)
    camera_ids: Optional[np.ndarray] = None   # which rig camera each point came from (merged captures)
    pass_dir: Optional[str] = None        # capture folder (rig: measured by measure.RigMeasurer at the scan pose)


def capture_contact(tp, pass_index, sheet):
    """Sheet point under the wheel when the capture after pass `pass_index` is taken.

    The robot (zigzag_forming.run_zigzag_forming, oct_2_demo) runs the passes of one toolpath
    back and forth without lifting the wheel and captures right where each pass ends: odd
    passes end at the last pivot, even passes at the first. Checked against the robot's own
    kinematics in robot_pr/dry_run_toolpath_sequence.py."""
    piv = tp["pivots_xy_mm"]
    return sheet.from_planner(np.array(piv[-1] if pass_index % 2 == 1 else piv[0], float))


# ---------------------------------------------------------------------------------------
# plants
# ---------------------------------------------------------------------------------------
class SimPlant:
    def __init__(self, workdir, sheet, rig=None, cores=(0, 8), step=2.0):
        if (sheet.width_mm, sheet.length_mm) != (300.0, 300.0):
            raise ValueError("SimPlant runs the training physics: sheet must be 300 x 300 mm")
        from . import add_repo_paths
        add_repo_paths()
        from mpc_rollout import sim_block                       # experiments/mpc_end2end
        self.sim_block, self.workdir, self.sheet, self.rig = sim_block, os.path.abspath(workdir), sheet, rig
        self.cores, self.step, self.state_in, self.k = cores, step, None, 0
        os.makedirs(self.workdir, exist_ok=True)

    def _capture(self, tp_id, k, contact, xyz_m):
        if self.rig is None:
            Pm = xyz_m * 1e3
            return Capture(tp_id, k, contact, field=frames.field_from_points(Pm[:, :2], Pm[:, 2], self.sheet, self.step))
        from scipy.interpolate import LinearNDInterpolator
        Pm = xyz_m * 1e3
        f = LinearNDInterpolator(Pm[:, :2], Pm[:, 2])
        return Capture(tp_id, k, contact, points=self.rig.capture(lambda X, Y: f(X, Y), contact))

    def baseline(self, round_dir):
        n = 51
        g = np.linspace(-0.15, 0.15, n)
        X, Y = np.meshgrid(g, g)
        flat = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1)
        return self._capture(None, 0, np.array([0.0, self.sheet.wheel_line_y]), flat)

    def execute(self, doc, round_dir):
        caps = []
        for tp in doc["toolpaths"]:
            o = tp["ours"]
            wa = dict(cx=o["cx"], cy=o["cy"], wx=o["wx"], wy=o["wy"], rot=o["rot"], N=o["N"], n=tp["passes"])
            d = os.path.join(self.workdir, f"block_{self.k:03d}_{tp['id']}")
            self.k += 1
            _, state_npy, state_out = self.sim_block((tp["id"], [wa], d, self.cores[0], self.cores[1], self.state_in, True))
            if state_npy is None:
                raise RuntimeError(f"simulator failed, see {d}/run.log")
            self.state_in = state_out
            xyz = np.load(os.path.join(d, "cycles.npz"))["xyz"]
            for k in range(1, tp["passes"] + 1):
                caps.append(self._capture(tp["id"], k, capture_contact(tp, k, self.sheet), xyz[k - 1]))
        return caps, len(caps)

    def state(self):
        return {"state_in": self.state_in, "k": self.k}

    def restore(self, d):
        self.state_in, self.k = d.get("state_in"), d.get("k", 0)


class RobotPlant:
    def __init__(self, sheet, robot_code_dir, robot_python, camera="merged", calib_dir=None, yes=False,
                 script="run_toolpath_sequence.py"):
        self.sheet, self.code, self.py, self.camera, self.yes, self.script = sheet, robot_code_dir, robot_python, camera, yes, script
        self.calib_dir = calib_dir or os.path.join(robot_code_dir, "mvs_recorder", "calibration results")

    def _run(self, args):
        cmd = [self.py, self.script] + args
        print("[robot]", " ".join(cmd), flush=True)
        return subprocess.run(cmd, cwd=self.code).returncode

    def baseline(self, round_dir):
        out = os.path.join(round_dir, "robot")
        if self._run(["baseline", "--out", out]) != 0:
            raise RuntimeError("robot baseline capture failed")
        cap = Capture(None, 0, np.array([0.0, self.sheet.wheel_line_y]))
        if self.camera == "merged":
            cap.points, cap.camera_ids = S.merge_capture(os.path.join(out, "baseline"), self.calib_dir, with_ids=True)
        else:
            cap.points = S.load_capture(os.path.join(out, "baseline"), self.camera)
        return cap

    def execute(self, doc, round_dir):
        out = os.path.join(round_dir, "robot")
        rc = self._run(["run", os.path.join(round_dir, "toolpath.json"), "--out", out] + (["--yes"] if self.yes else []))
        caps, executed = [], 0
        for i, tp in enumerate(doc["toolpaths"]):
            sub = os.path.join(out, f"tp{i}_{tp['id']}")
            for k in range(1, tp["passes"] + 1):
                pdir = os.path.join(sub, f"pointcloud_pass{k}")
                if not os.path.isdir(pdir):              # created right after the pass finished
                    break
                executed += 1
                try:
                    caps.append(Capture(tp["id"], k, capture_contact(tp, k, self.sheet),
                                        points=S.load_capture(pdir, self.camera, self.calib_dir)))
                except FileNotFoundError:
                    pass
        if rc != 0:
            print(f"[robot] run_toolpath_sequence.py exited with {rc}; {executed} pass(es) found", flush=True)
        return caps, executed

    def state(self):
        return {}

    def restore(self, d):
        pass


class ReplayPlant:
    """offline rehearsal: 'executes' a block by returning the next recorded captures of a run folder
    (pointcloud_pass0 = flat, pointcloud_pass1.. = after each pass). The toolpath is ignored, so the
    shapes do not follow the plan; the loop mechanics, the measurement and the controller do run."""
    def __init__(self, run):
        self.run, self.k = run, 0
        from .measure import pass_dir
        self._pd = pass_dir

    def baseline(self, round_dir):
        return Capture(None, 0, None, pass_dir=self._pd(self.run, 0))

    def execute(self, doc, round_dir):
        """one capture per block: the recorded capture after the block's passes (or the last one recorded);
        executed = the recorded passes consumed"""
        n = sum(int(tp["passes"]) for tp in doc["toolpaths"])
        avail = [j for j in range(self.k + 1, self.k + n + 1) if os.path.isdir(self._pd(self.run, j))]
        if not avail:
            return [], 0
        self.k = avail[-1]
        return [Capture(doc["toolpaths"][0]["id"], self.k, None, pass_dir=self._pd(self.run, self.k))], len(avail)

    def state(self):
        return {"k": self.k}

    def restore(self, d):
        self.k = d.get("k", 0)


class FilePlant:
    """the robot side is a person with the robot's own software: the loop writes toolpath.json (and
    preview.png) into <out>/round_NN/ and, if an exchange folder is given, a copy into <exchange>/round_NN/;
    the operator runs the WHOLE block on the robot, moves the sheet to the scan pose, captures ONCE
    (cadence of 2026-10-28) and presses Enter. Capture folders are read from `captures` (the robot
    software's own run folder when the loop runs on the robot PC) or from <exchange>/captures/ (copied
    by hand: USB, SharePoint, scp), named pointcloud_pass<k> with k counting the captures of the session
    (pointcloud_pass0 = the flat scan). If the operator captured after every pass anyway, every new folder
    is measured for the record and the last one is the block's measurement."""
    def __init__(self, exchange=None, captures=None):
        if not (exchange or captures):
            raise ValueError("FilePlant needs an exchange folder or a captures folder")
        self.exchange = os.path.abspath(exchange) if exchange else None
        self.captures = os.path.abspath(captures) if captures else os.path.join(self.exchange, "captures")
        os.makedirs(self.captures, exist_ok=True)
        self.k = 0
        from .measure import pass_dir
        self._pd = pass_dir

    def _have(self, k):
        d = self._pd(self.captures, k)
        return os.path.isdir(d) and bool(glob.glob(os.path.join(d, "device_*_pointcloud_*.ply")))

    def baseline(self, round_dir):
        while not self._have(0):
            input(f"  put the FLAT scan of the gripped sheet (device_1..3_pointcloud_001.ply) into {self._pd(self.captures, 0)} and press Enter ")
        return Capture(None, 0, None, pass_dir=self._pd(self.captures, 0))

    def execute(self, doc, round_dir):
        n = sum(int(tp["passes"]) for tp in doc["toolpaths"])
        tp_path = os.path.join(round_dir, "toolpath.json")
        if self.exchange:
            import shutil
            xd = os.path.join(self.exchange, os.path.basename(round_dir)); os.makedirs(xd, exist_ok=True)
            for f in ("toolpath.json", "preview.png"):
                if os.path.exists(os.path.join(round_dir, f)):
                    shutil.copy2(os.path.join(round_dir, f), xd)
            tp_path = os.path.join(xd, "toolpath.json")
        print(f"  >>> run {tp_path} on the robot: {n} pass(es), then the scan pose and ONE capture -> "
              f"{self._pd(self.captures, self.k + 1)}", flush=True)
        executed = n
        while True:
            ans = input("  press Enter when the capture folder is there (or type the number of passes actually run, then Enter) ").strip()
            if ans.isdigit():
                executed = int(ans)
            new = []
            while self._have(self.k + len(new) + 1):
                new.append(self.k + len(new) + 1)
            if new:
                break
            print(f"  no new capture folder found (expected {self._pd(self.captures, self.k + 1)} with device_1..3_pointcloud_*.ply)", flush=True)
        if len(new) > 1:
            print(f"  {len(new)} new captures (pointcloud_pass{new[0]}..{new[-1]}): the last one is the block's measurement", flush=True)
        tp = doc["toolpaths"][0]
        caps = [Capture(tp["id"], k, None, pass_dir=self._pd(self.captures, k)) for k in new]
        self.k = new[-1]
        return caps, executed

    def state(self):
        return {"k": self.k}

    def restore(self, d):
        self.k = d.get("k", 0)


class DobotPlant:
    """the Dobot CR5A cell (2026-10-28) through Danish's own driver, run by robot_dobot/run_toolpath_dobot.py
    in this Python with cwd = his Scripts folder. baseline() = scan pose + flat scan (pointcloud_pass0);
    execute() = the block: his driver runs the passes and scans after EVERY pass (his tested flow; the operator
    answers his Rx/Rz prompt between passes unless auto_radii), every capture is returned and the loop takes
    the last one as the block's measurement. Capture index = capture count of the session."""
    ADAPTER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot_dobot", "run_toolpath_dobot.py")

    def __init__(self, scripts, python=None, margin_top=12.0, auto_radii=False):
        self.scripts, self.python, self.margin_top, self.auto_radii = os.path.abspath(scripts), python or sys.executable, margin_top, auto_radii
        self.k = 0
        if not os.path.exists(os.path.join(self.scripts, "run_zigzag_oldclamps.py")):
            raise FileNotFoundError(f"{self.scripts}: run_zigzag_oldclamps.py not found")

    def _run(self, mode, round_dir, toolpath=None):
        out = os.path.join(round_dir, "robot"); os.makedirs(out, exist_ok=True)
        cmd = [self.python, self.ADAPTER, mode] + ([toolpath] if toolpath else []) + ["--scripts", self.scripts, "--out", out,
                                                                                      "--margin-top", str(self.margin_top)] + (["--auto-radii"] if self.auto_radii else [])
        print("  >>> " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
        r = subprocess.run(cmd, cwd=self.scripts)
        p = os.path.join(out, "robot_run.json")
        if not os.path.exists(p):
            raise RuntimeError(f"robot adapter exited {r.returncode} without {p}")
        res = json.load(open(p))
        if not res.get("ok"):
            raise RuntimeError(f"robot {mode} failed: {res.get('error')} (captures {sorted(res.get('captures', {}))}, run folder {res.get('output_dir')})")
        return res

    def baseline(self, round_dir):
        res = self._run("baseline", round_dir)
        return Capture(None, 0, None, pass_dir=res["captures"]["0"])

    def execute(self, doc, round_dir):
        res = self._run("run", round_dir, os.path.join(round_dir, "toolpath.json"))
        tp, caps = doc["toolpaths"][0], []
        for j in sorted(int(k) for k in res["captures"] if int(k) > 0):
            self.k += 1
            caps.append(Capture(tp["id"], self.k, None, pass_dir=res["captures"][str(j)]))
        return caps, int(res.get("passes_done", len(caps)))

    def state(self):
        return {"k": self.k}

    def restore(self, d):
        self.k = d.get("k", 0)


# ---------------------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------------------
def _save_field(fld, path_noext, profile=None):
    fld.save(path_noext + ".npz")
    (profile.render if profile is not None else frames.field_png)(fld, path_noext + ".png")


def run_loop(out, sheet, target_path, planner, plant, n=3, max_passes=30, schema=P.DEFAULT_SCHEMA,
             feed_dir_cam=None, camera=S.DEFAULT_CAMERA, up_toward_camera=True, allow_rot90=False,
             confirm=True, step=2.0, robust=True, resume=False, log=None, measurer=None, profile=None):
    """measurer (measure.RigMeasurer) + profile (model_newsheet.NewsheetModel) = the rig path (2026-10-28):
    captures are folders measured at the scan pose, the planner's plan_fields() takes the fields; the
    fields are cropped and plane-removed by the measurer, so no clamp-band mask is applied here."""
    log = log or (lambda *a: print(*a, flush=True))
    out = os.path.abspath(out)                     # the plants run tools from other folders
    os.makedirs(out, exist_ok=True)
    rig = measurer is not None
    target = HeightField.load(target_path) if rig else frames.mask_clamp_band(HeightField.load(target_path), sheet)
    xg, yg = (measurer.xg, measurer.yg) if rig else frames.grid_for(sheet, step)
    if target.z_mm.shape != (len(yg), len(xg)):
        raise ValueError(f"target grid {target.z_mm.shape} is not the loop's grid {(len(yg), len(xg))}")
    st_path, ref_dir = os.path.join(out, "state.json"), os.path.join(out, "round_00", "reference")
    ref = None
    if resume:
        saved = json.load(open(st_path))
        state = ST.LoopState.from_dict(saved["state"])
        state.stopped, state.reason = False, None
        plant.restore(saved.get("plant", {}))
        cur = HeightField.load(saved["last_field"])
        first_round = saved["round"] + 1
        if os.path.isdir(ref_dir):
            if rig:
                measurer.load(ref_dir)
            else:
                ref = S.Reference.load(ref_dir)
        log(f"resuming after round {saved['round']}: {state.exec_passes} passes done")
    else:
        r0 = os.path.join(out, "round_00")
        os.makedirs(r0, exist_ok=True)
        base = plant.baseline(r0)
        if rig:
            cur = measurer.reference(os.path.dirname(base.pass_dir))
            measurer.save(ref_dir)
            measurer.preview(os.path.join(ref_dir, "reference_preview.png"), cur)
            log(f"reference: {measurer.info}")
            if measurer.info.get("inside_fraction", 0) < 0.5 or cur.meta.get("coverage_observed", 0) < 0.3:
                raise RuntimeError(f"no sheet found in the flat scan {base.pass_dir}: inside_fraction {measurer.info.get('inside_fraction')}, "
                                   f"coverage {cur.meta.get('coverage_observed')} (is the sheet gripped and at the scan pose? see {ref_dir}/reference_preview.png)")
            log(f"flat field: z {cur.zrange()[0]:.2f}..{cur.zrange()[1]:.2f} mm, coverage of the observed region {cur.meta['coverage_observed']}")
        elif base.points is not None:
            if feed_dir_cam is None:
                raise ValueError("feed_dir_cam is needed to locate the sheet in the camera view")
            ref = S.make_reference(base.points, sheet, feed_dir_cam, camera, up_toward_camera, base.contact_xy, step,
                                   window=None if camera == "merged" else S.DEPTH_WINDOW_MM)
            if base.camera_ids is not None:
                ref.outline["camera_agreement"] = S.camera_agreement(base.points, base.camera_ids, ref)
                if not ref.outline["camera_agreement"]["ok"]:
                    raise RuntimeError(f"the cameras disagree on the flat sheet: {ref.outline['camera_agreement']} "
                                       f"(limit {S.CAMERA_AGREEMENT_MM} mm). Recalibrate the rig, or run with --camera 3.")
            ref.save(ref_dir)
            S.reference_preview(base.points, ref, os.path.join(ref_dir, "reference_preview.png"))
            cur = S.scan_to_field(base.points, ref, contact_xy=base.contact_xy, step=step)
            log(f"reference: {ref.outline}")
        else:
            cur = frames.mask_clamp_band(base.field, sheet)
        _save_field(cur, os.path.join(r0, "field_reference"), profile)
        _save_field(target, os.path.join(out, "target"), profile)
        state = ST.LoopState(max_passes, robust)
        state.record_initial(ST.compare(cur, target, robust))
        first_round = 1
    last_field = os.path.join(out, "round_00", "field_reference.npz") if not resume else saved["last_field"]
    r = first_round
    while True:
        if state.check_stop():
            break
        rd = os.path.join(out, f"round_{r:02d}")
        if rig:
            res = planner.plan_fields(cur, target, profile.history(state.history), rd)
        else:
            res = P.predict(cur, target, state.history, planner, rd, schema=schema)
        if res["was"] is None:
            state.stopped, state.reason = True, "plan_failure"
            break
        first = res["was"][0]
        k = state.block_size(first, n)
        if k == 0:
            state.stopped, state.reason = True, "next_pass_would_overshoot"
            break
        tps, problems = spec.plan_to_toolpaths([{**first, "n": k}], sheet, allow_rot90=allow_rot90)
        if problems:
            state.stopped, state.reason = True, f"not_executable: {problems[0][1]}"
            break
        doc = spec.toolpath_doc(sheet, tps, note=f"round {r}: first entry of the model plan, {k} pass(es)")
        spec.write_toolpath_json(os.path.join(rd, "toolpath.json"), doc)
        spec.preview_png(doc, os.path.join(rd, "preview.png"))
        m = state.scans[-1]
        log(f"round {r}: amp {m['amp_ratio']:.3f}, corr {m['pattern_corr']:.4f} -> {k} pass(es) of "
            f"WA ({first['cx']:.0f},{first['cy']:.0f}) {first['wx']:.0f}x{first['wy']:.0f} rot {first['rot']} N {first['N']}")
        if confirm and input(f"  execute round {r} ({rd}/preview.png)? [y/N] ").strip().lower() != "y":
            state.stopped, state.reason = True, "stopped_by_operator"
            break
        caps, executed = plant.execute(doc, rd)
        state.record_executed(first, executed)
        stop_now = None
        if rig:
            # cadence of 2026-10-28: the robot runs the whole block, ONE capture at the scan pose; the controller
            # is updated once per block (the block gain split per pass with the 0.8 decay, state.record_block)
            for i, cap in enumerate(caps):
                fld = measurer.field(cap.pass_dir)
                name = os.path.join(rd, f"field_{cap.toolpath_id}_capture{cap.pass_index}")
                _save_field(fld, name, profile)
                last_field, cur = name + ".npz", fld
                mt = ST.compare(fld, target, robust)
                final = i == len(caps) - 1
                if final:
                    state.record_block(mt, first, r, cap.toolpath_id, cap.pass_index, executed)
                log(f"  capture {cap.pass_index} ({'block measurement after %d pass(es)' % executed if final else 'intermediate, record only'}): "
                    f"amp {mt['amp_ratio']:.3f} corr {mt['pattern_corr']:.4f} z {fld.zrange()[0]:.1f}..{fld.zrange()[1]:.1f} mm "
                    f"coverage {fld.meta.get('coverage_observed')}")
            if not caps:
                stop_now = f"no capture after round {r}"
            elif executed < k:
                stop_now = f"robot executed {executed}/{k} pass(es)"
        else:
            for cap in caps:
                fld = cap.field if cap.points is None else S.scan_to_field(cap.points, ref, contact_xy=cap.contact_xy, step=step)
                fld = frames.mask_clamp_band(fld, sheet)
                name = os.path.join(rd, f"field_{cap.toolpath_id}_pass{cap.pass_index}")
                _save_field(fld, name, profile)
                last_field, cur = name + ".npz", fld
                mt = ST.compare(fld, target, robust)
                state.record_pass(mt, first, r, cap.toolpath_id, cap.pass_index)
                pc = fld.meta.get("pose_check")
                log(f"  pass {cap.pass_index}: amp {mt['amp_ratio']:.3f} corr {mt['pattern_corr']:.4f}"
                    + (f" coverage {fld.meta['coverage_usable']:.2f} pose check {pc}" if pc else ""))
                if pc is not None and pc.get("ok") is False:
                    stop_now = "pose_check_failed (sheet not where the robot motion says: slip, capture position or axis sign)"
            if len(caps) < executed or executed < k:
                stop_now = stop_now or f"robot executed {executed}/{k} pass(es), {len(caps)} capture(s)"
        json.dump({"round": r, "last_field": last_field, "state": state.to_dict(), "plant": plant.state()},
                  open(st_path, "w"), indent=1)
        if stop_now:
            state.stopped, state.reason = True, stop_now
            break
        r += 1
    json.dump({"round": r, "last_field": last_field, "state": state.to_dict(), "plant": plant.state()},
              open(st_path, "w"), indent=1)
    summary(out, state, target, HeightField.load(last_field))
    log(f"stopped: {state.reason}; {state.exec_passes} passes; last amp {state.scans[-1]['amp_ratio']:.3f} "
        f"corr {state.scans[-1]['pattern_corr']:.4f}")
    return state


def summary(out, state, target, last):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sc = state.scans
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, f, t in ((ax[0], target, "target"), (ax[1], last, f"final ({state.exec_passes} passes)")):
        im = a.imshow(f.z_mm, origin="lower", cmap="viridis",
                      extent=(f.x_mm[0], f.x_mm[-1], f.y_mm[0], f.y_mm[-1]))
        a.set_title(t, fontsize=9); plt.colorbar(im, ax=a, fraction=0.046)
    e = [s["exec_passes"] for s in sc]
    ax[2].plot(e, [s["amp_ratio"] for s in sc], "o-", label="amplitude ratio")
    ax[2].plot(e, [s["pattern_corr"] for s in sc], "s-", label="shape correlation")
    ax[2].axhline(ST.AMP_OK, color="grey", ls=":"); ax[2].set_xlabel("passes executed")
    ax[2].set_title(f"stopped: {state.reason}", fontsize=8); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "summary.png"), dpi=110); plt.close(fig)
