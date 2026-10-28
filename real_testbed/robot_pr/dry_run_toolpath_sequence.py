"""Robot side + loop, end to end, WITHOUT a robot (our pre-PR self-test, Linux).

 1. exports code/ of origin/oct_2_demo from the clone into a scratch folder, applies
    zigzag_forming_toolpath_api.patch, copies run_toolpath_sequence.py next to it
 2. `plan` on a 2-toolpath toolpath.json, as the operator would run it
 3. a stub runner that executes run_toolpath_sequence.py (baseline / run) with every
    robot-only call faked (RTDE, motion sockets, wheel, cameras). Its fake camera writes a
    point cloud of the sheet placed where the robot's OWN kinematics put it at capture time
    (the commanded TCP pose, inverted through zigzag_forming's WHEEL_CENTER and leveled axes),
    so the loop's capture-position convention is checked against the robot code, not
    against itself
 4. the closed loop (ew_loop.loop with RobotPlant -> the stub runner, a fixed plan instead of
    the model) for several rounds until it stops; every pose check must pass

Usage: conda run -n EW python real_testbed/robot_pr/dry_run_toolpath_sequence.py [--keep DIR]
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REAL_TESTBED = HERE.parent
CLONE = REAL_TESTBED / "Engish_Wheel_T109"
sys.path.insert(0, str(REAL_TESTBED))
from ew_loop import spec, frames, loop as L, predict as P            # noqa: E402
from ew_loop.frames import Sheet                                      # noqa: E402
from ew_loop.synthetic import FakeRig, bent_surface                   # noqa: E402

STUB_RUNNER = r'''
"""run_toolpath_sequence.py with the robot faked (see dry_run_toolpath_sequence.py)."""
import os, sys, types, time
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
from ew_loop.frames import Sheet
from ew_loop.synthetic import FakeRig, bent_surface, flat_surface

last = {"pose": None, "n_movel": 0, "n_seg": 0}
class _Env:
    def __init__(self, d): self.output_dir = d
    def startup_movel(self, host, target, a=0.1, v=0.1): last["pose"] = list(target)
    def wait_until_stopped(self, settle=0.0): pass
fake = types.ModuleType("robot_setup")
fake.setup_robot_environment = lambda output_dir=None, tag="x", enable_apriltag=False: _Env(output_dir)
fake.teardown_robot_environment = lambda env: None
fake.ROBOT_HOST, fake.ACCELERATION, fake.VELOCITY, fake.target_start_0 = "0.0.0.0", 0.1, 0.1, [0.5, 0.2, 0.3, 0, 0, 0]
sys.modules["robot_setup"] = fake

import zigzag_forming as zf
def _movel(pose, **k): last["pose"] = list(pose); last["n_movel"] += 1
def _segs(segments, **k): last["pose"] = list(segments[-1][0][-1]); last["n_seg"] += 1
zf._movel_single, zf._send_segments = _movel, _segs
zf._movej_single = lambda q, **k: None
zf._wait_stopped = lambda **k: None
zf._check_safety_gate = lambda: None
zf._rtde = lambda name, *a: {"getActualTCPPose": last["pose"], "getActualQ": [0.0] * 6,
                             "getActualQd": [0.0] * 6, "getSafetyMode": 1}[name]
zf.close_wheel_until_force = lambda retare=True: 42
zf.open_wheel_by_last_close_count = lambda frequency=None, pulse_count=None: None
time.sleep = lambda s: None

SHEET = Sheet()
RIG = FakeRig(SHEET, seed=int(time.time() * 1000) % 100000)
COUNTER = os.environ["EW_STUB_COUNTER"]
WC = np.array(zf.WHEEL_CENTER, float)
WD, FD = np.array(zf._WIDTH_DIR_LEVELED, float), np.array(zf._FEED_DIR_LEVELED, float)

def contact_from_robot():
    """planner point under the wheel from the commanded TCP pose (inverse of
    zigzag_forming._contact_to_tcp_pose for a flat sheet at leg angle 0)."""
    d = np.array(last["pose"][:3], float) - WC
    return np.array([(d @ WD) * 1000.0, -(d @ FD) * 1000.0])

def fake_capture(path):
    n = int(open(COUNTER).read()) if os.path.exists(COUNTER) else 0
    is_pass = os.path.basename(path).startswith("pointcloud_pass")
    if is_pass:
        n += 1
        open(COUNTER, "w").write(str(n))
    contact = SHEET.from_planner(contact_from_robot())
    surf = bent_surface(SHEET, amp_mm=1.5 * n) if n else flat_surface
    Pts = RIG.capture(surf, contact).astype(np.float32)
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(Pts)}\n"
           "property float x\nproperty float y\nproperty float z\nend_header\n")
    os.makedirs(path, exist_ok=True)
    open(os.path.join(path, "device_3_pointcloud_001.ply"), "wb").write(hdr.encode() + Pts.tobytes())
    with open(os.environ["EW_STUB_LOG"], "a") as f:
        f.write(f"{path}\t{contact[0]:.3f}\t{contact[1]:.3f}\n")
    return path
zf.capture_pointcloud = fake_capture

import run_toolpath_sequence as rts
rts.main(sys.argv[1:])
print(f"[stub] motion commands: {last['n_movel']} movel, {last['n_seg']} leg sequences")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=None, help="keep the scratch folder here instead of a temp dir")
    a = ap.parse_args()
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="ew_dryrun_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    # 1. fresh robot code + patch + entry script + stub runner
    subprocess.run(f"git -C {CLONE} archive origin/oct_2_demo code | tar -x -C {work}", shell=True, check=True)
    subprocess.run(["patch", "-p1", "-s", "-d", str(work)], stdin=open(HERE / "zigzag_forming_toolpath_api.patch"), check=True)
    code = work / "code"
    shutil.copy(HERE / "run_toolpath_sequence.py", code)
    (code / "_stub_runner.py").write_text(STUB_RUNNER)
    os.environ.update({"MPLBACKEND": "Agg", "PYTHONPATH": f"{code}{os.pathsep}{REAL_TESTBED}",
                       "EW_STUB_COUNTER": str(work / "stub_passes.txt"), "EW_STUB_LOG": str(work / "stub_captures.tsv")})
    py = sys.executable
    sheet = Sheet()

    # 2. plan mode on a two-zigzag toolpath.json
    tps, problems = spec.plan_to_toolpaths([dict(cx=-10.0, cy=38.0, wx=130.0, wy=75.0, rot=0, N=8, n=3),
                                            dict(cx=20.0, cy=45.0, wx=90.0, wy=60.0, rot=0, N=5, n=2)], sheet)
    assert not problems, problems
    tp_json = work / "toolpath.json"
    spec.write_toolpath_json(tp_json, spec.toolpath_doc(sheet, tps, note="dry run"))
    r = subprocess.run([py, "run_toolpath_sequence.py", "plan", str(tp_json), "--out", str(work / "plan_out")],
                       cwd=code, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for i, tp in enumerate(tps):
        sub = work / "plan_out" / f"tp{i}_{tp['id']}"
        assert (sub / "zigzag_preview.png").stat().st_size > 1000
        for k in range(1, tp["passes"] + 1):
            assert len(list(csv.reader(open(sub / f"zigzag_path_pass{k}.csv")))) > len(tp["pivots_xy_mm"])
    print(f"[ok] plan: previews and per-pass commanded-path CSVs for {len(tps)} toolpaths")

    # 3. stubbed baseline + run; capture positions from the robot's kinematics vs our convention
    r = subprocess.run([py, "_stub_runner.py", "baseline", "--out", str(work / "base_out")], cwd=code, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    r = subprocess.run([py, "_stub_runner.py", "run", str(tp_json), "--out", str(work / "run_out"), "--yes"],
                       cwd=code, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rows = [l.split("\t") for l in open(work / "stub_captures.tsv").read().splitlines()]
    assert rows[0][0].endswith("baseline") and np.allclose([float(rows[0][1]), float(rows[0][2])], [0.0, sheet.wheel_line_y], atol=1e-6)
    worst = 0.0
    for i, tp in enumerate(tps):
        for k in range(1, tp["passes"] + 1):
            path = str(work / "run_out" / f"tp{i}_{tp['id']}" / f"pointcloud_pass{k}")
            row = next(rw for rw in rows if rw[0] == path)
            robot_says = np.array([float(row[1]), float(row[2])])
            worst = max(worst, float(np.abs(robot_says - L.capture_contact(tp, k, sheet)).max()))
        assert (work / "run_out" / f"tp{i}_{tp['id']}" / "done.flag").exists()
    print(f"[ok] run: {sum(t['passes'] for t in tps)} passes; capture position from the robot kinematics vs "
          f"ew_loop.loop.capture_contact: max difference {worst:.2e} mm")
    assert worst < 1e-3, worst
    (work / "stub_passes.txt").unlink()

    # 4. the closed loop driving the stubbed robot, fixed plan, fake deformation 1.5 mm/pass
    target = frames.mask_clamp_band(frames.field_from_points(*_grid_points(sheet, bent_surface(sheet, 10.0)), sheet, 2.0), sheet)
    target.save(work / "target.npz")
    plan = {"plan": [{"wa": {"center_xy": [0.0, 40.0], "size_wh": [120.0, 70.0], "rot": 0, "N": 8}, "hits": 12}]}
    json.dump(plan, open(work / "fixed_plan.json", "w"))
    plant = L.RobotPlant(sheet, str(code), py, camera=3, yes=True, script="_stub_runner.py")
    st = L.run_loop(str(work / "loop"), sheet, str(work / "target.npz"), P.FixedPlanner(str(work / "fixed_plan.json")), plant,
                    n=3, max_passes=20, feed_dir_cam=FakeRig(sheet).feed_dir_cam(error_deg=20.0), confirm=False)
    checks = []
    for rd in sorted((work / "loop").glob("round_*")):
        for f in sorted(rd.glob("field_*_pass*.npz")):
            checks.append(frames.HeightField.load(f).meta.get("pose_check", {}).get("ok"))
    print(f"[ok] loop: stopped '{st.reason}' after {st.exec_passes} passes in {len(st.replans)} re-plans; "
          f"pose checks {checks}")
    assert st.reason in ("converged", "next_pass_would_overshoot") and st.scans[-1]["amp_ratio"] > 0.9, st.reason
    assert all(c is True for c in checks), checks
    if a.keep:
        print(f"DRY RUN OK; outputs kept under {work}")
    else:
        shutil.rmtree(work)
        print("DRY RUN OK (outputs removed; --keep DIR keeps them)")


def _grid_points(sheet, surf, step=1.0):
    xs = np.arange(-sheet.width_mm / 2, sheet.width_mm / 2 + 1e-9, step)
    ys = np.arange(-sheet.length_mm / 2, sheet.length_mm / 2 + 1e-9, step)
    X, Y = np.meshgrid(xs, ys)
    return np.stack([X.ravel(), Y.ravel()], 1), surf(X, Y).ravel()


if __name__ == "__main__":
    main()
