"""No-robot check of the manual wheel mode (manual_wheel.py) against Danish's REAL forming module, as
test_continuous_block.py: his run_zigzag_forming() runs with the robot calls recorded, a simulated load cell, and a simulated
operator who turns the wheel and confirms. Checked: the block loads once and unloads once BY HAND (the motor functions are
never called), the robot does not move during a wait, a load outside the window is not accepted, auto mode continues by
itself, and a stop leaves the robot where it is.

    python robot_dobot/test_manual_wheel.py --scripts <folder with zigzag_forming_oldclamps.py, zigzag_planner.py, run_output.py>
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manual_wheel as MW                       # noqa: E402
import test_continuous_block as TCB             # noqa: E402


class LoadCell:
    """the wheel force the operator produces by hand (N); the motor functions raise if anything calls them"""
    _FORCE_THRESHOLD = 250.0

    def __init__(self):
        self.f = 0.0

    def _tare_wheel_force_sensor(self, robot=None):
        pass

    def _read_wheel_force_n(self, robot=None):
        return self.f

    def _motor(self, *a, **k):
        raise AssertionError("the wheel motor was driven in manual mode")
    close_wheel_until_force = open_wheel_by_last_close_count = unload_wheel_if_loaded = _motor


def run(scripts, tp_path, mode, operator):
    """his forming run with the manual wheel; operator(what, cell) acts in a thread when a wait starts"""
    zf, zp, rtd = TCB.load(scripts)
    ev, cell = [], LoadCell()
    d = tempfile.mkdtemp(); open(os.path.join(d, "run_info.txt"), "w").write("Run Date: test\n"); zf._output_dir = d
    zf.capture_pointcloud = lambda path: (os.makedirs(path, exist_ok=True), ev.append("SCAN " + os.path.basename(path)))
    zf._rtde = lambda name, *a: [0.0] * 6
    zf._movel_single = lambda pose, *a, **k: ev.append("MOVE")
    zf._arc_single = lambda *a, **k: ev.append("MOVE"); zf._send_sequence = lambda *a, **k: ev.append("MOVE")
    zf._wait_stopped = lambda *a, **k: None
    o_exec = zf._execute_pass
    zf._execute_pass = lambda *a, **k: (ev.append("PASS BEGIN"), o_exec(*a, **k), ev.append("PASS END"))[1]
    doc = json.load(open(tp_path)); tp = doc["toolpaths"][0]
    piv = rtd.configure(zf, zp, doc, tp, "run", 12.0, True, "l2r"); rtd.quiet(zf, None, "run")

    def log(*x):
        line = " ".join(str(v) for v in x)
        for what in ("LOAD", "UNLOAD"):
            if f"===== {what} the wheel" in line:
                ev.append(f"WAIT {what}")
                threading.Thread(target=operator, args=(what, cell), daemon=True).start()
        if "accepted at" in line:
            ev.append("ACCEPT " + line.split("]")[1].split(" accepted")[0].strip().upper())
    mw = MW.install(zf, cell, mode, log=log)
    coincide, _ = rtd.turnaround_coincides(zf, piv)
    st = rtd.continuous_block(zf, zf.NUM_PASSES, coincide, log=lambda *a: None, wf=cell)
    err = None
    try:
        zf.run_zigzag_forming()
    except RuntimeError as e:
        err = str(e)
    return ev, mw, st, err


def compact(ev):
    out = []
    for e in ev:
        if out and out[-1].startswith(e + " x"):
            out[-1] = f"{e} x{int(out[-1].split(' x')[1]) + 1}"
        else:
            out.append(e if e != "MOVE" else "MOVE x1")
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scripts", required=True); ap.add_argument("--toolpath")
    a = ap.parse_args()
    home = tempfile.mkdtemp()
    MW.RESUME, MW.STOP = os.path.join(home, "EW_RESUME"), os.path.join(home, "EW_STOP")
    MW.HOLD_S, MW.COUNTDOWN_S, MW.PRINT_S = 0.5, 1.0, 0.5
    MW._beep = lambda: None
    tpath = a.toolpath
    if not tpath:
        sys.path.insert(0, os.path.dirname(HERE))
        from ew_loop import spec
        from ew_loop.frames import Sheet
        sheet = Sheet(width_mm=231.0, length_mm=174.0, thickness_mm=1.0, clamp_mm=12.0, margin_side_mm=43.4, margin_bottom_mm=0.0, margin_top_mm=12.0,
                      pad_offsets_from_left_mm=(6.5, 224.5), pad_diameter_mm=13.0)
        tps, _ = spec.plan_to_toolpaths([dict(cx=-20.0, cy=25.0, wx=60.0, wy=90.0, rot=0, N=6, n=3)], sheet)
        tpath = os.path.join(tempfile.mkdtemp(), "toolpath.json"); spec.write_toolpath_json(tpath, spec.toolpath_doc(sheet, tps))

    def turn(cell, to, secs=0.6):
        f0 = cell.f
        for i in range(1, 7):
            cell.f = f0 + (to - f0) * i / 6; time.sleep(secs / 6)

    def touch(p):
        open(p, "w").close()

    # confirm: the operator first stops at 200 N and confirms (refused), then goes to 250 N and confirms; unloads to 5 N
    def op_confirm(what, cell):
        time.sleep(0.3)
        if what == "LOAD":
            turn(cell, 200.0); touch(MW.RESUME); time.sleep(0.5); turn(cell, 250.0); touch(MW.RESUME)
        else:
            turn(cell, 5.0); touch(MW.RESUME)
    ev, mw, st, err = run(a.scripts, tpath, "confirm", op_confirm)
    s = compact(ev); print("confirm:", " | ".join(s))
    assert err is None, err
    assert s.count("WAIT LOAD") == 1 and s.count("WAIT UNLOAD") == 1, s
    i_wl, i_al = s.index("WAIT LOAD"), s.index("ACCEPT LOAD")
    assert i_al == i_wl + 1 and s[i_al + 1] == "PASS BEGIN", ("the robot moved between the load wait and its acceptance", s)
    i_wu, i_au = s.index("WAIT UNLOAD"), s.index("ACCEPT UNLOAD")
    assert i_au == i_wu + 1 and s[i_wu - 1] == "PASS END", ("the robot moved during the unload wait", s)
    assert s.count("PASS BEGIN") == 3 and [e for e in s if e.startswith("SCAN")] == ["SCAN pointcloud_pass3"], s
    assert [e["what"] for e in mw.events] == ["load", "unload"] and 230 <= mw.events[0]["force_N"] <= 270, mw.events
    print(f"[ok] confirm: 200 N refused, 250 N accepted; load once, 3 passes, unload once, one scan; events {mw.events}")

    # auto: no confirmation at all, the force alone lets the run go on
    def op_auto(what, cell):
        time.sleep(0.3); turn(cell, 252.0 if what == "LOAD" else 3.0)
    ev, mw, st, err = run(a.scripts, tpath, "auto", op_auto)
    s = compact(ev); print("auto:   ", " | ".join(s))
    assert err is None and [e["what"] for e in mw.events] == ["load", "unload"] and s.count("PASS BEGIN") == 3, (err, s)
    print(f"[ok] auto: continued by itself after hold + countdown; events {mw.events}")

    # stop during the load wait: the robot stays where it is, nothing else happens
    def op_stop(what, cell):
        time.sleep(0.3); turn(cell, 120.0); touch(MW.STOP)
    ev, mw, st, err = run(a.scripts, tpath, "confirm", op_stop)
    s = compact(ev); print("stop:   ", " | ".join(s))
    assert err and "stopped by the operator" in err and "PASS BEGIN" not in s and s[-1] == "WAIT LOAD", (err, s)
    print(f"[ok] stop: run ended at the load wait, no pass, no further move ({err})")
    print("ALL MANUAL-WHEEL CHECKS PASSED")


if __name__ == "__main__":
    main()
