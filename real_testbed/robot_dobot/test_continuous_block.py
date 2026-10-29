"""No-robot check of run_toolpath_dobot's block execution against Danish's REAL forming module: his
run_zigzag_forming() runs with only the lowest-level robot calls recorded (motion, wheel load/unload, scan), and
the recorded sequence is checked: load once, the passes back to back with the next pass starting where the last
ended, unload once, ONE scan at the end; and the fallback (ends do not coincide) runs his unload/reposition/reload.

    python robot_dobot/test_continuous_block.py --scripts <folder with zigzag_forming_oldclamps.py, zigzag_planner.py, run_output.py>
"""
import argparse
import importlib
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def load(scripts):
    sys.modules.setdefault("dobot_sdk", types.SimpleNamespace(DobotRobot=object, CoordinateType=object))
    sys.path.insert(0, scripts)
    import matplotlib; matplotlib.use("Agg")
    for m in ("zigzag_forming_oldclamps", "zigzag_planner", "run_output"):
        sys.modules.pop(m, None)
    zf = importlib.import_module("zigzag_forming_oldclamps")
    zp = importlib.import_module("zigzag_planner")
    spec = importlib.util.spec_from_file_location("rtd", os.path.join(HERE, "run_toolpath_dobot.py"))
    rtd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rtd)
    return zf, zp, rtd


def run_block(scripts, toolpath, start_direction, force_no_coincide=False):
    zf, zp, rtd = load(scripts)
    ev = []
    d = tempfile.mkdtemp()
    open(os.path.join(d, "run_info.txt"), "w").write("Run Date: test\n")
    zf._output_dir = d
    zf.close_wheel_until_force = lambda retare=True: (ev.append("LOAD"), 1067)[1]
    zf.open_wheel_by_last_close_count = lambda pulse_count=None: ev.append(f"UNLOAD {pulse_count}")
    zf.capture_pointcloud = lambda path: (os.makedirs(path, exist_ok=True), ev.append("SCAN " + os.path.basename(path)))
    zf._rtde = lambda name, *a: [0.0] * 6
    legs = []
    zf._movel_single = lambda pose, *a, **k: ev.append(("MOVE", [round(v, 5) for v in pose]))
    zf._arc_single = lambda mid, end, *a, **k: ev.append(("ARC", [round(v, 5) for v in end]))
    zf._send_sequence = lambda poses, *a, **k: ev.append(("SEQ", [round(v, 5) for v in poses[-1]]))
    zf._wait_stopped = lambda *a, **k: None
    o_exec = zf._execute_pass
    zf._execute_pass = lambda *a, **k: (ev.append("PASS BEGIN " + k.get("label", "")), o_exec(*a, **k), ev.append("PASS END"))[1]
    doc = json.load(open(toolpath)); tp = doc["toolpaths"][0]
    piv = rtd.configure(zf, zp, doc, tp, "run", 12.0, True, start_direction)
    rtd.quiet(zf, None, "run")
    coincide, gaps = rtd.turnaround_coincides(zf, piv)
    if force_no_coincide:
        coincide = False
    st = rtd.continuous_block(zf, zf.NUM_PASSES, coincide, log=lambda *a: None)
    zf.run_zigzag_forming()
    return ev, st, gaps, zf


def summarize(ev):
    out = []
    for e in ev:
        if isinstance(e, tuple):
            if out and out[-1].startswith(e[0] + " x"):
                n = int(out[-1].split(" x")[1]) + 1; out[-1] = f"{e[0]} x{n}"
            else:
                out.append(f"{e[0]} x1")
        else:
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scripts", required=True); ap.add_argument("--toolpath")
    a = ap.parse_args()
    tpath = a.toolpath
    if not tpath:
        sys.path.insert(0, os.path.dirname(HERE))
        from ew_loop import spec
        from ew_loop.frames import Sheet
        sheet = Sheet(width_mm=231.0, length_mm=174.0, thickness_mm=1.0, clamp_mm=12.0, margin_side_mm=43.4, margin_bottom_mm=0.0, margin_top_mm=12.0,
                      pad_offsets_from_left_mm=(6.5, 224.5), pad_diameter_mm=13.0)
        tps, _ = spec.plan_to_toolpaths([dict(cx=-20.0, cy=25.0, wx=60.0, wy=90.0, rot=0, N=6, n=3)], sheet)
        tpath = os.path.join(tempfile.mkdtemp(), "toolpath.json"); spec.write_toolpath_json(tpath, spec.toolpath_doc(sheet, tps))
    for start in ("l2r", "r2l"):
        ev, st, gaps, zf = run_block(a.scripts, tpath, start)
        s = summarize(ev)
        print(f"--- 3 passes, start {start}, turnaround {gaps}:\n   " + " | ".join(s))
        assert ev.count("LOAD") == 1, s
        assert sum(1 for e in ev if isinstance(e, str) and e.startswith("UNLOAD")) == 1, s
        assert [e for e in ev if isinstance(e, str) and e.startswith("SCAN")] == ["SCAN pointcloud_pass3"], s
        assert st["done"] == 3 and ev.index("LOAD") < ev.index("PASS BEGIN pass-1 (left -> right) ") < [i for i, e in enumerate(ev) if isinstance(e, str) and e.startswith("UNLOAD")][0], s
        # the move before each pass >= 2 is the previous pass's last move (no reposition), and nothing between the passes
        ends = [i for i, e in enumerate(ev) if e == "PASS END"]; begins = [i for i, e in enumerate(ev) if isinstance(e, str) and e.startswith("PASS BEGIN")]
        for e_i, b_i in zip(ends[:-1], begins[1:]):
            assert b_i == e_i + 1, ("something between passes", ev[e_i:b_i + 1])
        print(f"[ok] start {start}: LOAD once, 3 passes back to back, UNLOAD once, one SCAN (pointcloud_pass3); skipped: {st['skipped']}")
    ev, st, gaps, zf = run_block(a.scripts, tpath, "l2r", force_no_coincide=True)
    s = summarize(ev)
    print(f"--- fallback (ends treated as not coinciding):\n   " + " | ".join(s))
    assert ev.count("LOAD") == 3 and sum(1 for e in ev if isinstance(e, str) and e.startswith("UNLOAD")) == 3, s
    assert [e for e in ev if isinstance(e, str) and e.startswith("SCAN")] == ["SCAN pointcloud_pass3"], s
    print("[ok] fallback: his unload / reposition / reload between passes, still one scan per block")
    print("ALL CONTINUOUS-BLOCK CHECKS PASSED")


if __name__ == "__main__":
    main()
