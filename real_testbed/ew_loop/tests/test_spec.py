"""spec.py checks (plain python, run: conda run -n EW python -m ew_loop.tests.test_spec).

1. our pivots (planner frame) == the robot planner's own plan_zigzag() output, point for point,
   for cx = 0 (their planner has no x offset) — rot 0 and rot 90.
2. an x offset shifts X by PLANNER_X_SIGN * cx and nothing else.
3. infeasible entries are refused with a reason (rot 90 by default, out of zone, N < 3).
4. toolpath.json round trip + preview.
"""
import itertools
import os
import sys
import tempfile

import numpy as np

from ew_loop import spec, frames
from ew_loop.frames import Sheet

# the robot's own planner: EW_ROBOT_CODE = the folder holding zigzag_planner.py (on the robot PC the DoBot
# Scripts folder, i.e. the planner the robot runs today); default = the clone next to this repository
ROBOT_CODE = os.environ.get("EW_ROBOT_CODE") or os.path.join(os.path.dirname(frames.__file__), "..", "Engish_Wheel_T109", "code")
sys.path.insert(0, os.path.abspath(ROBOT_CODE))
import zigzag_planner as zp   # the robot's planner (pure python)


def their_pivots(tp, sheet):
    r = tp["robot"]
    piv = zp.plan_zigzag(x_dim=sheet.width_mm, y_dim=sheet.usable_length_mm,
                         zigzag_lv_mm=r["zigzag_lv_mm"], zigzag_alpha_deg=r["zigzag_alpha_deg"],
                         zigzag_N=r["zigzag_N"], rotation_deg=r["rotation_deg"],
                         resolution_mm=None, start_mm=r["start_mm"])
    return np.asarray(piv, dtype=float)


def test_matches_robot_planner():
    sheet = Sheet()                       # 195 x 190, clamp 100 -> wheel line y=+5, safe zone y in [5, 83] (sheet frame)
    worst, n = 0.0, 0
    for wy, wx, N, rot, cy in itertools.product([60.0, 80.0], [60.0, 120.0, 160.0], [4, 8, 12], [0, 90], [38.0, 42.0]):
        wa = dict(cx=0.0, cy=cy, wx=wx, wy=wy, rot=rot, N=N, n=2)
        try:
            tp = spec.wa_to_toolpath(wa, sheet, allow_rot90=True)
        except spec.Infeasible:
            continue
        ours, theirs = np.array(tp["pivots_xy_mm"]), their_pivots(tp, sheet)
        assert ours.shape == theirs.shape, (ours.shape, theirs.shape)
        worst = max(worst, np.abs(ours - theirs).max()); n += 1
    assert n >= 10, n
    assert worst < 1e-3, worst   # the robot block rounds alpha to 1e-4 deg (logging only); executed pivots are exact
    print(f"[ok] {n} toolpaths equal the robot planner's pivots (max diff {worst:.1e} mm)")


def test_x_offset():
    sheet = Sheet()
    a = spec.wa_to_toolpath(dict(cx=0.0, cy=40.0, wx=100.0, wy=70.0, rot=0, N=6, n=1), sheet)
    b = spec.wa_to_toolpath(dict(cx=20.0, cy=40.0, wx=100.0, wy=70.0, rot=0, N=6, n=1), sheet)
    d = np.array(b["pivots_xy_mm"]) - np.array(a["pivots_xy_mm"])
    assert np.allclose(d[:, 0], frames.PLANNER_X_SIGN * 20.0) and np.allclose(d[:, 1], 0.0), d
    assert b["robot"]["center_x_mm"] == frames.PLANNER_X_SIGN * 20.0
    # frame conversion: WA centre y maps to planner Y = y + L/2 - clamp
    assert abs((40.0 + 95.0 - 100.0) - (a["robot"]["start_mm"] + a["ours"]["lv_mm"] / 2)) < 1e-9
    print("[ok] x offset and frame conversion")


def test_infeasible():
    sheet = Sheet()
    def refused(wa, **kw):
        try:
            spec.wa_to_toolpath(wa, sheet, **kw)
        except spec.Infeasible as e:
            return str(e)
        return None
    assert "rot=90" in refused(dict(cx=0, cy=40, wx=100, wy=70, rot=90, N=6, n=1))
    assert refused(dict(cx=0, cy=40, wx=100, wy=70, rot=90, N=6, n=1), allow_rot90=True) is None
    assert "safe zone" in refused(dict(cx=0, cy=0, wx=100, wy=70, rot=0, N=6, n=1))    # centred on the sheet -> into the clamp band
    assert "safe zone" in refused(dict(cx=0, cy=60, wx=100, wy=70, rot=0, N=6, n=1))   # top edge beyond y=83
    assert "safe zone" in refused(dict(cx=60, cy=40, wx=100, wy=70, rot=0, N=6, n=1))  # x beyond 85.5
    assert "N=2" in refused(dict(cx=0, cy=40, wx=100, wy=70, rot=0, N=2, n=1))
    assert "hits" in refused(dict(cx=0, cy=40, wx=100, wy=70, rot=0, N=6, n=0))
    tps, probs = spec.plan_to_toolpaths([dict(cx=0, cy=40, wx=100, wy=70, rot=0, N=6, n=2),
                                         dict(cx=0, cy=40, wx=100, wy=70, rot=90, N=6, n=1),
                                         dict(cx=0, cy=40, wx=100, wy=70, rot=0, N=8, n=1)], sheet)
    assert len(tps) == 1 and probs == [(1, probs[0][1])] and "rot=90" in probs[0][1]
    print("[ok] infeasible entries refused:", probs)


def test_json_roundtrip_and_preview():
    sheet = Sheet()
    tps, _ = spec.plan_to_toolpaths([dict(cx=-10, cy=38, wx=120, wy=70, rot=0, N=8, n=3),
                                     dict(cx=15, cy=45, wx=80, wy=60, rot=0, N=5, n=1)], sheet)
    assert len(tps) == 2
    doc = spec.toolpath_doc(sheet, tps, note="test")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "toolpath.json"); spec.write_toolpath_json(p, doc)
        back = spec.read_toolpath_json(p)
        assert back["toolpaths"][0]["pivots_xy_mm"] == tps[0]["pivots_xy_mm"]
        assert Sheet.from_dict(back["sheet"]) == sheet
        spec.preview_png(back, os.path.join(d, "preview.png"))
        assert os.path.getsize(os.path.join(d, "preview.png")) > 1000
    print("[ok] toolpath.json round trip + preview; odd N accepted (N=5)")


def test_tilted():
    """schema v8 areas tilted by theta (2026-10-30): refused by default; with max_angle the pivots are the theta = 0
    zigzag turned about the area centre (CCW in the sheet frame, theta > 0 = legs' upper end toward -x), checked
    against the per-angle centre rectangle and the gripped-edge clearance"""
    from ew_loop.model_io import read_entry
    sheet = Sheet(width_mm=231.0, length_mm=174.0, thickness_mm=1.0, clamp_mm=12.0, margin_side_mm=43.4, margin_bottom_mm=0.0,
                  margin_top_mm=12.0, pad_offsets_from_left_mm=(6.5, 224.5), pad_diameter_mm=13.0)
    e = lambda cx, cy, th, rot=0: read_entry({"wa": {"center_xy": [cx, cy], "size_wh": [50, 45], "angle": th}, "zigzag": {"rot": rot, "N": 12}, "hits": 2})
    for kw, wa in (({}, e(-40, 10, 30)), ({"max_angle": 30}, e(-40, 10, 45)), ({"max_angle": 45, "allow_rot90": True}, e(0, 0, 30, 90)),
                   ({"max_angle": 45}, e(45, 0, 45)), ({"max_angle": 45}, e(0, -25, 45))):
        try:
            spec.wa_to_toolpath(wa, sheet, **kw); raise AssertionError(f"accepted {wa} {kw}")
        except spec.Infeasible:
            pass
    for th in (15, -15, 30, -30, 45, -45):
        wa = e(-40, 10, th)
        tp = spec.wa_to_toolpath(wa, sheet, max_angle=45)
        flat = spec.wa_pivots_sheet({**wa, "rot": 0, "zz_rot": 0, "angle": 0}) - [wa["cx"], wa["cy"]]
        t = np.radians(th); R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        assert np.allclose(sheet.from_planner(np.array(tp["pivots_xy_mm"]))[::-1], flat @ R.T + [wa["cx"], wa["cy"]], atol=1e-3)
        assert tp["robot"]["rotation_deg"] == th and (tp["ours"]["zz_rot"], tp["ours"]["angle"]) == (0, th)
        assert tp["checks"]["gripped_edge_clearance_mm"] >= spec.EDGE_CLEARANCE_MM
    leg = spec.wa_pivots_sheet(e(0, 0, 30))[1] - spec.wa_pivots_sheet(e(0, 0, 30))[0]
    assert leg[0] < 0 < leg[1]
    s = spec.normalise_wa({"cx": -40, "cy": 10, "wx": 50, "wy": 45, "rot": 0, "angle": 45, "N": 12, "n": 2})   # a scripted block
    assert (s["rot"], s["zz_rot"], s["angle"]) == (45, 0, 45) and spec.normalise_wa(s) == s
    assert spec.wa_to_toolpath(s, sheet, max_angle=45)["pivots_xy_mm"] == spec.wa_to_toolpath(e(-40, 10, 45), sheet, max_angle=45)["pivots_xy_mm"]
    with tempfile.TemporaryDirectory() as d:
        doc = spec.toolpath_doc(sheet, [spec.set_directions(spec.wa_to_toolpath(e(-40, 10, 45), sheet, max_angle=45), "l2r")])
        spec.preview_png(doc, os.path.join(d, "preview.png"))
    print("[ok] tilted areas: refused by default / outside the centre rectangle / with rot 90; pivots = the turned zigzag; "
          "scripted {rot, angle} == read_entry v8")


if __name__ == "__main__":
    test_matches_robot_planner(); test_x_offset(); test_infeasible(); test_json_roundtrip_and_preview(); test_tilted()
    print("ALL SPEC TESTS PASSED")
