"""Model plan entries -> robot toolpaths (toolpath.json), feasibility checks, preview.

A model plan entry is a "wa" dict in the MPC schema (experiments/mpc_end2end/common.read_entry):
    {cx, cy, wx, wy, rot, N, n}     centre / size of the working area (sheet frame, mm),
                                    zigzag orientation (0 or 90), strip count N, n = hits.
Schema v8 (2026-10-30) adds the area's tilt: read_entry gives zz_rot (0 / 90), angle theta and rot = rot_deg =
zz_rot + theta, the simulator's and the driver's rotation; the zigzag is inscribed as at theta = 0 and turned by theta
about the area centre (newsheet_pipeline/zigzag_rot.py). Tilted areas run only with zigzag rot 0 and |theta| <=
max_angle (default 0: refused until Danish has cleared the gripper at that angle, ROTATION_RANGE_2026-10-29.md).
One hit = one simulator cycle = ONE traversal of the zigzag = one robot pass
(zigzag_forming.LEG_REPEAT = 1; a reverse pass counts as a pass too).

toolpath.json (schema 1) is the only contract between the loop and the robot executor
(robot_pr/run_toolpath_sequence.py). The executor consumes `pivots_xy_mm` (planner frame)
and `passes` directly — it does not re-derive the geometry — so the zigzag parametrisation
used by the model/simulator stays the single source of truth. The `robot` block holds the
equivalent zigzag_planner.plan_zigzag() parameters for logging/preview and for the
cross-check in tests/test_spec.py.
"""
import json
import math

import numpy as np

from . import frames
from .frames import Sheet
from .model_io import zigzag_vertices, inscribe, strip_stack_dims, rot_angle, W as STRIP_W

SCHEMA_VERSION = 1
# per-angle reachable centre rectangle, sheet frame, for |theta| up to the row's angle: (|theta|, |cx| max, cy min, cy max)
# (ROTATION_RANGE_2026-10-29.md section 2; cy min includes the gripped-edge margin; IK checked inside, 50 x 50 mm areas)
CENTRE_RECT = ((10, 47.5, -40.0, 50.0), (15, 47.5, -37.5, 50.0), (20, 45.0, -32.5, 47.5), (30, 42.5, -27.5, 47.5),
               (45, 42.5, -20.0, 45.0))
EDGE_CLEARANCE_MM = 12.0      # gripped edge before the wheel line wherever it is within EDGE_LANE_MM of the wheel laterally
EDGE_LANE_MM = 40.0           # (the margin the wheel line keeps at theta = 0; same rule as the table above)


class Infeasible(ValueError):
    """The entry cannot be executed on the robot; str(exc) says why."""


def orientation(wa):
    """(zigzag rot 0 / 90, area angle theta, rotation_deg = rot + theta) of a wa dict: model_io.read_entry's ('zz_rot',
    'angle'; its 'rot' is the sum), a scripted block's {rot 0 / 90, angle}, or the old {rot} / {rot_deg} alone"""
    if "zz_rot" in wa:
        zz, th = int(wa["zz_rot"]), int(round(float(wa.get("angle", 0))))
    elif "angle" in wa:
        zz, th = int(wa["rot"]), int(round(float(wa["angle"])))
    else:
        zz, th = rot_angle(wa)
    return zz, th, zz + th


def normalise_wa(wa):
    """the wa dict with read_entry's keys: rot = rot_deg = zz_rot + angle (the loop's area key and history use these)"""
    zz, th, rd = orientation(wa)
    return {**wa, "rot": rd, "rot_deg": rd, "zz_rot": zz, "angle": th}


def orient_txt(wa):
    zz, th, _ = orientation(wa)
    return f"rot {zz}" + (f" angle {th:+d}" if th else "")


def wa_zigzag(wa):
    """WA box -> (lv_mm, alpha_deg) of the inscribed zigzag (category_tests.inscribe)."""
    rd = orientation(wa)[2]
    a, b = strip_stack_dims(wa["wx"], wa["wy"], rd)
    if wa["N"] < 3:
        raise Infeasible(f"N={wa['N']} < 3")
    if a - STRIP_W <= 0 or b - STRIP_W <= 0:
        raise Infeasible(f"WA {wa['wx']}x{wa['wy']} too small for the {STRIP_W} mm strip width")
    lv, alpha = inscribe(wa["wx"], wa["wy"], rd, wa["N"])
    return float(lv), float(alpha)


def wa_pivots_sheet(wa):
    """(N+1, 2) zigzag corner points in the sheet frame, centred on the WA centre, turned by rotation_deg about it,
    in OUR traversal order (strips swept left to right)."""
    lv, alpha = wa_zigzag(wa)
    v = zigzag_vertices(lv, alpha, int(wa["N"]), float(orientation(wa)[2]))[:, :2]
    return v + np.array([wa["cx"], wa["cy"]], dtype=float)


def wa_corners(wa):
    """the 4 corners (sheet frame) of the area box turned by theta about its centre"""
    t = math.radians(orientation(wa)[1]); c, s = math.cos(t), math.sin(t)
    hx, hy = wa["wx"] / 2.0, wa["wy"] / 2.0
    return np.array([[wa["cx"] + c * u - s * v, wa["cy"] + s * u + c * v] for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])


def leg_heading(p0, p1):
    """heading of a leg in the planner frame as his driver turns the sheet for it (zigzag_forming_oldclamps._leg_theta:
    push legs folded onto pull legs), rad"""
    dx, dz = p1[0] - p0[0], p1[1] - p0[1]
    mag = math.hypot(dx, dz)
    if mag < 1e-9:
        return 0.0
    if dz < 0:
        dx, dz = -dx, -dz
    return math.atan2(dx / mag, dz / mag)


def gripper_clearance(piv_planner, sheet):
    """over every leg end with the sheet turned by that leg's heading about the contact (robot_dobot/rotation_range.py):
    the gripped edge's distance before the wheel line within EDGE_LANE_MM laterally, and how far the gripper pads pass
    the wheel line (positive = past it; the physical yoke clearance is Danish's jog check)"""
    W, L = sheet.width_mm, sheet.length_mm
    edge = sheet.to_planner(np.stack([np.linspace(-W / 2, W / 2, 47), np.full(47, -L / 2)], axis=1))
    pads = sheet.to_planner(np.array(sheet.pad_centres(), dtype=float))
    clear, pad_past, heads = np.inf, -np.inf, []
    for p0, p1 in zip(piv_planner[:-1], piv_planner[1:]):
        h = leg_heading(p0, p1); heads.append(h); c, s = math.cos(h), math.sin(h)
        for cpt in (p0, p1):
            for pts, kind in ((edge, "edge"), (pads, "pads")):
                d = pts - np.asarray(cpt, dtype=float)
                w = np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], axis=1)
                if kind == "edge":
                    near = w[np.abs(w[:, 0]) < EDGE_LANE_MM]
                    if len(near):
                        clear = min(clear, float(-near[:, 1].max()))
                else:
                    pad_past = max(pad_past, float(w[:, 1].max()))
    return {"gripped_edge_clearance_mm": None if clear == np.inf else round(clear, 1),
            "pad_past_wheel_line_mm": round(pad_past, 1),
            "leg_heading_deg": [round(math.degrees(min(heads)), 1), round(math.degrees(max(heads)), 1)]}


def check_tilt(wa, th, max_angle):
    """a tilted area: |theta| cleared, centre inside the per-angle rectangle. Raises Infeasible."""
    if abs(th) > max_angle:
        raise Infeasible(f"area angle {th:+d} deg: tilted areas are cleared up to |angle| {max_angle:g} on this rig "
                         "(--max-angle, after Danish's clearance check)")
    row = next((r for r in CENTRE_RECT if abs(th) <= r[0]), None)
    if row is None:
        raise Infeasible(f"area angle {th:+d} deg: beyond the +-{CENTRE_RECT[-1][0]} deg the rig was checked for")
    a, xm, y0, y1 = row
    if not (abs(wa["cx"]) <= xm + 1e-9 and y0 - 1e-9 <= wa["cy"] <= y1 + 1e-9):
        raise Infeasible(f"area centre ({wa['cx']:.1f},{wa['cy']:.1f}) outside the reachable centre rectangle for "
                         f"|angle| <= {a}: |x| <= {xm}, y {y0}..{y1} (ROTATION_RANGE_2026-10-29.md)")


def check_pivots_planner(pivots_p, sheet):
    """indices of pivots outside the robot safe zone (same rule as zigzag_planner.point_in_safe_zone)."""
    x0, x1, y0, y1 = sheet.safe_zone_planner()
    bad = [i for i, (x, y) in enumerate(pivots_p)
           if not (x0 - 1e-9 <= x <= x1 + 1e-9 and y0 - 1e-9 <= y <= y1 + 1e-9)]
    return bad


def wa_to_toolpath(wa, sheet, idx=0, allow_rot90=False, max_angle=0.0):
    """One model entry -> one executable toolpath dict. Raises Infeasible."""
    rot, th, rd = orientation(wa)
    if rot not in (0, 90):
        raise Infeasible(f"rot={rot} not in (0, 90)")
    if rot == 90 and not allow_rot90:
        raise Infeasible("rot=90 (strips across the feed direction) is not executable on the "
                         "robot yet — the sheet cannot be turned under the wheel")
    if th:
        if rot != 0:
            raise Infeasible(f"area angle {th:+d} with zigzag rot {rot}: tilted areas only with rot 0")
        check_tilt(wa, th, max_angle)
    if int(wa["n"]) < 1:
        raise Infeasible(f"hits={wa['n']} < 1")
    lv, alpha = wa_zigzag(wa)
    piv_sheet = wa_pivots_sheet(wa)
    # the robot planner traverses the same pattern in the opposite order (verified point for
    # point in bridge_checks/check_zigzag_param_mapping.py); hand it its own order
    piv_planner = sheet.to_planner(piv_sheet)[::-1]
    bad = check_pivots_planner(piv_planner, sheet)
    if bad:
        x0, x1, y0, y1 = sheet.safe_zone_planner()
        b = piv_planner[bad]
        raise Infeasible(f"{len(bad)}/{len(piv_planner)} pivots outside the safe zone "
                         f"X[{x0:.1f},{x1:.1f}] Y[{y0:.1f},{y1:.1f}] (planner frame): "
                         f"X[{b[:,0].min():.1f},{b[:,0].max():.1f}] Y[{b[:,1].min():.1f},{b[:,1].max():.1f}]")
    checks = None
    if th:
        checks = gripper_clearance(piv_planner, sheet)
        if checks["gripped_edge_clearance_mm"] is not None and checks["gripped_edge_clearance_mm"] < EDGE_CLEARANCE_MM - 1e-9:
            raise Infeasible(f"area angle {th:+d}: the gripped edge comes within {checks['gripped_edge_clearance_mm']} mm of the "
                             f"wheel line in the wheel's lane (< {EDGE_CLEARANCE_MM:g} mm); move the area up")
    cx_p, cy_p = sheet.to_planner([wa["cx"], wa["cy"]])
    tp = {
        "id": f"TP{idx}",
        "passes": int(wa["n"]),
        "start_direction": "l2r",       # direction of the block's first pass; set by the loop (alternates over the session)
        "robot": {  # zigzag_planner.plan_zigzag() equivalent (lv doubled: their lv = one up+down V)
            "zigzag_lv_mm": round(2.0 * lv, 4), "zigzag_alpha_deg": round(alpha, 4),
            "zigzag_N": int(wa["N"]), "rotation_deg": rd,
            "center_x_mm": round(float(cx_p), 4),          # not a plan_zigzag() parameter: applied as a shift
            "start_mm": round(float(cy_p) - lv / 2.0, 4),  # bbox bottom (pre-rotation) in the planner frame
        },
        "ours": {"lv_mm": round(lv, 4), "alpha_deg": round(alpha, 4), "N": int(wa["N"]), "rot": rd,
                 "cx": float(wa["cx"]), "cy": float(wa["cy"]), "wx": float(wa["wx"]), "wy": float(wa["wy"]),
                 "strip_width_mm": STRIP_W},
        "pivots_xy_mm": [[round(float(x), 4), round(float(y), 4)] for x, y in piv_planner],
    }
    if th:                                   # ours.rot = rotation_deg as before; the split only for tilted areas
        tp["ours"].update({"zz_rot": rot, "angle": th})
        tp["checks"] = checks
    return tp


def flip(d):
    return "r2l" if d == "l2r" else "l2r"


def pass_directions(tp):
    """direction of every pass of a toolpath: the first pass as tp['start_direction'], then alternating
    (the robot retraces the pattern backwards). l2r = increasing planner X = the robot's 'left -> right'."""
    d = tp.get("start_direction", "l2r"); out = []
    for _ in range(int(tp["passes"])):
        out.append(d); d = flip(d)
    return out


def set_directions(tp, start):
    tp["start_direction"] = start
    tp["pass_directions"] = pass_directions(tp)
    return tp


def pass_order(tp, d):
    """the pivots in traversal order for a pass of direction d (pivots_xy_mm is stored in the robot planner's
    own order, decreasing X = r2l)"""
    piv = list(tp["pivots_xy_mm"])
    return piv[::-1] if d == "l2r" else piv


def plan_to_toolpaths(was, sheet, allow_rot90=False, max_angle=0.0):
    """model plan (list of wa dicts) -> (toolpaths, problems). Entries are converted in order;
    the first infeasible entry STOPS the conversion (later entries depend on it having been
    executed), and is reported in problems as (index, reason)."""
    toolpaths, problems = [], []
    for i, wa in enumerate(was):
        try:
            toolpaths.append(wa_to_toolpath(wa, sheet, idx=i, allow_rot90=allow_rot90, max_angle=max_angle))
        except Infeasible as e:
            problems.append((i, str(e)))
            break
    return toolpaths, problems


def toolpath_doc(sheet, toolpaths, note=""):
    return {
        "schema_version": SCHEMA_VERSION,
        "units": {"length": "mm", "angle": "deg"},
        "frame": "planner: origin = middle of the wheel contact line, X across the sheet, "
                 "Y along the feed away from the clamp (zigzag_planner.plan_zigzag frame)",
        "sheet": sheet.to_dict(),
        "note": note,
        "toolpaths": toolpaths,
    }


def write_toolpath_json(path, doc):
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def read_toolpath_json(path):
    doc = json.load(open(path))
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {doc.get('schema_version')} != {SCHEMA_VERSION}")
    return doc


def preview_png(doc, path, title=None):
    """Sheet-frame preview: sheet outline, clamped band, wheel line, safe zone, pads, every toolpath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, Polygon
    sheet = Sheet.from_dict(doc["sheet"])
    W, L = sheet.width_mm, sheet.length_mm
    fig, ax = plt.subplots(figsize=(7.5, 7.5 * (L + 30) / (W + 30)))
    ax.add_patch(Rectangle((-W / 2, -L / 2), W, L, fc="#eceff1", ec="#263238", lw=2, zorder=1))
    ax.add_patch(Rectangle((-W / 2, -L / 2), W, sheet.clamp_mm, fc="#cfd8dc", ec="none", zorder=2,
                           hatch="//", alpha=0.6))
    ax.axhline(sheet.wheel_line_y, color="#c62828", ls="--", lw=1.2, zorder=3)
    ax.text(W / 2 + 2, sheet.wheel_line_y, "wheel line", color="#c62828", fontsize=7, va="center")
    x0, x1, y0, y1 = sheet.safe_zone_sheet()
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="#2e7d32", ls="--", lw=1.2, zorder=4))
    for (px, py) in sheet.pad_centres():
        ax.add_patch(Circle((px, py), sheet.pad_diameter_mm / 2, fc="#f5cba7", ec="#935116", zorder=5))
    colors = plt.get_cmap("tab10")
    arrow = "L\u2192R"; arrow_r = "R\u2192L"
    for k, tp in enumerate(doc["toolpaths"]):
        o = tp["ours"]
        c = colors(k % 10)
        dirs = tp.get("pass_directions") or pass_directions(tp)
        first = sheet.from_planner(np.array(pass_order(tp, dirs[0]), dtype=float))     # traversal order of pass 1
        last = sheet.from_planner(np.array(pass_order(tp, dirs[-1]), dtype=float))
        if o.get("angle"):
            ax.add_patch(Polygon(wa_corners(o), closed=True, fc="none", ec=c, ls=":", lw=1, zorder=6))
        else:
            ax.add_patch(Rectangle((o["cx"] - o["wx"] / 2, o["cy"] - o["wy"] / 2), o["wx"], o["wy"],
                                   fc="none", ec=c, ls=":", lw=1, zorder=6))
        ax.plot(first[:, 0], first[:, 1], "-", color=c, lw=1.4, zorder=7,
                label=f"{tp['id']}: {tp['passes']} pass(es) " + ", ".join(f"{i + 1}:{arrow if d == 'l2r' else arrow_r}" for i, d in enumerate(dirs))
                      + f"; N={o['N']}, " + (f"rot={o['zz_rot']}, angle={o['angle']:+d}" if o.get("angle") else f"rot={o['rot']}"))
        for a, b in zip(first[:-1], first[1:]):                                        # arrow heads along pass 1
            mid = (a + b) / 2; v = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            ax.annotate("", xy=mid + 2.0 * v, xytext=mid - 2.0 * v, arrowprops=dict(arrowstyle="->", color=c, lw=1.0), zorder=8)
        ax.plot(first[0, 0], first[0, 1], "o", color="#2e7d32", ms=9, mec="k", zorder=9)
        ax.plot(last[-1, 0], last[-1, 1], "s", color="#c62828", ms=6, mec="k", zorder=9)
        events = {}                                                                      # every pass's start and end, by location
        for i, d in enumerate(dirs):
            o_i = sheet.from_planner(np.array(pass_order(tp, d), dtype=float))
            for pt, what in ((o_i[0], f"P{i + 1} start"), (o_i[-1], f"P{i + 1} end")):
                events.setdefault((round(float(pt[0]), 1), round(float(pt[1]), 1)), []).append(what)
        for (ex, ey), what in events.items():
            right = ex >= o["cx"]
            ax.annotate("\n".join(what), (ex, ey), textcoords="offset points", xytext=(6 if right else -6, -4), ha="left" if right else "right",
                        va="top", fontsize=7, color="k", weight="bold", zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, lw=0.8, alpha=0.9))
    ax.set_xlim(-W / 2 - 15, W / 2 + 15); ax.set_ylim(-L / 2 - 15, L / 2 + 15)
    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm, feed direction)")
    ax.set_title(title or f"sheet {W:g}x{L:g}x{sheet.thickness_mm:g} mm, clamp {sheet.clamp_mm:g} mm\n"
                 "green dot = start of pass 1, red square = end of the last pass, arrows = pass 1", fontsize=9)
    if doc["toolpaths"]:
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
