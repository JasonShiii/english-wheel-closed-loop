"""Which zigzag orientations can the Dobot cell execute? (2026-10-29, no motion: kinematics queries only)

For every angle theta and every working-area centre: a 50 x 50 mm area, N = 12, our zigzag geometry at rot 0,
rotated by theta (CCW, sheet frame: x across, y along the feed away from the gripper) about the area centre, then
    1. his per-pivot safe-zone check (zigzag_planner.point_in_safe_zone, our sheet and zone),
    2. his motion plan (zigzag_forming_oldclamps._build_motion_plan + the leveling shift), both pass directions,
    3. the controller's inverse kinematics for every pose (dashboard InverseKin, current user/tool frame: the
       solution the controller itself would use from the current configuration), joint values and the largest
       joint change between consecutive poses (a branch flip shows as a large jump),
    4. where the gripper sits relative to the wheel: for every pose the two pads and the gripped edge in the
       wheel frame (the sheet turned by the leg heading about the contact point).
Writes rotation_range.json. Run on the robot PC with the controller reachable and in TCP mode (no motion sent):
    python robot_dobot/rotation_range.py --scripts <Scripts> --out <dir>
"""
import argparse, json, math, os, socket, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RT = os.path.dirname(HERE)
sys.path.insert(0, RT)
from ew_loop import spec                      # noqa: E402
from ew_loop.frames import Sheet              # noqa: E402

THETAS = [0, 10, -10, 15, -15, 20, -20, 30, -30, 45, -45, 60, -60]
XS, YS = [-60, -30, 0, 30, 60], [-40, 0, 40, 70]
SHEET = Sheet(width_mm=231.0, length_mm=174.0, thickness_mm=1.0, clamp_mm=12.0, margin_side_mm=43.4, margin_bottom_mm=0.0,
              margin_top_mm=12.0, pad_offsets_from_left_mm=(12.5, 219.5), pad_diameter_mm=13.0)


class IK:
    def __init__(self, host="192.168.137.100", port=29999):
        self.s = socket.create_connection((host, port), timeout=5); self.buf = b""

    def q(self, pose_str):
        self.s.sendall(f"InverseKin({pose_str})\n".encode())
        while b";" not in self.buf:
            self.buf += self.s.recv(4096)
        line, self.buf = self.buf.split(b";", 1)
        r = line.decode(errors="replace").strip()
        code = int(r.split(",", 1)[0])
        if code != 0:
            return code, None
        return 0, [float(v) for v in r.split("{")[1].split("}")[0].split(",")]


def pivots_sheet(cx, cy, theta, wx=50.0, wy=50.0, N=12):
    p = spec.wa_pivots_sheet(dict(cx=0.0, cy=0.0, wx=wx, wy=wy, rot=0, N=N, n=1))
    t = math.radians(theta); R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return p @ R.T + np.array([cx, cy])


def wheel_frame(points_planner, contact, heading):
    """planner-frame points -> wheel frame (x lateral, y along the feed, + = past the wheel line toward the free edge... of
    the sheet seen from the wheel: the sheet is turned so that the leg runs along the feed)"""
    c, s = math.cos(heading), math.sin(heading)
    d = np.asarray(points_planner, float) - np.asarray(contact, float)
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], axis=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scripts", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sys.modules.setdefault("dobot_sdk", type(sys)("dobot_sdk")); sys.modules["dobot_sdk"].DobotRobot = object; sys.modules["dobot_sdk"].CoordinateType = object
    sys.path.insert(0, a.scripts)
    import zigzag_forming_oldclamps as zf
    zf.X_DIM_MM, zf.Y_DIM_MM, zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM = 231.0, 174.0, 12.0
    zf.MARGIN_SIDE_MM, zf.MARGIN_BOTTOM_MM, zf.MARGIN_TOP_MM = 43.4, 0.0, 12.0
    zf.ARC_RADIUS_X_MM = zf.ARC_RADIUS_Z_MM = zf.ARC_RADIUS_MM = None; zf.LEG_REPEAT = 1
    args = (zf.X_DIM_MM, zf.Y_DIM_MM, zf.MARGIN_SIDE_MM, zf.MARGIN_BOTTOM_MM, zf.MARGIN_TOP_MM, zf.SHEET_BOTTOM_TO_WHEEL_LINE_MM)
    ik = IK()
    c0, q_pos9 = ik.q(zf._dobot_pose_str(zf.TARGET_POS_9)); c1, q_scan = ik.q(zf._dobot_pose_str(zf.pointcloud_position))
    # gripper pads and the gripped edge in the planner frame (sheet frame y = -87 edge; pads 18 mm deep)
    W, L = SHEET.width_mm, SHEET.length_mm
    pads_sheet = np.array([[-W / 2 + o, -L / 2 + 9.0] for o in SHEET.pad_offsets_from_left_mm])
    edge_sheet = np.stack([np.linspace(-W / 2, W / 2, 47), np.full(47, -L / 2)], axis=1)
    pads_p, edge_p = SHEET.to_planner(pads_sheet), SHEET.to_planner(edge_sheet)
    rows, t0 = [], time.time()
    for th in THETAS:
        for cy in YS:
            for cx in XS:
                piv_s = pivots_sheet(cx, cy, th); piv_p = SHEET.to_planner(piv_s)
                oob = [list(map(float, p)) for p in piv_p if not zf.point_in_safe_zone(tuple(p), *args)]
                row = {"theta": th, "cx": cx, "cy": cy, "safe_zone_ok": not oob, "n_out_of_zone": len(oob),
                       "extent_sheet": [round(float(piv_s[:, 0].min()), 1), round(float(piv_s[:, 0].max()), 1),
                                        round(float(piv_s[:, 1].min()), 1), round(float(piv_s[:, 1].max()), 1)]}
                poses, heads, contacts = [], [], []
                for order in (list(map(tuple, piv_p))[::-1], list(map(tuple, piv_p))):
                    plan, thetas, _ = zf._build_motion_plan(order)
                    dz = zf.TARGET_POS_9[2] - plan[0][2][0][2]
                    for kind, idx, ps in zf._shift_plan_z(plan, dz):
                        for k, p in enumerate(ps):
                            poses.append(p)
                    for i, h in enumerate(thetas):
                        heads.append(h); contacts.append(order[i]); heads.append(h); contacts.append(order[i + 1])
                row["leg_headings_deg"] = [round(math.degrees(min(heads)), 1), round(math.degrees(max(heads)), 1)]
                # IK along the sequence (pos_9 -> pass 1 -> pass 2)
                prev, jumps, codes, J = q_pos9, np.zeros(6), [], []
                for p in poses:
                    code, q = ik.q(zf._dobot_pose_str(p)); codes.append(code)
                    if q is None:
                        continue
                    J.append(q); jumps = np.maximum(jumps, np.abs(np.array(q) - np.array(prev))); prev = q
                J = np.array(J) if J else np.zeros((0, 6))
                row.update({"ik_ok": all(c == 0 for c in codes), "ik_fail": int(sum(c != 0 for c in codes)), "n_poses": len(poses),
                            "ik_codes": sorted(set(codes)),
                            "joint_min": [round(float(v), 1) for v in J.min(0)] if len(J) else None,
                            "joint_max": [round(float(v), 1) for v in J.max(0)] if len(J) else None,
                            "max_joint_step_deg": [round(float(v), 1) for v in jumps]})
                # gripper relative to the wheel, over every leg end
                pad_fwd, pad_lat, edge_hit = -1e9, 1e9, 1e9
                for h, cpt in zip(heads, contacts):
                    pw = wheel_frame(pads_p, cpt, h); ew = wheel_frame(edge_p, cpt, h)
                    pad_fwd = max(pad_fwd, float(pw[:, 1].max()))
                    near = pw[pw[:, 1] > -10.0]
                    if len(near):
                        pad_lat = min(pad_lat, float(np.abs(near[:, 0]).min()))
                    close = ew[np.abs(ew[:, 0]) < 40.0]                   # the gripped edge within 40 mm of the wheel's lateral position
                    if len(close):
                        edge_hit = min(edge_hit, float(-close[:, 1].max()))  # distance of that edge part before the wheel line
                row.update({"pad_max_past_wheel_line_mm": round(pad_fwd, 1),
                            "pad_min_lateral_when_at_wheel_line_mm": None if pad_lat == 1e9 else round(pad_lat, 1),
                            "gripped_edge_clearance_in_wheel_lane_mm": None if edge_hit == 1e9 else round(edge_hit, 1)})
                rows.append(row)
                print(f"theta {th:+4d} ({cx:+4d},{cy:+4d}): zone {'ok ' if not oob else 'OUT'} IK {'ok ' if row['ik_ok'] else 'FAIL'} "
                      f"step {max(row['max_joint_step_deg']):6.1f} deg  pad past line {row['pad_max_past_wheel_line_mm']:7.1f}  "
                      f"edge clearance {row['gripped_edge_clearance_in_wheel_lane_mm']}", flush=True)
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "secs": round(time.time() - t0), "area": "50 x 50 mm, N 12 (our geometry, rot 0, then rotated)",
           "sheet": SHEET.to_dict(), "safe_zone_planner": zf.safe_zone_bounds(*args), "q_pos9": q_pos9, "q_scan": q_scan, "rows": rows}
    json.dump(res, open(os.path.join(a.out, "rotation_range.json"), "w"), indent=1)
    print("written", os.path.join(a.out, "rotation_range.json"))


if __name__ == "__main__":
    main()
