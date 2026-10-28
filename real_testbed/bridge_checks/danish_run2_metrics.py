"""Rigid-motion-invariant shape metrics of a height field on the 2 mm sheet grid, used on the MEASURED fields
(danish_run2_measure.py) and on the SIMULATED clamped states (newsheet_pipeline/dose_fit/compare_231x174.py) alike.
The sheet is held at its bottom edge (y = -87) and one gripper is compliant, so a rigid tilt between scans is possible;
curvatures and plane-removed amplitudes are compared, not raw heights."""
import numpy as np

ZONE = dict(x=(-55.0, 55.0), y=(-67.0 + 12.0, -67.0 + 112.0))   # zigzag zone: legs 0..100 above the wheel line (12 mm above the gripped edge)
OBS_Y = -67.0                                                    # observed region: above the strip hidden by the gripper


def quad_fit(X, Y, Z, m):
    """z ~ a x^2 + b xy + c y^2 + d x + e y + f on the cells m -> coefficients"""
    A = np.column_stack([X[m] ** 2, X[m] * Y[m], Y[m] ** 2, X[m], Y[m], np.ones(m.sum())])
    return np.linalg.lstsq(A, Z[m], rcond=None)[0]


def plane_remove(X, Y, Z, m):
    A = np.column_stack([X[m], Y[m], np.ones(m.sum())]); cf = np.linalg.lstsq(A, Z[m], rcond=None)[0]
    return Z - (cf[0] * X + cf[1] * Y + cf[2])


def metrics(xs, ys, Z):
    X, Y = np.meshgrid(xs, ys); fin = np.isfinite(Z); obs = fin & (Y > OBS_Y)
    out = dict(cells=int(obs.sum()))
    if obs.sum() < 200:
        return out
    # 1 curvature across x of the whole width, band |y| < 10 around the zigzag centre line -> sagitta over 200 mm
    m = obs & (np.abs(Y) < 10) & (np.abs(X) < 100)
    if m.sum() > 50:
        cf = np.polyfit(X[m], Z[m], 2); out["sag_x_200"] = round(float(abs(2 * cf[0]) * 200 ** 2 / 8), 2)
    # 2 quadratic surface in the zigzag zone -> sagitta across the zigzag (105 mm) and along the legs (100 mm)
    m = obs & (X > ZONE["x"][0]) & (X < ZONE["x"][1]) & (Y > ZONE["y"][0]) & (Y < ZONE["y"][1])
    if m.sum() > 100:
        cf = quad_fit(X, Y, Z, m); out["zone_sag_x_105"] = round(float(abs(2 * cf[0]) * 105 ** 2 / 8), 2); out["zone_sag_y_100"] = round(float(abs(2 * cf[2]) * 100 ** 2 / 8), 2)
        out["zone_kxx"] = float(2 * cf[0]); out["zone_kyy"] = float(2 * cf[2])
    # 3 amplitude after removing the best-fit plane of the observed region
    Zp = plane_remove(X, Y, Z, obs); v = Zp[obs]
    out["amp_plane_removed"] = round(float(np.percentile(v, 99.5) - np.percentile(v, 0.5)), 2)
    out["rms_plane_removed"] = round(float(np.sqrt(np.mean(v ** 2))), 2)
    # 4 bending angle along y across the zone: slope of h(y) at x ~ 0 in the far band minus the near band (deg)
    def slope(y0, y1):
        mm = obs & (np.abs(X) < 15) & (Y > y0) & (Y < y1)
        return float(np.polyfit(Y[mm], Z[mm], 1)[0]) if mm.sum() > 30 else np.nan
    out["bend_angle_y_deg"] = round(float(np.degrees(np.arctan(slope(35, 85)) - np.arctan(slope(-62, -30)))), 2)
    # 5 free-edge lift relative to the lowest visible strip (both along x ~ 0)
    near = obs & (np.abs(X) < 30) & (Y > -62) & (Y < -45); far = obs & (np.abs(X) < 30) & (Y > 70)
    if near.sum() > 20 and far.sum() > 20:
        out["lift_far_minus_near"] = round(float(np.median(Z[far]) - np.median(Z[near])), 2)
    return out, Zp
