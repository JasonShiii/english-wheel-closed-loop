"""danish_run2_metrics.py with the sheet LENGTH as a parameter (2026-10-29): the runs of 10-21/22 were on a 231 x 124 sheet,
not 231 x 174, so the observed region, the zigzag zone and the y-bands of the secondary metrics must follow the length.
Every band is expressed relative to the gripped edge (y = -L/2) or the free edge (y = +L/2) exactly as the 174 mm
constants were: OBS_Y = gripped edge + 20 (the strip hidden by the gripper), zone legs 12..112 above the gripped edge,
far band = free edge - 52 .. -2, near band = OBS_Y + 5 .. +37, lift bands OBS_Y + 5 .. +22 and free edge - 17.
make(174.0) reproduces danish_run2_metrics.metrics exactly (plus sag_pmax_200 / ridge_axis_deg: principal-curvature sagitta
over 200 mm and the cylinder-axis angle from the quadratic fit over |y| < 30, |x| < 100 -- invariant to the frame rotation)."""
import numpy as np


def make(L):
    yg, yf = -L / 2.0, L / 2.0
    OBS_Y = yg + 20.0
    ZONE = dict(x=(-55.0, 55.0), y=(yg + 12.0, yg + 112.0))

    def quad_fit(X, Y, Z, m):
        A = np.column_stack([X[m] ** 2, X[m] * Y[m], Y[m] ** 2, X[m], Y[m], np.ones(m.sum())])
        return np.linalg.lstsq(A, Z[m], rcond=None)[0]

    def plane_remove(X, Y, Z, m):
        A = np.column_stack([X[m], Y[m], np.ones(m.sum())]); cf = np.linalg.lstsq(A, Z[m], rcond=None)[0]
        return Z - (cf[0] * X + cf[1] * Y + cf[2])

    def metrics(xs, ys, Z):
        X, Y = np.meshgrid(xs, ys); fin = np.isfinite(Z); obs = fin & (Y > OBS_Y)
        out = dict(cells=int(obs.sum()))
        if obs.sum() < 200:
            return out, np.full_like(Z, np.nan)
        m = obs & (np.abs(Y) < 10) & (np.abs(X) < 100)
        if m.sum() > 50:
            cf = np.polyfit(X[m], Z[m], 2); out["sag_x_200"] = round(float(abs(2 * cf[0]) * 200 ** 2 / 8), 2)
        m = obs & (X > ZONE["x"][0]) & (X < ZONE["x"][1]) & (Y > ZONE["y"][0]) & (Y < ZONE["y"][1])
        if m.sum() > 100:
            cf = quad_fit(X, Y, Z, m); out["zone_sag_x_105"] = round(float(abs(2 * cf[0]) * 105 ** 2 / 8), 2); out["zone_sag_y_100"] = round(float(abs(2 * cf[2]) * 100 ** 2 / 8), 2)
            out["zone_kxx"] = float(2 * cf[0]); out["zone_kyy"] = float(2 * cf[2])
        # 1b rotation-invariant: principal curvatures of a quadratic fit over the central band |y| < 30, |x| < 100 of the
        # observed region -> sagitta of the larger principal curvature over 200 mm and the cylinder-axis angle to y
        m = obs & (np.abs(Y) < 30) & (np.abs(X) < 100)
        if m.sum() > 200:
            cf = quad_fit(X, Y, Z, m); Hm = np.array([[2 * cf[0], cf[1]], [cf[1], 2 * cf[2]]]); ev, evec = np.linalg.eigh(Hm)
            i = int(np.argmax(np.abs(ev))); out["sag_pmax_200"] = round(float(abs(ev[i]) * 200 ** 2 / 8), 2)
            ax = evec[:, 1 - i]; out["ridge_axis_deg"] = round(float(np.degrees(np.arctan2(ax[0], ax[1]))), 1) if ax[1] != 0 else 90.0
            if out["ridge_axis_deg"] > 90: out["ridge_axis_deg"] -= 180
            if out["ridge_axis_deg"] < -90: out["ridge_axis_deg"] += 180
        Zp = plane_remove(X, Y, Z, obs); v = Zp[obs]
        out["amp_plane_removed"] = round(float(np.percentile(v, 99.5) - np.percentile(v, 0.5)), 2)
        out["rms_plane_removed"] = round(float(np.sqrt(np.mean(v ** 2))), 2)

        def slope(y0, y1):
            mm = obs & (np.abs(X) < 15) & (Y > y0) & (Y < y1)
            return float(np.polyfit(Y[mm], Z[mm], 1)[0]) if mm.sum() > 30 else np.nan
        out["bend_angle_y_deg"] = round(float(np.degrees(np.arctan(slope(yf - 52, yf - 2)) - np.arctan(slope(OBS_Y + 5, OBS_Y + 37)))), 2)
        near = obs & (np.abs(X) < 30) & (Y > OBS_Y + 5) & (Y < OBS_Y + 22); far = obs & (np.abs(X) < 30) & (Y > yf - 17)
        if near.sum() > 20 and far.sum() > 20:
            out["lift_far_minus_near"] = round(float(np.median(Z[far]) - np.median(Z[near])), 2)
        return out, Zp

    return dict(metrics=metrics, plane_remove=plane_remove, quad_fit=quad_fit, ZONE=ZONE, OBS_Y=OBS_Y)
