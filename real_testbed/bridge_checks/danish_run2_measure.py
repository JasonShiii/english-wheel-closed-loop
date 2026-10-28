"""Measure Danish's runs of 2026-10-21 (14-leg zigzag, pivot 5 deg) and 2026-10-22 (30-leg zigzag, pivot 2 deg) on the
231 x 174 x 0.6 sheet: new arm, force control (300 N setpoint, ~267 N achieved), one gripper compliant, flat scan (pass0)
+ 6 passes, every scan at the withdrawn pose. Chain: merge (09-28 rig calibration, device-1 frame) -> gate -> OUR sheet
finder (RANSAC planes on the flat pass0, the plane that best contains a 231 x 174 rectangle; their segmentation is not
used because it grabs other planes in the new scene) -> sheet frame from pass0 (x along 231, +y toward the free edge)
-> per-camera plane correction from the flat scan (camera 2 sits 5-8 mm off the others with the 09-28 calibration)
-> every pass: points in the band around the pass0 plane inside the pass0 rectangle -> 2 mm height field h = signed
distance to the FLAT sheet plane -> hardware mask (cells that are not flat in pass0) -> rigid-invariant metrics
(danish_run2_metrics.py). Outputs: figures/danish_run2/<run>_passes.png, _fields.npz, _stats.json.
prepare(run) / field_points() are reused by danish_run2_visualize.py."""
import glob, json, os, sys
import numpy as np
try:
    import cv2
except ImportError:                      # 2026-10-27: no cv2 in the EW env (plan_from_capture.py); exact numpy fallback below
    cv2 = None
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, RT); sys.path.insert(0, os.path.join(RT, "bridge_checks"))
from ew_loop import scan_to_field as S
from plate_refill import fit_known_rectangle
from danish_run2_metrics import metrics, ZONE, OBS_Y
CALIB = os.path.join(RT, "pc_post_processing", "deliverable_092826", "calibration results")
OUT = os.path.join(RT, "figures", "danish_run2"); os.makedirs(OUT, exist_ok=True)
W, L, STEP = 231.0, 174.0, 2.0
GATE = np.array([(-300.0, 400.0), (-300.0, 400.0), (300.0, 700.0)])
BAND = 45.0                      # |distance to the flat plane| accepted for a deformed pass (mm)
FLAT_TOL = 3.0                   # a cell is sheet if the flat scan is within this of the plane (else hardware / outlier)
XS = np.arange(-W / 2, W / 2 + STEP, STEP); YS = np.arange(-L / 2, L / 2 + STEP, STEP)


def load(run, k):
    xyz, ids = S.merge_capture(os.path.join(run, f"pointcloud_pass{k}"), CALIB, window=(1.0, 1e6), cameras=(1, 2, 3), with_ids=True)
    xyz = xyz.astype(np.float32); g = np.all((xyz > GATE[:, 0]) & (xyz < GATE[:, 1]), axis=1)
    return xyz[g], ids[g]


def ransac_planes(P, n_iter=600, tol=1.5, seed=0):
    rng = np.random.default_rng(seed); sub = P[rng.choice(len(P), min(60000, len(P)), replace=False)]
    found = []
    for _ in range(n_iter):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]; n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n /= nn; d = (sub - a) @ n; found.append((int((np.abs(d) < tol).sum()), a, n))
    found.sort(key=lambda t: -t[0]); planes = []
    for cnt, a, n in found:
        if all(abs(n @ m) < 0.996 or abs((a - q) @ m) > 15 for _, q, m in planes):
            planes.append((cnt, a, n))
        if len(planes) == 10:
            break
    return planes


def find_sheet(P):
    """the plane whose inliers best fill a 231 x 174 rectangle and little else"""
    best = None
    for cnt, a, n in ransac_planes(P):
        d = (P - a) @ n; m = np.abs(d) < 2.5; Q = P[m]
        if len(Q) < 15000 or len(Q) > 400000:
            continue
        inside, rec = fit_known_rectangle(Q, W, L); frac = inside.mean(); score = rec["score_cells"] * frac
        if best is None or score > best[0]:
            best = (score, Q[inside], n, rec, len(Q), frac)
    return best


def _min_area_rect(q):
    """numpy twin of cv2.minAreaRect: ((cx, cy), (w, h), angle_deg) with w along the returned angle. The minimum-area
    rectangle has one side collinear with a convex-hull edge, so every hull edge direction is tried."""
    from scipy.spatial import ConvexHull
    H = q[ConvexHull(q).vertices]; E = np.roll(H, -1, axis=0) - H; best = None
    for ex, ey in E / np.linalg.norm(E, axis=1)[:, None]:
        u = H @ np.array([ex, ey]); v = H @ np.array([-ey, ex]); area = np.ptp(u) * np.ptp(v)
        if best is None or area < best[0]:
            best = (area, ex, ey, u, v)
    _, ex, ey, u, v = best; cu, cv = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
    return ((cu * ex - cv * ey, cu * ey + cv * ex), (float(np.ptp(u)), float(np.ptp(v))), float(np.degrees(np.arctan2(ey, ex))))


def frame_from_flat(Q):
    c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c, full_matrices=False); nrm = Vt[2]
    if (np.zeros(3) - c) @ nrm < 0:
        nrm = -nrm
    e1 = Vt[0]; e2 = np.cross(nrm, e1); q = np.column_stack([(Q - c) @ e1, (Q - c) @ e2])
    (rx, ry), (rw, rh), ang = cv2.minAreaRect(q.astype(np.float32)) if cv2 else _min_area_rect(q); a = np.radians(ang)
    if rw < rh:
        a += np.pi / 2
    ex = np.cos(a) * e1 + np.sin(a) * e2; ey = np.cross(nrm, ex); c = c + rx * e1 + ry * e2
    return dict(c=c, ex=ex, ey=ey, n=nrm)


def to_sheet(P, F):
    d = P - F["c"]; return np.column_stack([d @ F["ex"], d @ F["ey"], d @ F["n"]])


def despeckle(Z, size=5, tol=6.0):
    """drop cells that differ from the local median (size x size, NaN-aware) by more than tol mm: reflections, hardware
    remnants and merge seams; the same rule will apply to real captures in deployment"""
    from scipy.ndimage import generic_filter
    med = generic_filter(Z, np.nanmedian, size=size, mode="constant", cval=np.nan)
    bad = np.isfinite(Z) & np.isfinite(med) & (np.abs(Z - med) > tol)
    out = Z.copy(); out[bad] = np.nan; return out, int(bad.sum())


def grid(xy, h, min_pts=3, fallback_cells=3):
    """2 mm cells: median of >= min_pts points per cell. Cells left empty (sparse depth returns on the specular sheet, e.g.
    the far-left corner with ~0.2 points/mm^2) get the median of the points within +-fallback_cells cells (a 14 mm box)
    if >= min_pts points lie there; nothing is interpolated across larger gaps."""
    Z = np.full((len(YS), len(XS)), np.nan)
    ix = np.round((xy[:, 0] - XS[0]) / STEP).astype(int); iy = np.round((xy[:, 1] - YS[0]) / STEP).astype(int)
    ok = (ix >= 0) & (ix < len(XS)) & (iy >= 0) & (iy < len(YS)); ix, iy, hv = ix[ok], iy[ok], h[ok]
    lin = iy * len(XS) + ix; o = np.argsort(lin); ls = lin[o]; hs = hv[o]; b = np.flatnonzero(np.diff(ls)) + 1
    cnt = np.zeros(Z.shape, int)
    for gid, gg in zip(ls[np.r_[0, b]], np.split(hs, b)):
        cnt.flat[gid] = len(gg)
        if len(gg) >= min_pts:
            Z.flat[gid] = np.median(gg)
    if fallback_cells:
        from scipy.ndimage import uniform_filter
        sup = uniform_filter(cnt.astype(float), size=2 * fallback_cells + 1, mode="constant") * (2 * fallback_cells + 1) ** 2
        need = np.isnan(Z) & (cnt > 0) & (sup >= min_pts)
        if need.any():
            order = np.lexsort((ix, iy)); ix, iy, hv = ix[order], iy[order], hv[order]
            for r, c in zip(*np.where(need)):
                m = (np.abs(iy - r) <= fallback_cells) & (np.abs(ix - c) <= fallback_cells)
                Z[r, c] = np.median(hv[m])
    return Z


def prepare(run):
    """flat scan -> sheet frame (gripped edge at y = -L/2) and the per-camera plane correction; returns (P0, I0, F, corr, info)"""
    P0, I0 = load(run, 0); score, Q, n, rec, nin, frac = find_sheet(P0)
    F = frame_from_flat(Q); s0 = to_sheet(P0, F); beyond = {}
    for side, sgn in (("y-", -1), ("y+", 1)):           # the gripper bar lies beyond the gripped edge
        m = (np.abs(s0[:, 2]) < 25) & (np.abs(s0[:, 0]) < W / 2) & (sgn * s0[:, 1] > L / 2 + 5) & (sgn * s0[:, 1] < L / 2 + 60)
        beyond[side] = int(m.sum())
    if beyond["y+"] > beyond["y-"]:
        F["ex"], F["ey"] = -F["ex"], -F["ey"]; s0 = to_sheet(P0, F)
    # flat-sheet points for the per-camera correction: a generous height window (a camera may sit 5-10 mm off), the
    # clamp hardware at the gripped corners excluded by x and y
    on = (np.abs(s0[:, 2]) < 12) & (np.abs(s0[:, 0]) < W / 2 - 15) & (s0[:, 1] > OBS_Y + 12) & (s0[:, 1] < L / 2 - 5)
    corr, raw = {}, {}
    for cam in (1, 2, 3):
        mc = on & (I0 == cam)
        if mc.sum() > 300:                                   # enough flat points: offset + tilt
            A = np.column_stack([s0[mc, 0], s0[mc, 1], np.ones(mc.sum())]); corr[cam] = np.linalg.lstsq(A, s0[mc, 2], rcond=None)[0]
        elif mc.sum() > 50:                                  # few flat points (camera 2 on 10-21): offset only
            corr[cam] = np.array([0.0, 0.0, float(np.median(s0[mc, 2]))])
        if mc.sum() > 50:
            raw[cam] = round(float(np.median(s0[mc, 2])), 2)
    info = dict(sheet_plane_inliers=int(nin), rect_score_cells=int(rec["score_cells"]), inside_fraction=round(float(frac), 3), beyond_edge_points=beyond,
                camera_flat_median_before_mm=raw, camera_flat_offset_mm={c: round(float(v[2]), 2) for c, v in corr.items()},
                camera_flat_tilt_deg={c: [round(float(np.degrees(v[0])), 3), round(float(np.degrees(v[1])), 3)] for c, v in corr.items()})
    return P0, I0, F, corr, info


def field_points(P, I, F, corr):
    """sheet-frame xy, corrected heights and the band mask of one capture"""
    s = to_sheet(P, F); h = s[:, 2].copy()
    for cam, v in corr.items():
        mc = I == cam; h[mc] -= v[0] * s[mc, 0] + v[1] * s[mc, 1] + v[2]
    m = (np.abs(h) < BAND) & (np.abs(s[:, 0]) < W / 2 + 6) & (np.abs(s[:, 1]) < L / 2 + 6)
    return s[:, :2], h, m


def measure(run):
    name = os.path.basename(run); print("===", name, flush=True)
    P0, I0, F, corr, info = prepare(run); print(info, flush=True)
    xy0, h0, m0 = field_points(P0, I0, F, corr); Z0 = grid(xy0[m0], h0[m0]); sheet = np.isfinite(Z0) & (np.abs(Z0) < FLAT_TOL)
    GX, GY = np.meshgrid(XS, YS); sheet &= ~((np.abs(GX) > W / 2 - 14) & (GY < -15))     # clamp hardware (pads + handles) along the gripped corners
    fields, stats, planed = {}, {}, {}
    for k in range(7):
        if k == 0:
            Z, nb = Z0, int(m0.sum())
        else:
            xy, h, m = field_points(*load(run, k), F, corr); Z, nb = grid(xy[m], h[m]), int(m.sum())
        Z, nbad = despeckle(np.where(sheet, Z, np.nan)); fields[k] = Z
        mt, Zp = metrics(XS, YS, Z); planed[k] = Zp
        v = Z[np.isfinite(Z)]
        stats[k] = dict(band_points=nb, despeckled_cells=nbad, coverage=round(float(np.isfinite(Z).mean()), 3), h_p0p5_p50_p99p5=[round(float(np.percentile(v, p)), 2) for p in (0.5, 50, 99.5)], **mt)
        print(k, {kk: vv for kk, vv in stats[k].items() if not kk.startswith("zone_k")}, flush=True)
    for k in range(6):
        d = fields[k + 1] - fields[k]; mm = np.isfinite(d); dv = d[mm]
        stats[f"pass{k+1}-pass{k}"] = dict(cells=int(mm.sum()), p5_p50_p95_mm=[round(float(np.percentile(dv, p)), 2) for p in (5, 50, 95)])
    np.savez_compressed(os.path.join(OUT, f"{name}_fields.npz"), xs=XS, ys=YS, **{f"pass{k}": fields[k] for k in range(7)})
    json.dump({"info": info, "stats": stats}, open(os.path.join(OUT, f"{name}_stats.json"), "w"), indent=1)
    vmax = max(np.nanpercentile(np.abs(fields[k]), 99.5) for k in range(1, 7)); ext = (XS[0], XS[-1], YS[0], YS[-1])
    fig, axs = plt.subplots(3, 7, figsize=(30, 12))
    for k in range(7):
        im = axs[0, k].imshow(fields[k], origin="lower", extent=ext, cmap="viridis", vmin=-vmax, vmax=vmax)
        axs[0, k].set_title(f"pass {k}: p0.5/50/99.5 {stats[k]['h_p0p5_p50_p99p5']} mm, cov {stats[k]['coverage']*100:.0f} %", fontsize=8)
        axs[0, k].add_patch(plt.Rectangle((ZONE["x"][0], ZONE["y"][0]), ZONE["x"][1] - ZONE["x"][0], ZONE["y"][1] - ZONE["y"][0], fill=False, ec="r", lw=0.8, ls="--"))
        lim = max(np.nanpercentile(np.abs(planed[k]), 99.5), 0.5)
        im2 = axs[1, k].imshow(planed[k], origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim); plt.colorbar(im2, ax=axs[1, k], fraction=0.045)
        axs[1, k].set_title(f"pass {k} minus its best-fit plane: amp {stats[k].get('amp_plane_removed')} mm, sag_x(200) {stats[k].get('sag_x_200')}", fontsize=8)
    plt.colorbar(im, ax=axs[0, 6], fraction=0.045, label="h above the flat sheet (mm)")
    iy = np.argmin(np.abs(YS)); ix = np.argmin(np.abs(XS))
    for k in range(7):
        col = plt.cm.viridis(k / 6)
        axs[2, 0].plot(XS, fields[k][iy, :], color=col, lw=1.2, label=f"pass {k}"); axs[2, 1].plot(YS, fields[k][:, ix], color=col, lw=1.2, label=f"pass {k}")
    axs[2, 0].set_title("profile across x at y = 0", fontsize=8); axs[2, 1].set_title("profile along y at x = 0 (gripped edge -87, free edge +87)", fontsize=8)
    for ax in axs[2, :2]:
        ax.legend(fontsize=7); ax.grid(color="#eee"); ax.set_ylabel("h (mm)")
    ks = list(range(1, 7))
    for key, mk in (("sag_x_200", "o-"), ("zone_sag_x_105", "^-"), ("zone_sag_y_100", "v-"), ("amp_plane_removed", "s-"), ("lift_far_minus_near", "d-")):
        axs[2, 2].plot(ks, [stats[k].get(key, np.nan) for k in ks], mk, label=key)
    axs[2, 2].plot(ks, [stats[k].get("bend_angle_y_deg", np.nan) for k in ks], "x--", label="bend angle along y (deg)")
    axs[2, 2].set_xlabel("pass"); axs[2, 2].legend(fontsize=7); axs[2, 2].grid(color="#eee"); axs[2, 2].set_title("rigid-invariant shape metrics per pass", fontsize=8)
    axs[2, 3].axis("off"); axs[2, 3].text(0, 0.5, json.dumps(info, indent=1), fontsize=7, va="center", family="monospace")
    for ax in axs[2, 4:]:
        ax.axis("off")
    fig.suptitle(f"{name}: 231 x 174 x 0.6 sheet, flat scan + 6 passes; h relative to the flat sheet plane, per-camera flat correction, hardware masked", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"{name}_passes.png"), dpi=70); plt.close(fig); print("->", os.path.join(OUT, f"{name}_passes.png"))


if __name__ == "__main__":
    runs = sorted(glob.glob(os.path.join(RT, "share", "Danish_Run_2", "2026-*")))
    for run in (runs if len(sys.argv) < 2 else [r for r in runs if sys.argv[1] in r]):
        measure(run)
