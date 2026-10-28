"""Refill step after the metrology segmentation (prototype, 2026-10-07).
Their cut (06_0_segment_plate) reliably finds the sheet but drops ~1/4 of its points: normals estimated from 18
neighbours (~4-5 mm) are swamped by ~1 mm depth noise (median 16-21 deg error vs their 11 deg link limit), so
single voxels cannot link. Loosening the normal test does not help: the wheel rim is tangent to the sheet, so a
smooth-surface region grows from the sheet onto the wheel and gripper. Fix that uses what we know instead:
keep their cut as the locator, build a LOCAL smooth surface from it (median height per 4 mm cell, gaps
interpolated, Gaussian-smoothed; follows any formed shape, unlike one global fit), and add every fused point
within band_mm of that surface AND inside the cut's outline rectangle (shrunk by shrink_mm). Two passes (the
surface is rebuilt from the union). Returns the refilled point set (same frame as the input)."""
import numpy as np, cv2

F = lambda u, v: np.column_stack([u**3, u*u*v, u*v*v, v**3, u*u, u*v, v*v, u, v, np.ones_like(u)])


def refill(fused, plate, band_mm=3.0, shrink_mm=3.0, passes=2):
    c = plate.mean(0); _, _, Vt = np.linalg.svd(plate - c, full_matrices=False)
    Q = (fused - c) @ Vt.T; q = (plate - c) @ Vt.T
    (rx, ry), (rw, rh), ang = cv2.minAreaRect(q[:, :2].astype(np.float32))
    a = np.radians(ang); x = (Q[:, 0] - rx) * np.cos(a) + (Q[:, 1] - ry) * np.sin(a); y = -(Q[:, 0] - rx) * np.sin(a) + (Q[:, 1] - ry) * np.cos(a)
    inside = (np.abs(x) <= rw / 2 - shrink_mm) & (np.abs(y) <= rh / 2 - shrink_mm)
    # robust start: drop points of THEIR cut that are far (> 2 * band) from a global cubic surface, so that a coplanar
    # attachment (gripper bracket, wheel rim) cannot seed the local surface
    coef = np.linalg.lstsq(F(q[:, 0], q[:, 1]), q[:, 2], rcond=None)[0]
    cur = q[np.abs(q[:, 2] - F(q[:, 0], q[:, 1]) @ coef) < 2 * band_mm]
    for _ in range(passes):
        surf = local_surface(cur, Q[:, :2].min(0) - 8, Q[:, :2].max(0) + 8)
        keep = inside & (np.abs(Q[:, 2] - surf(Q[:, 0], Q[:, 1])) < band_mm)
        cur = Q[keep]
    return fused[keep], dict(outline_mm=[round(rw, 1), round(rh, 1)], points=int(keep.sum()))


def local_surface(pts, lo, hi, cell=4.0, sigma_cells=1.0, min_pts=3):
    """median height per cell -> gaps interpolated (linear, nearest outside the hull) -> Gaussian smoothing;
    returns a bilinear interpolator s(u, v)."""
    from scipy.interpolate import griddata, RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    xs = np.arange(lo[0], hi[0] + cell, cell); ys = np.arange(lo[1], hi[1] + cell, cell)
    ix = np.clip(((pts[:, 0] - xs[0]) / cell).astype(int), 0, len(xs) - 1); iy = np.clip(((pts[:, 1] - ys[0]) / cell).astype(int), 0, len(ys) - 1)
    lin = iy * len(xs) + ix; o = np.argsort(lin); ls = lin[o]; hs = pts[o, 2]; b = np.flatnonzero(np.diff(ls)) + 1
    Z = np.full(len(xs) * len(ys), np.nan)
    for gid, g in zip(ls[np.r_[0, b]], np.split(hs, b)):
        if len(g) >= min_pts: Z[gid] = np.median(g)
    Z = Z.reshape(len(ys), len(xs)); X, Y = np.meshgrid(xs + cell / 2, ys + cell / 2); m = np.isfinite(Z)
    Zf = griddata((X[m], Y[m]), Z[m], (X, Y), method="linear")
    Zn = griddata((X[m], Y[m]), Z[m], (X, Y), method="nearest"); Zf = np.where(np.isfinite(Zf), Zf, Zn)
    Zs = gaussian_filter(Zf, sigma_cells)
    it = RegularGridInterpolator((ys + cell / 2, xs + cell / 2), Zs, bounds_error=False, fill_value=None)
    return lambda u, v: it(np.column_stack([v, u]))


def coverage(points, plate, step=2.0, shrink_mm=3.0, min_pts=3):
    """share of step-mm cells inside the plate's outline (shrunk) holding >= min_pts points"""
    c = plate.mean(0); _, _, Vt = np.linalg.svd(plate - c, full_matrices=False)
    q = (plate - c) @ Vt.T; P = (points - c) @ Vt.T
    (rx, ry), (rw, rh), ang = cv2.minAreaRect(q[:, :2].astype(np.float32)); a = np.radians(ang)
    x = (P[:, 0] - rx) * np.cos(a) + (P[:, 1] - ry) * np.sin(a); y = -(P[:, 0] - rx) * np.sin(a) + (P[:, 1] - ry) * np.cos(a)
    W, H = rw - 2 * shrink_mm, rh - 2 * shrink_mm; ok = (np.abs(x) <= W / 2) & (np.abs(y) <= H / 2)
    ix = np.floor((x[ok] + W / 2) / step).astype(int); iy = np.floor((y[ok] + H / 2) / step).astype(int)
    cnt = np.zeros((int(H // step) + 1, int(W // step) + 1)); np.add.at(cnt, (iy, ix), 1)
    return float((cnt >= min_pts).mean())


def fit_known_rectangle(points, width_mm, length_mm, search_mm=50.0, step=2.0, shrink_mm=2.0):
    """Cut a point set (already roughly the sheet, maybe with coplanar attachments) with a rectangle of the KNOWN
    sheet size. Orientation = the point set's minimum-area rectangle; position = the offset (within +-search_mm)
    that maximises the number of occupied step-mm cells inside. Both orientations of the rectangle are tried.
    Returns (mask of points inside, dict(center_uv, angle_deg, size, score)) in the plane frame given."""
    c = points.mean(0); _, _, Vt = np.linalg.svd(points - c, full_matrices=False)
    q = (points - c) @ Vt.T
    (rx, ry), (rw, rh), ang = cv2.minAreaRect(q[:, :2].astype(np.float32)); a = np.radians(ang)
    x = (q[:, 0] - rx) * np.cos(a) + (q[:, 1] - ry) * np.sin(a); y = -(q[:, 0] - rx) * np.sin(a) + (q[:, 1] - ry) * np.cos(a)
    ix = np.floor(x / step).astype(int); iy = np.floor(y / step).astype(int)
    occ = {}
    for k in zip(ix, iy): occ[k] = occ.get(k, 0) + 1
    cells = np.array([k for k, n in occ.items() if n >= 3]); best = None
    offs = np.arange(-search_mm, search_mm + step, step)
    for W, L in ((width_mm, length_mm), (length_mm, width_mm)):
        hw, hl = (W - 2 * shrink_mm) / 2 / step, (L - 2 * shrink_mm) / 2 / step
        for ox in offs:
            for oy in offs:
                s = int(np.sum((np.abs(cells[:, 0] + 0.5 - ox / step) <= hw) & (np.abs(cells[:, 1] + 0.5 - oy / step) <= hl)))
                if best is None or s > best[0]: best = (s, ox, oy, W, L)
    s, ox, oy, W, L = best
    inside = (np.abs(x - ox) <= W / 2 - shrink_mm) & (np.abs(y - oy) <= L / 2 - shrink_mm)
    return inside, dict(score_cells=s, offset_mm=[float(ox), float(oy)], size_mm=[W, L], angle_deg=float(ang), frame=(c, Vt, rx, ry, a))
