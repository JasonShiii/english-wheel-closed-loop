"""Camera point cloud of one capture -> height field of the sheet in the sheet frame.

Input: either one rig camera's cloud or the three rig cameras merged into the device_1
frame with the rig calibration (load_capture(..., camera="merged", calib_dir=...)). Merged
is the normal case (more of the sheet is seen); the merge is only as good as the rig
calibration (the metrology team's ICP report of 2026-09-28 gives ~6 mm median cross-camera
distance), so check the single cameras against each other on the flat reference capture.

Steps
  1. depth window: keep points 200-700 mm from the camera (the sheet is 250-600 mm away,
     floor and walls are farther)
  2. reference plane: fitted once on the FLAT reference scan taken before forming and kept
     for the whole run; h = signed distance to it, positive toward the camera
  3. sheet pose in that plane:
       - on the flat reference scan, from the visible outline: the known rectangle is fitted
         to the free edge and the two side edges (the gripped edge is hidden by the gripper)
       - on every later capture, from the robot motion: the gripper holds the sheet rigidly
         and the robot moves it only within the plane, so the pose is the reference pose
         shifted by the change of the sheet point under the wheel (known from the toolpath).
         The capture's own outline is still fitted and compared ("pose check"); a mismatch
         means slip, a wrong capture-position convention or a wrong axis sign.
       - without robot information (old data) the outline fit is used directly; it anchors
         the free edge, which on a strongly curved sheet is pulled in by a few mm
  4. points -> sheet frame -> per-cell median on the HeightField grid
  5. cleanup: cells that the flat reference scan saw covered by something above the sheet are
     dropped (the upper wheel, fixed in the camera view; the gripper, fixed on the sheet); cells that disagree with their 7 x 7 neighbourhood
     median by more than 3 mm are dropped; one cell is eroded off every boundary (mixed
     pixels at depth edges)
  6. the flat reference field is subtracted (sag, waviness, plane residual); where the
     reference did not see the sheet, a smooth quadratic fit of it stands in
  7. the clamp band (y below the wheel line) is set to no-data, as in every model input
"""
import glob
import json
import os
from dataclasses import dataclass, field as dc_field

import numpy as np

from . import frames
from .frames import Sheet, HeightField

DEFAULT_CAMERA = 3
DEPTH_WINDOW_MM = (200.0, 700.0)
CELL_MM = 4.0               # coarse cell for outline and obstruction analysis
FLAT_BAND_MM = 5.0          # |h| for "on the flat sheet" in the reference scan
BAND_MM = 100.0             # |h| for "on the sheet" in later (formed) captures
OUTLIER_MM = 3.0            # cell vs neighbourhood median
POSE_CHECK_DEG, POSE_CHECK_MM = 2.0, 8.0     # gross errors only: a wrong capture position or axis sign is off by
                                            # tens of mm; a strongly curled sheet moves its outline by a few mm


# ---------------------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------------------
_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
              "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply(src):
    """x, y, z of the vertex element of a PLY file (path or bytes) as (n, 3) float64 mm.
    Handles binary little-endian (what the rig writes) and ascii; vertex element first."""
    data = src if isinstance(src, (bytes, bytearray)) else open(src, "rb").read()
    end = data.index(b"end_header")
    body = data.index(b"\n", end) + 1
    lines = data[:end].decode("ascii", "replace").splitlines()
    fmt, n, props, in_vertex = None, 0, [], False
    for ln in lines:
        t = ln.split()
        if not t:
            continue
        if t[0] == "format":
            fmt = t[1]
        elif t[0] == "element":
            in_vertex = t[1] == "vertex"
            if in_vertex:
                n = int(t[2])
        elif t[0] == "property" and in_vertex:
            if t[1] == "list":
                raise ValueError("list properties in the vertex element are not supported")
            props.append((t[2], _PLY_TYPES[t[1]]))
    if fmt == "binary_little_endian":
        arr = np.frombuffer(data, dtype=np.dtype([(nm, "<" + ty) for nm, ty in props]), count=n, offset=body)
    elif fmt == "ascii":
        txt = data[body:].decode("ascii").split("\n")[:n]
        arr = np.loadtxt(txt, ndmin=2)
        arr = {nm: arr[:, i] for i, (nm, _) in enumerate(props)}
    else:
        raise ValueError(f"unsupported PLY format {fmt}")
    return np.stack([np.asarray(arr["x"], float), np.asarray(arr["y"], float), np.asarray(arr["z"], float)], 1)


def load_capture(pass_dir, camera=DEFAULT_CAMERA, calib_dir=None):
    """points of one capture folder written by the robot (pointcloud_pass<k>/, baseline/).

    camera = 1, 2 or 3: that camera's device_<camera>_pointcloud_*.ply, in its own frame.
    camera = "merged": the three rig cameras merged into the device_1 frame with the rig
    calibration in calib_dir (the robot repository's code/mvs_recorder/"calibration results":
    extrinsics.json from the checkerboard, refined by icp/refined_transforms.json when present,
    exactly as the metrology team's merge viewer does). Each camera is clipped to its own depth
    window before the transform."""
    if camera == "merged":
        return merge_capture(pass_dir, calib_dir)
    hits = sorted(glob.glob(os.path.join(pass_dir, f"device_{camera}_pointcloud_*.ply")))
    if not hits:
        raise FileNotFoundError(f"no device_{camera}_pointcloud_*.ply in {pass_dir}")
    return read_ply(hits[0])


def load_rig_calibration(calib_dir):
    """{device index: 4x4 T_device1_from_device} in mm. The ICP-refined file replaces the
    checkerboard one when it exists and is marked accepted for the same calibration run."""
    ext = json.load(open(os.path.join(calib_dir, "extrinsics.json")))
    data, source = ext, "extrinsics.json"
    icp_path = os.path.join(calib_dir, "icp", "refined_transforms.json")
    if os.path.isfile(icp_path):
        icp = json.load(open(icp_path))
        if icp.get("status") == "accepted" and icp.get("calibration_run") == ext.get("source_run"):
            data, source = icp, "icp/refined_transforms.json"
    T = {int(k.split("_")[1]): np.array(v, float) for k, v in data["devices"].items()}
    return T, source


def merge_capture(pass_dir, calib_dir, window=DEPTH_WINDOW_MM, cameras=(1, 2, 3), with_ids=False):
    """merged cloud in the device_1 frame; with_ids also returns the camera index of every point."""
    if not calib_dir:
        raise ValueError("camera='merged' needs the rig calibration folder (calib_dir)")
    T, _ = load_rig_calibration(calib_dir)
    parts, ids = [], []
    for dev in cameras:
        hits = sorted(glob.glob(os.path.join(pass_dir, f"device_{dev}_pointcloud_*.ply")))
        if not hits or dev not in T:
            continue
        P = depth_window(read_ply(hits[0]), window)
        parts.append(P @ T[dev][:3, :3].T + T[dev][:3, 3])
        ids.append(np.full(len(P), dev, np.int8))
    if not parts:
        raise FileNotFoundError(f"no calibrated rig camera clouds in {pass_dir}")
    return (np.concatenate(parts), np.concatenate(ids)) if with_ids else np.concatenate(parts)


CAMERA_AGREEMENT_MM = 2.0     # per-camera median height offset on the flat reference above this -> warning


def camera_agreement(P, ids, ref):
    """median height of each camera's points on the flat reference sheet, relative to the plane
    fitted to all of them: how well the rig calibration lines the cameras up (mm)."""
    u, v, h = ref.plane_coords(P)
    xy = (np.stack([u, v], 1) - ref.t2) @ ref.R2
    s = ref.sheet
    on = (np.abs(xy[:, 0]) < s.width_mm / 2 - 10) & (xy[:, 1] > s.wheel_line_y) & (xy[:, 1] < s.length_mm / 2 - 10) & (np.abs(h) < FLAT_BAND_MM)
    out = {}
    for dev in np.unique(ids):
        m = on & (ids == dev)
        if m.sum() > 200:
            out[f"camera_{int(dev)}"] = {"median_mm": round(float(np.median(h[m])), 2),
                                        "mad_mm": round(float(np.median(np.abs(h[m] - np.median(h[m])))), 2), "points": int(m.sum())}
    meds = [c["median_mm"] for c in out.values()]
    out["spread_mm"] = round(float(max(meds) - min(meds)), 2) if len(meds) > 1 else 0.0
    out["ok"] = bool(out["spread_mm"] <= CAMERA_AGREEMENT_MM)
    return out


# ---------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------
def depth_window(P, window=DEPTH_WINDOW_MM):
    return P[(P[:, 2] > window[0]) & (P[:, 2] < window[1])]


def fit_plane(P, tol=1.5, iters=800, seed=0):
    """Dominant plane (RANSAC on a subsample, refined by PCA of its inliers).
    Returns origin, e1, e2, n with (e1, e2, n) right-handed and n pointing at the camera."""
    rng = np.random.default_rng(seed)
    Q = P[rng.choice(len(P), min(len(P), 60000), replace=False)]
    best_cnt, best = 0, None
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        nn = np.linalg.norm(nrm)
        if nn < 1e-9:
            continue
        nrm /= nn
        cnt = int(np.sum(np.abs((Q - a) @ nrm) < tol))
        if cnt > best_cnt:
            best_cnt, best = cnt, (a, nrm)
    a, nrm = best
    inl = Q[np.abs((Q - a) @ nrm) < tol]
    origin = inl.mean(0)
    _, _, Vt = np.linalg.svd(inl - origin, full_matrices=False)
    n = Vt[2] if Vt[2] @ (-origin) > 0 else -Vt[2]
    e1 = Vt[0] - (Vt[0] @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return origin, e1, e2, n


def _cells(u, v, cell=CELL_MM, min_frac=0.3):
    """dense occupied cells of a 2D point set, largest connected group: (centres (k, 2), cell)."""
    from scipy import ndimage
    ue = np.arange(u.min() - cell, u.max() + 2 * cell, cell)
    ve = np.arange(v.min() - cell, v.max() + 2 * cell, cell)
    H, _, _ = np.histogram2d(u, v, bins=[ue, ve])
    occ = H > 0
    if not occ.any():
        return np.zeros((0, 2))
    thr = max(3.0, min_frac * np.median(H[occ]))
    dense = ndimage.binary_closing(H >= thr, iterations=1)
    lab, nl = ndimage.label(dense, structure=np.ones((3, 3)))
    if nl == 0:
        return np.zeros((0, 2))
    big = 1 + int(np.argmax(ndimage.sum(dense, lab, range(1, nl + 1))))
    iu, iv = np.nonzero(lab == big)
    return np.stack([ue[iu] + cell / 2, ve[iv] + cell / 2], 1)


def fit_outline(u, v, sheet, feed_hint_uv, up_sign=1.0, cell=CELL_MM):
    """Fit the known sheet rectangle to the visible outline in plane coordinates.

    The orientation comes from the minimum-area rectangle around the dense sheet cells; of
    its four side directions the one closest to feed_hint_uv becomes +y (gripped edge ->
    free edge). x is centred between the two side edges and y is anchored at the free edge;
    every edge is moved inward by half the excess of the measured over the nominal width
    (mixed pixels at depth edges make the sheet look slightly larger).
    Returns R2 (columns = sheet x and y axes in (u, v)), t2 (sheet centre in (u, v)), quality."""
    from scipy.spatial import ConvexHull
    C = _cells(u, v, cell)
    if len(C) < 30:
        raise ValueError(f"outline fit: only {len(C)} dense sheet cells")
    hull = C[ConvexHull(C).vertices]
    best_area, best_phi = np.inf, 0.0
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        phi = np.arctan2(b[1] - a[1], b[0] - a[0])
        r = hull @ np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        area = np.ptp(r[:, 0]) * np.ptp(r[:, 1])
        if area < best_area:
            best_area, best_phi = area, phi
    hint = np.asarray(feed_hint_uv, float) / np.linalg.norm(feed_hint_uv)
    cands = [best_phi + k * np.pi / 2 for k in range(4)]
    c = max(cands, key=lambda a: np.cos(a) * hint[0] + np.sin(a) * hint[1])
    yh = np.array([np.cos(c), np.sin(c)])
    xh = up_sign * np.array([yh[1], -yh[0]])
    xs, ys = C @ xh, C @ yh
    x_lo, x_hi = np.percentile(xs, 0.5), np.percentile(xs, 99.5)
    y_lo, y_hi = np.percentile(ys, 0.5), np.percentile(ys, 99.5)
    width = x_hi - x_lo + cell
    excess = (width - sheet.width_mm) / 2.0
    y_free = y_hi + cell / 2.0 - excess
    centre = 0.5 * (x_lo + x_hi) * xh + (y_free - sheet.length_mm / 2.0) * yh
    quality = {"width_measured_mm": round(float(width), 1), "width_nominal_mm": sheet.width_mm,
               "edge_excess_mm": round(float(excess), 2),
               "visible_length_mm": round(float(y_hi - y_lo + cell), 1),
               "feed_angle_deg": round(float(np.degrees(c)), 3), "n_cells": int(len(C))}
    return np.column_stack([xh, yh]), centre, quality


def _grid_index(x, y, sheet, xg, yg):
    """nearest grid node (frames.grid_for nodes include the sheet edges)."""
    dx, dy = xg[1] - xg[0], yg[1] - yg[0]
    ix = np.round((x - xg[0]) / dx).astype(int)
    iy = np.round((y - yg[0]) / dy).astype(int)
    ok = (ix >= 0) & (ix < len(xg)) & (iy >= 0) & (iy < len(yg))
    return ix, iy, ok


def grid_median(x, y, z, sheet, step, min_count=3):
    """per-node median of z (points nearest to each node of frames.grid_for(sheet, step));
    NaN where < min_count points."""
    xg, yg = frames.grid_for(sheet, step)
    ix, iy, ok = _grid_index(x, y, sheet, xg, yg)
    idx = iy[ok] * len(xg) + ix[ok]
    zz = z[ok]
    Z = np.full(len(xg) * len(yg), np.nan)
    if idx.size:
        order = np.lexsort((zz, idx))
        idx_s, z_s = idx[order], zz[order]
        starts = np.flatnonzero(np.r_[True, idx_s[1:] != idx_s[:-1]])
        counts = np.diff(np.r_[starts, len(idx_s)])
        med = 0.5 * (z_s[starts + (counts - 1) // 2] + z_s[starts + counts // 2])
        keep = counts >= min_count
        Z[idx_s[starts[keep]]] = med[keep]
    return xg, yg, Z.reshape(len(yg), len(xg))


def clean(Z, outlier_mm=OUTLIER_MM, win=7, erode=1):
    import warnings
    from scipy import ndimage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN windows
        loc = ndimage.generic_filter(Z, np.nanmedian, size=win, mode="constant", cval=np.nan)
    Z = np.where(np.abs(Z - loc) > outlier_mm, np.nan, Z)
    if erode:
        valid = ndimage.binary_erosion(np.isfinite(Z), structure=np.ones((3, 3)), iterations=erode)
        Z = np.where(valid, Z, np.nan)
    return Z


def quadratic_fill(Z, xg, yg):
    """Z with its NaN cells replaced by a least-squares quadratic surface through the valid ones
    (clipped to the measured range, so far-away holes are not extrapolated wildly)."""
    Y, X = np.meshgrid(yg, xg, indexing="ij")
    m = np.isfinite(Z)
    A = lambda X, Y: np.stack([np.ones_like(X), X, Y, X * X, X * Y, Y * Y], -1)
    coef, *_ = np.linalg.lstsq(A(X[m], Y[m]), Z[m], rcond=None)
    fill = np.clip(A(X, Y) @ coef, Z[m].min() - 0.5, Z[m].max() + 0.5)   # no wild extrapolation
    return np.where(m, Z, fill)


# ---------------------------------------------------------------------------------------
# the reference (flat sheet, before forming)
# ---------------------------------------------------------------------------------------
@dataclass
class Reference:
    sheet: Sheet
    camera: object                # 1, 2, 3 or "merged"
    origin: np.ndarray            # reference plane, camera frame (mm)
    e1: np.ndarray
    e2: np.ndarray
    n: np.ndarray                 # toward the camera
    R2: np.ndarray                # sheet x / y axes in plane (u, v) coordinates (columns)
    t2: np.ndarray                # sheet centre in (u, v) at the reference capture
    up_sign: float                # +1: the sheet's +z (the model's "up") faces the camera
    contact_xy: np.ndarray        # sheet-frame point under the wheel at the reference capture
    obstruction: dict             # {"u0", "v0", "cell", "mask"}: plane cells covered from above (wheel)
    baseline: HeightField         # reference field, holes filled (to subtract)
    gripper_mask: np.ndarray = None   # grid nodes (sheet frame) covered by the gripper, which moves with the sheet
    outline: dict = dc_field(default_factory=dict)
    depth_window: tuple = DEPTH_WINDOW_MM   # None for merged clouds (already clipped per camera)

    def plane_coords(self, P):
        d = P - self.origin
        return d @ self.e1, d @ self.e2, d @ self.n

    def pose_for_contact(self, contact_xy):
        """sheet pose when the sheet point contact_xy is under the wheel (pure in-plane shift)."""
        return self.R2, self.t2 - self.R2 @ (np.asarray(contact_xy, float) - self.contact_xy)

    def obstructed(self, u, v):
        ob = self.obstruction
        iu = np.floor((u - ob["u0"]) / ob["cell"]).astype(int)
        iv = np.floor((v - ob["v0"]) / ob["cell"]).astype(int)
        M = ob["mask"]
        ok = (iu >= 0) & (iu < M.shape[0]) & (iv >= 0) & (iv < M.shape[1])
        out = np.zeros(len(u), bool)
        out[ok] = M[iu[ok], iv[ok]]
        return out

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        meta = {"sheet": self.sheet.to_dict(), "camera": self.camera, "up_sign": self.up_sign,
                "depth_window": None if self.depth_window is None else list(self.depth_window), "outline": self.outline,
                **{k: getattr(self, k).tolist() for k in ("origin", "e1", "e2", "n", "R2", "t2", "contact_xy")},
                "obstruction": {k: float(self.obstruction[k]) for k in ("u0", "v0", "cell")}}
        json.dump(meta, open(os.path.join(d, "reference.json"), "w"), indent=1)
        np.save(os.path.join(d, "obstruction_mask.npy"), self.obstruction["mask"])
        np.save(os.path.join(d, "gripper_mask.npy"), self.gripper_mask)
        self.baseline.save(os.path.join(d, "baseline_field.npz"))

    @classmethod
    def load(cls, d):
        m = json.load(open(os.path.join(d, "reference.json")))
        ob = {**m["obstruction"], "mask": np.load(os.path.join(d, "obstruction_mask.npy"))}
        arr = {k: np.array(m[k], float) for k in ("origin", "e1", "e2", "n", "R2", "t2", "contact_xy")}
        return cls(sheet=Sheet.from_dict(m["sheet"]), camera=m["camera"], up_sign=m["up_sign"],
                   obstruction=ob, baseline=HeightField.load(os.path.join(d, "baseline_field.npz")),
                   gripper_mask=np.load(os.path.join(d, "gripper_mask.npy")),
                   outline=m["outline"], depth_window=None if m["depth_window"] is None else tuple(m["depth_window"]), **arr)


def make_reference(P, sheet, feed_dir_cam, camera=DEFAULT_CAMERA, up_toward_camera=True,
                   contact_xy=None, step=2.0, window=DEPTH_WINDOW_MM, flat_band_mm=FLAT_BAND_MM):
    """Build the run's reference from the flat scan taken before forming.

    feed_dir_cam: rough direction (camera frame) from the gripped edge to the free edge; it
    only has to be right within +-45 deg (it picks which rectangle side is which). Set once per
    rig setup (see the reference preview).
    contact_xy: sheet point under the wheel at this capture; default = the robot's forming start
    pose TARGET_POS_9 = planner origin = (0, wheel line).
    flat_band_mm: |h| that counts as "on the flat sheet"; keep the default for a truly flat
    sheet (only the validation on old, already formed captures needs more)."""
    P = depth_window(P, window) if window is not None else P
    up_sign = 1.0 if up_toward_camera else -1.0
    f = np.asarray(feed_dir_cam, float)
    # the sheet is the planar surface whose outline has the sheet's width; try the three
    # largest planes in view (a table or a stand can be as large as the visible sheet)
    rest, best = P, None
    for _ in range(3):
        if len(rest) < 1000:
            break
        origin, e1, e2, n = fit_plane(rest)
        d = P - origin
        u, v, h = d @ e1, d @ e2, d @ n
        flat = np.abs(h) < flat_band_mm
        try:
            R2, t2, quality = fit_outline(u[flat], v[flat], sheet, np.array([f @ e1, f @ e2]), up_sign)
            err = abs(quality["width_measured_mm"] - sheet.width_mm)
            if err < 25.0 and (best is None or err < best[0]):
                best = (err, origin, e1, e2, n, R2, t2, quality)
        except ValueError:
            pass
        dr = rest - origin
        rest = rest[np.abs(dr @ n) > 3.0]
    if best is None:
        raise ValueError("no planar surface with a sheet-sized outline in the depth window "
                         f"(sheet width {sheet.width_mm} mm); check the camera, the window and feed_dir_cam")
    _, origin, e1, e2, n, R2, t2, quality = best
    d = P - origin
    u, v, h = d @ e1, d @ e2, d @ n
    flat = np.abs(h) < flat_band_mm
    # sheet-frame coordinates of everything at the reference pose
    xy = (np.stack([u, v], 1) - t2) @ R2
    inside = (np.abs(xy[:, 0]) <= sheet.width_mm / 2 + 10) & (np.abs(xy[:, 1]) <= sheet.length_mm / 2 + 10)
    # obstruction: things above the sheet around the wheel contact (the upper wheel), fixed in the
    # camera view; taken wherever they are, also beyond the sheet outline (the sheet moves under
    # the wheel later). The gripper bar moves with the sheet and is left out.
    c_ref = np.array(contact_xy if contact_xy is not None else (0.0, sheet.wheel_line_y), float)
    c_w = R2 @ c_ref + t2
    near_wheel = np.hypot(u - c_w[0], v - c_w[1]) < 180.0
    gripper = (xy[:, 1] < -sheet.length_mm / 2 + 40.0) & (np.abs(xy[:, 0]) < sheet.width_mm / 2 + 20.0)
    above = near_wheel & ~gripper & (h > 4.0) & (h < 300.0)
    from scipy import ndimage
    u0, v0 = u.min() - CELL_MM, v.min() - CELL_MM
    shape = (int((u.max() - u0) / CELL_MM) + 2, int((v.max() - v0) / CELL_MM) + 2)
    M = np.zeros(shape, bool)
    M[((u[above] - u0) / CELL_MM).astype(int), ((v[above] - v0) / CELL_MM).astype(int)] = True
    lab, nl = ndimage.label(M, structure=np.ones((3, 3)))          # keep wheel-sized blobs only
    sizes = ndimage.sum(M, lab, range(1, nl + 1))
    M = np.isin(lab, 1 + np.flatnonzero(sizes >= 20))
    M = ndimage.binary_dilation(M, iterations=2)
    ref = Reference(sheet=sheet, camera=camera, origin=origin, e1=e1, e2=e2, n=n, R2=R2, t2=t2,
                    up_sign=up_sign, contact_xy=c_ref,
                    obstruction={"u0": u0, "v0": v0, "cell": CELL_MM, "mask": M},
                    baseline=None, outline=quality, depth_window=None if window is None else tuple(window))
    # gripper: things above the sheet along the gripped edge, fixed in the SHEET frame
    xg, yg = frames.grid_for(sheet, step)
    G = np.zeros((len(yg), len(xg)), bool)
    g = gripper & (h > 4.0) & (h < 300.0)
    ix, iy, ok = _grid_index(xy[g, 0], xy[g, 1], sheet, xg, yg)
    G[iy[ok], ix[ok]] = True
    ref.gripper_mask = ndimage.binary_dilation(G, iterations=2)
    keep = inside & flat & ~ref.obstructed(u, v)
    xg, yg, Z = grid_median(xy[keep, 0], xy[keep, 1], up_sign * h[keep], sheet, step)
    Z[ref.gripper_mask] = np.nan
    Z = clean(Z)
    ref.baseline = HeightField(xg, yg, quadratic_fill(Z, xg, yg).astype(np.float32),
                               {"source": "reference", "measured_cells": int(np.isfinite(Z).sum())})
    ref.outline["plane_normal_cam"] = [round(float(c), 4) for c in n]
    ref.outline["baseline_rms_mm"] = round(float(np.sqrt(np.nanmean((Z - np.nanmean(Z)) ** 2))), 3)
    return ref


# ---------------------------------------------------------------------------------------
# one capture -> height field
# ---------------------------------------------------------------------------------------
def scan_to_field(P, ref, contact_xy=None, step=2.0, band_mm=BAND_MM, subtract_baseline=True):
    """Height field (sheet frame, mm, NaN = no data) of one capture.

    contact_xy: sheet point under the wheel when the capture was taken (from the toolpath and
    the pass number). Given -> pose from the robot motion + pose check; None -> outline fit."""
    sheet = ref.sheet
    P = depth_window(P, ref.depth_window) if ref.depth_window is not None else P
    u, v, h = ref.plane_coords(P)
    uv = np.stack([u, v], 1)
    near = np.abs(h) < band_mm
    meta = {"source": "scan", "camera": ref.camera, "sheet": sheet.to_dict()}
    if contact_xy is not None:
        R2, t2 = ref.pose_for_contact(contact_xy)
        meta["pose_source"] = "robot_motion"
        meta["contact_xy"] = [float(c) for c in contact_xy]
    else:
        xy0 = (uv - ref.t2) @ ref.R2                      # search around the reference footprint
        zone = near & (np.abs(xy0[:, 0]) < sheet.width_mm / 2 + 60) & (np.abs(xy0[:, 1]) < sheet.length_mm / 2 + 60)
        R2, t2, q = fit_outline(u[zone], v[zone], sheet, ref.R2[:, 1], ref.up_sign)
        meta["pose_source"] = "outline"
        meta["outline"] = q
    xy = (uv - t2) @ R2
    inside = (np.abs(xy[:, 0]) <= sheet.width_mm / 2) & (np.abs(xy[:, 1]) <= sheet.length_mm / 2)
    keep = inside & near & ~ref.obstructed(u, v)
    if contact_xy is not None:                           # pose check against the capture's own outline
        try:
            # search well beyond the predicted footprint: a wrong prediction must not crop the sheet
            reach = 0.5 * np.hypot(sheet.width_mm, sheet.length_mm) + 150.0
            zone = near & (np.hypot(uv[:, 0] - t2[0], uv[:, 1] - t2[1]) < reach)
            R2o, t2o, q = fit_outline(u[zone], v[zone], sheet, R2[:, 1], ref.up_sign)
            dth = np.degrees(np.arctan2(R2o[1, 1] * R2[0, 1] - R2o[0, 1] * R2[1, 1], R2o[:, 1] @ R2[:, 1]))
            dxy = R2.T @ (t2o - t2)
            sides_seen = abs(q["width_measured_mm"] - ref.outline["width_measured_mm"]) < 15.0
            ok = bool(abs(dth) <= POSE_CHECK_DEG and abs(dxy[0]) <= POSE_CHECK_MM) if sides_seen else None
            meta["pose_check"] = {"dtheta_deg": round(float(dth), 3), "dx_mm": round(float(dxy[0]), 2),
                                  "dy_mm_free_edge": round(float(dxy[1]), 2), "ok": ok,
                                  "width_measured_mm": q["width_measured_mm"]}
        except ValueError as e:
            meta["pose_check"] = {"ok": None, "reason": str(e)}
    xg, yg, Z = grid_median(xy[keep, 0], xy[keep, 1], ref.up_sign * h[keep], sheet, step)
    if ref.gripper_mask is not None:
        Z[ref.gripper_mask] = np.nan
    Z = clean(Z)
    if ref.baseline.z_mm.shape != Z.shape:
        raise ValueError("reference baseline grid differs from the scan grid (step changed?)")
    if subtract_baseline:
        Z = Z - ref.baseline.z_mm
    meta["baseline_subtracted"] = bool(subtract_baseline)
    fld = frames.mask_clamp_band(HeightField(xg, yg, Z.astype(np.float32), meta), sheet)
    usable = fld.y_mm >= sheet.wheel_line_y
    fld.meta["coverage_usable"] = round(float(np.isfinite(fld.z_mm[usable]).mean()), 3)
    fld.meta["n_points"] = int(keep.sum())
    return fld


def reference_preview(P, ref, path):
    """What the reference fit found, for the one-time setup check: points near the plane
    (colour = height), the fitted sheet rectangle with its +x / +y axes, the gripped edge,
    the wheel line, and the cells masked as covered from above (the upper wheel)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = depth_window(P, ref.depth_window)
    u, v, h = ref.plane_coords(P)
    near = np.abs(h) < 60
    s, R2, t2 = ref.sheet, ref.R2, ref.t2
    corner = lambda x, y: R2 @ np.array([x, y]) + t2
    W2, L2 = s.width_mm / 2, s.length_mm / 2
    rect = np.array([corner(x, y) for x, y in ((-W2, -L2), (W2, -L2), (W2, L2), (-W2, L2), (-W2, -L2))])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(u[near][::4], v[near][::4], c=h[near][::4], s=0.4, cmap="viridis", vmin=-10, vmax=40)
    ob = ref.obstruction
    iu, iv = np.nonzero(ob["mask"])
    ax.scatter(ob["u0"] + (iu + 0.5) * ob["cell"], ob["v0"] + (iv + 0.5) * ob["cell"], s=2, c="r", alpha=0.3,
               label="covered from above (dropped)")
    ax.plot(rect[:, 0], rect[:, 1], "k-", lw=1.5, label="fitted sheet")
    g = np.array([corner(-W2, -L2), corner(W2, -L2)])
    ax.plot(g[:, 0], g[:, 1], color="#935116", lw=4, label="gripped edge")
    wl = np.array([corner(-W2, s.wheel_line_y), corner(W2, s.wheel_line_y)])
    ax.plot(wl[:, 0], wl[:, 1], "r--", lw=1, label="wheel line")
    c0 = corner(0, 0)
    for k, lab in ((0, "+x"), (1, "+y (feed)")):
        d = 60 * R2[:, k]
        ax.annotate(lab, c0 + d, c0, arrowprops=dict(arrowstyle="<-", lw=2), fontsize=9)
    ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"camera {ref.camera}: plane coordinates (mm); outline {ref.outline.get('width_measured_mm')} mm wide", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
