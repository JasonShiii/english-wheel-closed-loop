"""Measurement for the rig (2026-10-28): a capture folder -> HeightField on the loop's grid.

The chain is the one that works on Danish's 10-21/22 scans (bridge_checks/danish_run2_measure.py,
slides 2026-10-22_pointcloud_pipeline.pptx): merge the three cameras with the rig calibration -> gate box ->
sheet finder (RANSAC planes on the FLAT scan, the plane whose inliers best fill a 231 x 174 rectangle)
-> sheet frame from the flat scan (x along the 231 mm side, +y toward the free edge; the gripper bar beyond
the gripped edge decides the sign) -> per-camera plane correction from the flat scan -> every capture:
points within BAND of the flat plane inside the rectangle -> height above the flat sheet per grid node
(median, 14 mm fallback for sparse spots) -> hardware mask from the flat scan -> despeckle -> crop below
cfg.CROP_MM -> best-fit plane removed. The last two steps are pairs.py's, i.e. exactly what the simulated
training fields went through.

Every capture is taken at the same withdrawn scan pose (protocol 2026-10-12), so the frame and the camera
correction from the flat scan apply to every later capture; no robot pose is needed. If a run folder also
records the TCP pose per scan, it is only a check.

The grid is frames.grid_for(cfg.SHEET, STEP): the same nodes the targets (pairs.field_of) are on.

2026-10-29 (handoff section 0, dose_fit/RECALIBRATION_231x124_1mm_2026-10-29.md): the chain is now
bridge_checks/danish_run2_measure_231x124.Chain with the sheet size as a parameter. The rectangle fit searches
ORIENTATION and position (Chain.fit_rect_rot, FFT occupancy correlation) instead of plate_refill's min-area-rect
orientation, which was 7-22 deg off on the 10-21/22 scans; the hardware mask covers |x| > W/2 - 14 up to 37 mm above
the gripped edge (the old -15 mm was 35-37 mm above the real gripped edge of those runs).
"""
import json
import os
import sys

import numpy as np

from . import frames as F

RT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # real_testbed
for _p in (os.path.join(RT, "bridge_checks"), os.path.join(os.path.dirname(RT), "experiments", "newsheet_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import danish_run2_measure as DM          # noqa: E402  GATE, FLAT_TOL (the constants are the same in both chains)
import danish_run2_measure_231x124 as D124  # noqa: E402  the chain with the sheet size as a parameter, fit_rect_rot
import cfg                                # noqa: E402  the sheet, crop line, reachable zone of the trained model
import pairs as NP                        # noqa: E402  mask_crop, plane_removed
from . import scan_to_field as S          # noqa: E402  merge_capture

PASS_DIR = "pointcloud_pass{k}"


def pass_dir(run, k):
    return os.path.join(run, PASS_DIR.format(k=k))


HW_MM = 37.0        # hardware mask height above the gripped edge (recalibration 2026-10-29)


def load_capture(pdir, calib_dir):
    """merged points (device-1 frame, mm) and camera ids of one capture folder, gated to the rig volume"""
    xyz, ids = S.merge_capture(pdir, calib_dir, window=(1.0, 1e6), cameras=(1, 2, 3), with_ids=True)
    xyz = xyz.astype(np.float32)
    g = np.all((xyz > D124.GATE[:, 0]) & (xyz < D124.GATE[:, 1]), axis=1)
    return xyz[g], ids[g]


def grid_nodes(xy, h, xg, yg, min_pts=3, fallback_cells=3):
    """danish_run2_measure.grid on arbitrary node vectors: median of >= min_pts points per node, a 14 mm
    fallback box for sparse spots, nothing interpolated across larger gaps."""
    dx, dy = xg[1] - xg[0], yg[1] - yg[0]
    Z = np.full((len(yg), len(xg)), np.nan)
    ix = np.round((xy[:, 0] - xg[0]) / dx).astype(int); iy = np.round((xy[:, 1] - yg[0]) / dy).astype(int)
    ok = (ix >= 0) & (ix < len(xg)) & (iy >= 0) & (iy < len(yg)); ix, iy, hv = ix[ok], iy[ok], h[ok]
    if len(hv) == 0:
        return Z
    lin = iy * len(xg) + ix; o = np.argsort(lin); ls = lin[o]; hs = hv[o]; b = np.flatnonzero(np.diff(ls)) + 1
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


def planner_box(ax, sheet, label=True):
    """outline of the part of the sheet that goes to the planner (above the crop line, full width) and, dashed, the
    wheel-reachable zone, on an axis in sheet-frame mm"""
    from matplotlib.patches import Rectangle
    W, L, yc = sheet.width_mm, sheet.length_mm, cfg.crop_y()
    ax.add_patch(Rectangle((-W / 2, yc), W, L / 2 - yc, fill=False, ec="#e8590c", lw=1.8, zorder=5))
    x0, x1, y0, y1 = cfg.reachable_rect()                 # the model's reachable zone (x +-72.1, y -67..75)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#e8590c", lw=1.0, ls="--", zorder=5))
    if label:
        ax.text(-W / 2 + 2, L / 2 - 2, "planner input (above the crop line)", color="#e8590c", fontsize=7, va="top", zorder=6)
        ax.text(x0 + 2, y1 - 2, "wheel-reachable zone", color="#e8590c", fontsize=6, va="top", zorder=6)


class FakeFieldMeasurer:
    """dry runs without a sheet (2026-10-29): the robot runs and scans for real and every real capture is measured and
    saved for the record (field_*_real.npz/png; failures tolerated), but the field handed to the controller and the
    planner is a stand-in: <fields_dir>/field_<c>.npz after c passes executed in total (the largest c available), e.g.
    the simulated trajectory behind the target. advance(executed) is called by run_loop after every block."""
    def __init__(self, real, fields_dir):
        import glob as _g
        self.real, self.dir = real, os.path.abspath(fields_dir)
        self.avail = sorted(int(os.path.basename(p)[6:-4]) for p in _g.glob(os.path.join(self.dir, "field_*.npz")))
        if not self.avail:
            raise FileNotFoundError(f"no field_<c>.npz in {fields_dir}")
        self.xg, self.yg, self.cum, self.info = real.xg, real.yg, 0, {}

    def _fake(self, extra):
        c = max([a for a in self.avail if a <= self.cum] or [self.avail[0]])
        fld = F.HeightField.load(os.path.join(self.dir, f"field_{c}.npz"))
        fld.meta.update({"source": f"FAKE: {self.dir}/field_{c}.npz (after {self.cum} pass(es) in total)", "fake_cycle": c,
                         "coverage_observed": fld.meta.get("coverage_observed", round(float(np.isfinite(fld.z_mm[fld.y_mm >= cfg.crop_y()]).mean()), 3)), **extra})
        return fld

    def _real(self, fn, *a):
        try:
            fld = fn(*a); return fld, None
        except Exception as e:                            # no sheet in the scan: record the failure, keep going
            return None, f"{type(e).__name__}: {e}"

    def advance(self, executed):
        self.cum += int(executed)

    def reference(self, run):
        fld, err = self._real(self.real.reference, run)
        self.info = {**(self.real.info or {}), "fake_fields": self.dir, "real_reference_error": err}
        self._last_real = fld
        return self._fake({"real_capture": os.path.abspath(os.path.join(run, "pointcloud_pass0")), "real_error": err})

    def field(self, pdir):
        fld, err = self._real(self.real.field, pdir) if self.real.frame is not None else (None, "no real reference")
        self._last_real = fld
        return self._fake({"real_capture": os.path.abspath(pdir), "real_error": err})

    def last_real(self):
        return getattr(self, "_last_real", None)

    def full_view(self, pdir=None):
        try:
            return self.real.full_view(pdir) if self.real.frame is not None else None
        except Exception:
            return None

    def save(self, d):
        os.makedirs(d, exist_ok=True)
        if self.real.frame is not None:
            self.real.save(d)
        json.dump({"fake_fields": self.dir, "cum": self.cum, "info": self.info}, open(os.path.join(d, "fake.json"), "w"), indent=1, default=str)

    def load(self, d):
        if os.path.exists(os.path.join(d, "reference.npz")):
            self.real.load(d)
        m = json.load(open(os.path.join(d, "fake.json"))); self.cum = m.get("cum", 0)
        return self

    def preview(self, path, fld=None):
        if self.real.frame is not None:
            self.real.preview(path, self.last_real())


class RigMeasurer:
    """reference(run) once on the flat scan (pointcloud_pass0), then field(pass_dir) for every capture."""

    def __init__(self, calib_dir=None, step=cfg.STEP_MM, sheet=None, hw_mm=HW_MM, legacy_fit=False):
        self.sheet = sheet or cfg.SHEET
        self.calib = calib_dir or D124.CALIB
        D124.CALIB = DM.CALIB = self.calib
        self.hw_mm = float(hw_mm)
        self.chain = D124.Chain(self.sheet.width_mm, self.sheet.length_mm, band=DM.BAND, hw_mm=self.hw_mm,
                                tag=f"{self.sheet.width_mm:g}x{self.sheet.length_mm:g}_rig", legacy_fit=legacy_fit)
        self.xg, self.yg = F.grid_for(self.sheet, step)
        self.frame = self.corr = self.sheet_mask = self.info = None
        self._flat_raw = None

    # ---- the flat scan ------------------------------------------------------------------
    def reference(self, run):
        """run must contain pointcloud_pass0 (the flat gripped sheet at the scan pose)."""
        D124.CALIB = self.calib
        P0, I0, Fr, corr, info = self.chain.prepare(run)
        xy0, h0, m0 = self.chain.field_points(P0, I0, Fr, corr)
        Z0 = grid_nodes(xy0[m0], h0[m0], self.xg, self.yg)
        W, L = self.sheet.width_mm, self.sheet.length_mm
        GX, GY = np.meshgrid(self.xg, self.yg)
        sheet = np.isfinite(Z0) & (np.abs(Z0) < D124.FLAT_TOL)
        sheet &= ~((np.abs(GX) > W / 2 - 14) & (GY < -L / 2 + self.hw_mm))   # clamp hardware along the gripped corners
        self.frame, self.corr, self.sheet_mask = Fr, corr, sheet
        self._flat_raw = Z0
        info = {k: v for k, v in info.items() if k != "rect_score_curve"}
        self.info = {**info, "hw_mm": self.hw_mm, "flat_cells": int(sheet.sum()),
                     "flat_z_p0p5_p99p5": [round(float(np.nanpercentile(Z0[sheet], q)), 2) for q in (0.5, 99.5)] if sheet.any() else None}
        return self._finish(Z0, {"source": "flat reference", "run": os.path.abspath(run)})

    def field(self, pdir):
        """one capture folder (device_<1|2|3>_pointcloud_*.ply) -> HeightField, cropped, plane removed"""
        if self.frame is None:
            raise RuntimeError("call reference(run) on the flat scan first")
        Z, nb = self.raw_grid(pdir)
        return self._finish(Z, {"source": os.path.abspath(pdir), "band_points": nb})

    def raw_grid(self, pdir):
        """heights above the flat sheet on the grid, before mask / despeckle / crop / plane removal"""
        P, I = load_capture(pdir, self.calib)
        xy, h, m = self.chain.field_points(P, I, self.frame, self.corr)
        return grid_nodes(xy[m], h[m], self.xg, self.yg), int(m.sum())

    def full_view(self, pdir=None):
        """DISPLAY ONLY: the whole sheet incl. the cropped band below the crop line (pdir None = the flat scan), with the
        chain's mask and despeckle and the best-fit plane of the observed region (the same plane the planner field
        removes) subtracted everywhere. Shows what the crop removes; never fed to the planner or the controller."""
        Z = self._flat_raw if pdir is None else self.raw_grid(pdir)[0]
        if Z is None:
            return None
        Z, _ = self.masked(Z)
        X, Y = np.meshgrid(self.xg, self.yg); m = np.isfinite(Z) & (Y >= cfg.crop_y())
        if m.sum() < 10:
            return None
        A = np.column_stack([X[m], Y[m], np.ones(m.sum())]); cf = np.linalg.lstsq(A, Z[m], rcond=None)[0]
        return F.HeightField(self.xg.astype(float), self.yg.astype(float), (Z - (cf[0] * X + cf[1] * Y + cf[2])).astype(np.float32),
                             {"source": "full sheet, display only", "crop_y": cfg.crop_y()})

    def masked(self, Z):
        """the chain's own end product: sheet mask from the flat scan, despeckled (what danish_run2_measure_231x124 stores)"""
        return D124.Chain.despeckle(np.where(self.sheet_mask, Z, np.nan))

    def _finish(self, Z, meta):
        Z, nbad = self.masked(Z)
        fld = F.HeightField(self.xg.astype(float), self.yg.astype(float), Z.astype(np.float32),
                            {**meta, "despeckled_cells": int(nbad), "calib": self.calib,
                             "coverage_sheet": round(float(np.isfinite(Z).mean()), 3)})
        fld = NP.plane_removed(NP.mask_crop(fld))
        fld.meta["coverage_observed"] = round(float(np.isfinite(fld.z_mm[fld.y_mm >= cfg.crop_y()]).mean()), 3)
        return fld

    # ---- persistence (resume) -------------------------------------------------------------
    def save(self, d):
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "reference.npz"), c=self.frame["c"], ex=self.frame["ex"], ey=self.frame["ey"], n=self.frame["n"],
                 sheet_mask=self.sheet_mask, corr_cams=np.array(list(self.corr.keys())), corr_vals=np.array(list(self.corr.values())))
        json.dump({"info": self.info, "calib": self.calib}, open(os.path.join(d, "reference.json"), "w"), indent=1)

    def load(self, d):
        z = np.load(os.path.join(d, "reference.npz"))
        self.frame = {k: z[k] for k in ("c", "ex", "ey", "n")}
        self.sheet_mask = z["sheet_mask"]
        self.corr = {int(c): v for c, v in zip(z["corr_cams"], z["corr_vals"])}
        m = json.load(open(os.path.join(d, "reference.json"))); self.info, self.calib = m["info"], m["calib"]
        D124.CALIB = DM.CALIB = self.calib
        return self

    def preview(self, path, fld=None):
        """the flat field and the sheet mask, to check the frame (gripped edge at the bottom, +y up)"""
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2)); ext = (self.xg[0], self.xg[-1], self.yg[0], self.yg[-1])
        ax[0].imshow(self.sheet_mask, origin="lower", extent=ext, cmap="Greys"); ax[0].set_title(f"sheet mask from the flat scan ({int(self.sheet_mask.sum())} nodes)\ninside fraction {self.info.get('inside_fraction')}, bar beyond {self.info.get('beyond_edge_points')}", fontsize=8)
        if fld is not None:
            im = ax[1].imshow(np.ma.masked_invalid(fld.z_mm), origin="lower", extent=ext, cmap="viridis"); plt.colorbar(im, ax=ax[1], fraction=0.04)
            ax[1].set_title(f"flat field after crop + plane removal: z {fld.zrange()[0]:.1f}..{fld.zrange()[1]:.1f} mm\ncamera offsets {self.info.get('camera_flat_offset_mm')}", fontsize=8)
        for a in ax:
            a.axhline(cfg.crop_y(), color="r", ls="--", lw=0.8); a.set_xlabel("x (mm)"); a.set_ylabel("y (mm), gripped edge at the bottom")
            planner_box(a, self.sheet)
        fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
