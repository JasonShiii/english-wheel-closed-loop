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
import danish_run2_measure as DM          # noqa: E402  sheet finder, frame, camera correction, despeckle
import cfg                                # noqa: E402  the sheet, crop line, reachable zone of the trained model
import pairs as NP                        # noqa: E402  mask_crop, plane_removed
from . import scan_to_field as S          # noqa: E402  merge_capture

PASS_DIR = "pointcloud_pass{k}"


def pass_dir(run, k):
    return os.path.join(run, PASS_DIR.format(k=k))


def load_capture(pdir, calib_dir):
    """merged points (device-1 frame, mm) and camera ids of one capture folder, gated to the rig volume"""
    xyz, ids = S.merge_capture(pdir, calib_dir, window=(1.0, 1e6), cameras=(1, 2, 3), with_ids=True)
    xyz = xyz.astype(np.float32)
    g = np.all((xyz > DM.GATE[:, 0]) & (xyz < DM.GATE[:, 1]), axis=1)
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


class RigMeasurer:
    """reference(run) once on the flat scan (pointcloud_pass0), then field(pass_dir) for every capture."""

    def __init__(self, calib_dir=None, step=cfg.STEP_MM):
        self.sheet = cfg.SHEET
        self.calib = calib_dir or DM.CALIB
        DM.CALIB = self.calib
        self.xg, self.yg = F.grid_for(self.sheet, step)
        self.frame = self.corr = self.sheet_mask = self.info = None

    # ---- the flat scan ------------------------------------------------------------------
    def reference(self, run):
        """run must contain pointcloud_pass0 (the flat gripped sheet at the scan pose)."""
        P0, I0, Fr, corr, info = DM.prepare(run)
        xy0, h0, m0 = DM.field_points(P0, I0, Fr, corr)
        Z0 = grid_nodes(xy0[m0], h0[m0], self.xg, self.yg)
        W, L = self.sheet.width_mm, self.sheet.length_mm
        GX, GY = np.meshgrid(self.xg, self.yg)
        sheet = np.isfinite(Z0) & (np.abs(Z0) < DM.FLAT_TOL)
        sheet &= ~((np.abs(GX) > W / 2 - 14) & (GY < -15))          # clamp hardware along the gripped corners
        self.frame, self.corr, self.sheet_mask = Fr, corr, sheet
        self.info = {**info, "flat_cells": int(sheet.sum()), "flat_z_p0p5_p99p5": [round(float(np.nanpercentile(Z0[sheet], q)), 2) for q in (0.5, 99.5)]}
        return self._finish(Z0, {"source": "flat reference", "run": os.path.abspath(run)})

    def field(self, pdir):
        """one capture folder (device_<1|2|3>_pointcloud_*.ply) -> HeightField, cropped, plane removed"""
        if self.frame is None:
            raise RuntimeError("call reference(run) on the flat scan first")
        P, I = load_capture(pdir, self.calib)
        xy, h, m = DM.field_points(P, I, self.frame, self.corr)
        Z = grid_nodes(xy[m], h[m], self.xg, self.yg)
        return self._finish(Z, {"source": os.path.abspath(pdir), "band_points": int(m.sum())})

    def _finish(self, Z, meta):
        Z, nbad = DM.despeckle(np.where(self.sheet_mask, Z, np.nan))
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
        DM.CALIB = self.calib
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
        fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
