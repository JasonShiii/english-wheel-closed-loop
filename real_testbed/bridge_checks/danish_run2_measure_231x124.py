"""Copy of danish_run2_measure.py with the sheet size as a PARAMETER (2026-10-29). Danish's runs of 2026-10-21/22 were on
a 231 x 124 x 1.0 mm sheet, not the 231 x 174 x 0.6 assumed by the original chain (W, L = 231, 174 hard-coded: the
rectangle fitted to the flat scan, the grid, the gripped-edge test, the camera-correction window, the hardware mask and
every y-band of the metrics). Same chain, same constants, every length-dependent quantity derived from W, L:
  rectangle W x L on the flat plane (orientation AND position searched, fit_rect_rot; the original offset-only
  fit_known_rectangle takes its orientation from the min-area rectangle of all plane inliers and was 21.7 deg off on
  10-22 with the 124 rectangle) -> sheet frame -> per-camera correction on |x| < W/2-15, OBS_Y+12 < y < L/2-5
  -> band |h| < BAND inside the rectangle (+6 mm) -> 2 mm grid -> hardware mask |x| > W/2-14 and y < gripped edge +
  HW_MM -> despeckle -> metrics (danish_run2_metrics_param.make(L)).
--W 231 --L 174 --hw 72 --legacy-fit reproduces the stored figures/danish_run2/<run>_stats.json exactly (checked 2026-10-29);
without --legacy-fit the orientation search is used. --hw: hardware mask height above the gripped edge (72 in the 174
chain = 35-37 mm above the REAL gripped edge, which sat at old-frame y = -50 / -55; use 37 with L = 124).
Outputs: figures/danish_run2_<tag>/<run>_{passes.png,fields.npz,stats.json,frame.json}; frame.json holds the 3D sheet
frame (device-1 coordinates) so that two frames can be compared (shift / rotation of the sheet frame).
Usage: python danish_run2_measure_231x124.py --W 231 --L 124 --tag 231x124 [--hw 22] [--band 45] [--run 10-21]"""
import argparse, glob, json, os, sys
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, RT); sys.path.insert(0, os.path.join(RT, "bridge_checks"))
from ew_loop import scan_to_field as S
from plate_refill import fit_known_rectangle
import danish_run2_metrics_param
CALIB = os.path.join(RT, "pc_post_processing", "deliverable_092826", "calibration results")
GATE = np.array([(-300.0, 400.0), (-300.0, 400.0), (300.0, 700.0)])
FLAT_TOL = 3.0
STEP = 2.0


class Chain:
    def __init__(self, W, L, band=45.0, hw_mm=72.0, tag=None, legacy_fit=False):
        self.W, self.L, self.BAND, self.HW, self.legacy = float(W), float(L), float(band), float(hw_mm), legacy_fit
        self.XS = np.arange(-W / 2, W / 2 + STEP, STEP); self.YS = np.arange(-L / 2, L / 2 + STEP, STEP)
        M = danish_run2_metrics_param.make(L); self.metrics, self.ZONE, self.OBS_Y = M["metrics"], M["ZONE"], M["OBS_Y"]
        self.OUT = os.path.join(RT, "figures", f"danish_run2_{tag or f'{W:g}x{L:g}'}"); os.makedirs(self.OUT, exist_ok=True)

    # ---- identical to danish_run2_measure.py except for W / L / OBS_Y / HW ----
    def load(self, run, k):
        xyz, ids = S.merge_capture(os.path.join(run, f"pointcloud_pass{k}"), CALIB, window=(1.0, 1e6), cameras=(1, 2, 3), with_ids=True)
        xyz = xyz.astype(np.float32); g = np.all((xyz > GATE[:, 0]) & (xyz < GATE[:, 1]), axis=1)
        return xyz[g], ids[g]

    @staticmethod
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

    def fit_rect_rot(self, Q, step=2.0, shrink=2.0, angles=np.arange(-45.0, 135.0, 1.0)):
        """Rectangle of the KNOWN size W x L on the plane inliers Q, searching ORIENTATION and position: for every angle
        (about the min-area-rectangle orientation of Q) the occupancy image of Q (step-mm cells holding >= 3 points) is
        correlated with the rectangle kernel (FFT) and the best (angle, offset) by occupied cells inside wins. Unlike
        plate_refill.fit_known_rectangle (orientation fixed by the min-area rectangle of ALL inliers, offset only) this
        is not fooled by coplanar hardware next to the sheet (10-22: 21.7 deg error). Returns (inside mask, frame, info)."""
        from scipy.signal import fftconvolve
        c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c, full_matrices=False); nrm = Vt[2]
        if (np.zeros(3) - c) @ nrm < 0:
            nrm = -nrm
        e1 = Vt[0]; e2 = np.cross(nrm, e1); u = np.column_stack([(Q - c) @ e1, (Q - c) @ e2])
        (rx, ry), _, ang0 = cv2.minAreaRect(u.astype(np.float32)) if cv2 else self._min_area_rect(u)
        kw, kl = int(round((self.W - 2 * shrink) / step)), int(round((self.L - 2 * shrink) / step)); ker = np.ones((kl, kw))
        best = None; curve = []
        for da in angles:
            a = np.radians(ang0 + da); x = (u[:, 0] - rx) * np.cos(a) + (u[:, 1] - ry) * np.sin(a); y = -(u[:, 0] - rx) * np.sin(a) + (u[:, 1] - ry) * np.cos(a)
            x0, y0 = x.min() - step, y.min() - step; ix = ((x - x0) / step).astype(int); iy = ((y - y0) / step).astype(int)
            occ = np.zeros((iy.max() + 2, ix.max() + 2)); np.add.at(occ, (iy, ix), 1); occ = (occ >= 3).astype(float)
            sc = fftconvolve(occ, ker, mode="same"); j = int(np.argmax(sc)); r, col = divmod(j, sc.shape[1]); curve.append(float(sc.flat[j]))
            if best is None or sc.flat[j] > best[0]:
                best = (float(sc.flat[j]), float(da), a, x0 + (col + 0.5) * step, y0 + (r + 0.5) * step)
        score, da, a, ox, oy = best
        x = (u[:, 0] - rx) * np.cos(a) + (u[:, 1] - ry) * np.sin(a); y = -(u[:, 0] - rx) * np.sin(a) + (u[:, 1] - ry) * np.cos(a)
        inside = (np.abs(x - ox) <= self.W / 2 - shrink) & (np.abs(y - oy) <= self.L / 2 - shrink)
        ex = np.cos(a) * e1 + np.sin(a) * e2; ey = np.cross(nrm, ex); c3 = c + (rx + ox * np.cos(a) - oy * np.sin(a)) * e1 + (ry + ox * np.sin(a) + oy * np.cos(a)) * e2
        F = dict(c=c3, ex=ex, ey=ey, n=nrm, angle_vs_minarearect_deg=float(da), rect_wh=(self.W - 2 * shrink, self.L - 2 * shrink))
        return inside, F, dict(score_cells=int(round(score)), angle_vs_minarearect_deg=da, offset_mm=[round(ox, 1), round(oy, 1)], size_mm=[self.W, self.L],
                               score_curve_deg=[float(v) for v in angles[::5]], score_curve=[round(v) for v in curve[::5]])

    def find_sheet(self, P):
        best = None
        for cnt, a, n in self.ransac_planes(P):
            d = (P - a) @ n; m = np.abs(d) < 2.5; Q = P[m]
            if len(Q) < 15000 or len(Q) > 400000:
                continue
            if self.legacy:
                inside, rec = fit_known_rectangle(Q, self.W, self.L); F = self.frame_from_flat(Q[inside]); rec = {**rec, "angle_vs_minarearect_deg": None, "score_curve_deg": [], "score_curve": []}
            else:
                inside, F, rec = self.fit_rect_rot(Q)
            frac = inside.mean(); score = rec["score_cells"] * frac
            if best is None or score > best[0]:
                best = (score, Q[inside], n, rec, len(Q), frac, F)
        return best

    @staticmethod
    def _min_area_rect(q):
        from scipy.spatial import ConvexHull
        H = q[ConvexHull(q).vertices]; E = np.roll(H, -1, axis=0) - H; best = None
        for ex, ey in E / np.linalg.norm(E, axis=1)[:, None]:
            u = H @ np.array([ex, ey]); v = H @ np.array([-ey, ex]); area = np.ptp(u) * np.ptp(v)
            if best is None or area < best[0]:
                best = (area, ex, ey, u, v)
        _, ex, ey, u, v = best; cu, cv = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
        return ((cu * ex - cv * ey, cu * ey + cv * ex), (float(np.ptp(u)), float(np.ptp(v))), float(np.degrees(np.arctan2(ey, ex))))

    def frame_from_flat(self, Q):
        c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c, full_matrices=False); nrm = Vt[2]
        if (np.zeros(3) - c) @ nrm < 0:
            nrm = -nrm
        e1 = Vt[0]; e2 = np.cross(nrm, e1); q = np.column_stack([(Q - c) @ e1, (Q - c) @ e2])
        (rx, ry), (rw, rh), ang = cv2.minAreaRect(q.astype(np.float32)) if cv2 else self._min_area_rect(q); a = np.radians(ang)
        if rw < rh:
            a += np.pi / 2
        ex = np.cos(a) * e1 + np.sin(a) * e2; ey = np.cross(nrm, ex); c = c + rx * e1 + ry * e2
        return dict(c=c, ex=ex, ey=ey, n=nrm, rect_wh=(float(rw), float(rh)))

    @staticmethod
    def to_sheet(P, F):
        d = P - F["c"]; return np.column_stack([d @ F["ex"], d @ F["ey"], d @ F["n"]])

    @staticmethod
    def despeckle(Z, size=5, tol=6.0):
        from scipy.ndimage import generic_filter
        med = generic_filter(Z, np.nanmedian, size=size, mode="constant", cval=np.nan)
        bad = np.isfinite(Z) & np.isfinite(med) & (np.abs(Z - med) > tol)
        out = Z.copy(); out[bad] = np.nan; return out, int(bad.sum())

    def grid(self, xy, h, min_pts=3, fallback_cells=3):
        XS, YS = self.XS, self.YS
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

    def prepare(self, run):
        W, L, OBS_Y = self.W, self.L, self.OBS_Y
        P0, I0 = self.load(run, 0); score, Q, n, rec, nin, frac, F = self.find_sheet(P0)
        s0 = self.to_sheet(P0, F); beyond = {}
        for side, sgn in (("y-", -1), ("y+", 1)):
            m = (np.abs(s0[:, 2]) < 25) & (np.abs(s0[:, 0]) < W / 2) & (sgn * s0[:, 1] > L / 2 + 5) & (sgn * s0[:, 1] < L / 2 + 60)
            beyond[side] = int(m.sum())
        if beyond["y+"] > beyond["y-"]:
            F["ex"], F["ey"] = -F["ex"], -F["ey"]; s0 = self.to_sheet(P0, F)
        on = (np.abs(s0[:, 2]) < 12) & (np.abs(s0[:, 0]) < W / 2 - 15) & (s0[:, 1] > OBS_Y + 12) & (s0[:, 1] < L / 2 - 5)
        corr, raw = {}, {}
        for cam in (1, 2, 3):
            mc = on & (I0 == cam)
            if mc.sum() > 300:
                A = np.column_stack([s0[mc, 0], s0[mc, 1], np.ones(mc.sum())]); corr[cam] = np.linalg.lstsq(A, s0[mc, 2], rcond=None)[0]
            elif mc.sum() > 50:
                corr[cam] = np.array([0.0, 0.0, float(np.median(s0[mc, 2]))])
            if mc.sum() > 50:
                raw[cam] = round(float(np.median(s0[mc, 2])), 2)
        # how well the flat plane is populated along y: fraction of occupied 2 mm rows inside the rectangle (sheet-length check)
        inside = (np.abs(s0[:, 2]) < FLAT_TOL) & (np.abs(s0[:, 0]) < W / 2) & (np.abs(s0[:, 1]) < L / 2)
        info = dict(W=W, L=L, sheet_plane_inliers=int(nin), rect_score_cells=int(rec["score_cells"]), inside_fraction=round(float(frac), 3),
                    rect_offset_mm=rec["offset_mm"], rect_size_mm=rec["size_mm"], rect_angle_vs_minarearect_deg=rec["angle_vs_minarearect_deg"],
                    rect_score_curve=dict(zip(rec["score_curve_deg"], rec["score_curve"])),
                    beyond_edge_points=beyond, flat_points_inside_rect=int(inside.sum()),
                    camera_flat_median_before_mm=raw, camera_flat_offset_mm={c: round(float(v[2]), 2) for c, v in corr.items()},
                    camera_flat_tilt_deg={c: [round(float(np.degrees(v[0])), 3), round(float(np.degrees(v[1])), 3)] for c, v in corr.items()})
        return P0, I0, F, corr, info

    def field_points(self, P, I, F, corr):
        s = self.to_sheet(P, F); h = s[:, 2].copy()
        for cam, v in corr.items():
            mc = I == cam; h[mc] -= v[0] * s[mc, 0] + v[1] * s[mc, 1] + v[2]
        m = (np.abs(h) < self.BAND) & (np.abs(s[:, 0]) < self.W / 2 + 6) & (np.abs(s[:, 1]) < self.L / 2 + 6)
        return s[:, :2], h, m

    def measure(self, run):
        W, L, XS, YS, ZONE = self.W, self.L, self.XS, self.YS, self.ZONE
        name = os.path.basename(run); print("===", name, f"W {W:g} L {L:g} hw {self.HW:g} band {self.BAND:g}", flush=True)
        P0, I0, F, corr, info = self.prepare(run); print(info, flush=True)
        json.dump({k: (F[k].tolist() if isinstance(F[k], np.ndarray) else F[k]) for k in F}, open(os.path.join(self.OUT, f"{name}_frame.json"), "w"), indent=1)
        xy0, h0, m0 = self.field_points(P0, I0, F, corr); Z0 = self.grid(xy0[m0], h0[m0]); sheet = np.isfinite(Z0) & (np.abs(Z0) < FLAT_TOL)
        GX, GY = np.meshgrid(XS, YS); sheet &= ~((np.abs(GX) > W / 2 - 14) & (GY < -L / 2 + self.HW))
        # occupancy of the flat plane along y (is the rectangle really filled over its whole length?)
        rows = np.isfinite(Z0) & (np.abs(Z0) < FLAT_TOL); info["flat_row_fill"] = [round(float(v), 2) for v in rows.mean(1)[::5]]
        fields, stats, planed = {}, {}, {}
        for k in range(7):
            if k == 0:
                Z, nb = Z0, int(m0.sum())
            else:
                xy, h, m = self.field_points(*self.load(run, k), F, corr); Z, nb = self.grid(xy[m], h[m]), int(m.sum())
            Z, nbad = self.despeckle(np.where(sheet, Z, np.nan)); fields[k] = Z
            mt, Zp = self.metrics(XS, YS, Z); planed[k] = Zp
            v = Z[np.isfinite(Z)]
            stats[k] = dict(band_points=nb, despeckled_cells=nbad, coverage=round(float(np.isfinite(Z).mean()), 3), h_p0p5_p50_p99p5=[round(float(np.percentile(v, p)), 2) for p in (0.5, 50, 99.5)], **mt)
            print(k, {kk: vv for kk, vv in stats[k].items() if not kk.startswith("zone_k")}, flush=True)
        for k in range(6):
            d = fields[k + 1] - fields[k]; mm = np.isfinite(d); dv = d[mm]
            stats[f"pass{k+1}-pass{k}"] = dict(cells=int(mm.sum()), p5_p50_p95_mm=[round(float(np.percentile(dv, p)), 2) for p in (5, 50, 95)])
        np.savez_compressed(os.path.join(self.OUT, f"{name}_fields.npz"), xs=XS, ys=YS, **{f"pass{k}": fields[k] for k in range(7)})
        json.dump({"info": info, "stats": stats}, open(os.path.join(self.OUT, f"{name}_stats.json"), "w"), indent=1)
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
        axs[2, 0].set_title("profile across x at y = 0", fontsize=8); axs[2, 1].set_title(f"profile along y at x = 0 (gripped edge {-L/2:g}, free edge {L/2:g})", fontsize=8)
        for ax in axs[2, :2]:
            ax.legend(fontsize=7); ax.grid(color="#eee"); ax.set_ylabel("h (mm)")
        ks = list(range(1, 7))
        for key, mk in (("sag_x_200", "o-"), ("zone_sag_x_105", "^-"), ("zone_sag_y_100", "v-"), ("amp_plane_removed", "s-"), ("lift_far_minus_near", "d-")):
            axs[2, 2].plot(ks, [stats[k].get(key, np.nan) for k in ks], mk, label=key)
        axs[2, 2].plot(ks, [stats[k].get("bend_angle_y_deg", np.nan) for k in ks], "x--", label="bend angle along y (deg)")
        axs[2, 2].set_xlabel("pass"); axs[2, 2].legend(fontsize=7); axs[2, 2].grid(color="#eee"); axs[2, 2].set_title("rigid-invariant shape metrics per pass", fontsize=8)
        axs[2, 3].axis("off"); axs[2, 3].text(0, 0.5, json.dumps({k: v for k, v in info.items() if k not in ("flat_row_fill", "rect_score_curve")}, indent=1), fontsize=7, va="center", family="monospace")
        for ax in axs[2, 4:]:
            ax.axis("off")
        fig.suptitle(f"{name}: {W:g} x {L:g} sheet (rectangle fitted on the flat scan), flat scan + 6 passes; h relative to the flat sheet plane, per-camera flat correction, hardware masked", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(self.OUT, f"{name}_passes.png"), dpi=70); plt.close(fig); print("->", os.path.join(self.OUT, f"{name}_passes.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--W", type=float, default=231.0); ap.add_argument("--L", type=float, default=124.0)
    ap.add_argument("--band", type=float, default=45.0); ap.add_argument("--hw", type=float, default=72.0, help="hardware mask: corner cells below gripped edge + hw mm")
    ap.add_argument("--tag", default=None); ap.add_argument("--run", default=None); ap.add_argument("--legacy-fit", action="store_true"); a = ap.parse_args()
    ch = Chain(a.W, a.L, a.band, a.hw, a.tag, a.legacy_fit)
    runs = sorted(glob.glob(os.path.join(RT, "share", "Danish_Run_2", "2026-*")))
    for run in (runs if not a.run else [r for r in runs if a.run in r]):
        ch.measure(run)
