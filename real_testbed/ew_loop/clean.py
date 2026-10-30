"""Post-processing of an aligned measured height field (2026-10-30, agreed with the user). The raw field (cropped,
plane-removed, the measurement chain's own product) is the record and the calibration data; the CLEANED field is
what the planner sees and what the controller measures, so both use the same field and the same span:

  1. despeckle: drop cells further than DESPECKLE_MM from their 5x5 NaN-median (glare edges, stray points; the
     chain's own despeckle keeps everything within 6 mm). A wheel-formed surface never deviates that much within 10 mm.
  2. fill interior holes (not touching the side / top edges, the gripper band or the hardware corners) by a smooth
     least-squares (biharmonic-like) continuation of the measured surroundings (glare dropouts, speckle gaps)
  3. Gaussian smoothing, sigma SIGMA_MM, NaN-aware (normalized convolution: data and weights blurred separately, so
     holes do not spread; a missing cell is filled only where >= 70 % of the kernel weight has data)
  4. clip to the P_LO / P_HI percentiles: the image colours and the z range in the prompt are then the field's robust
     range (state.span with robust=True is the same percentile span), not set by single cells.

Measured on sheet 1 (2026-10-29): residual noise 0.56-0.65 mm rms with a ~4 mm correlation length; the 200 mm sagitta
changes by <= 0.2 mm through this cleaning, the robust span drops by 2-3 mm (the noise floor: a flat sheet 4.6 -> 2.3 mm)."""
import numpy as np
import scipy.ndimage as nd

from .frames import HeightField

DESPECKLE_MM, SIGMA_MM, P_LO, P_HI = 2.0, 4.0, 0.5, 99.5
FILL_EDGE_MM = 8.0          # holes within this of a side edge are not filled (sheet border), nor holes reaching the crop band


def inpaint_biharmonic(z, fill):
    """values for the cells in `fill` such that the discrete Laplacian is as small as possible (least squares) over
    them and their 1-cell ring, the measured cells fixed: a smooth (biharmonic-like) continuation of the surroundings.
    scipy only (the robot PC has no scikit-image)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr
    ny, nx = z.shape
    idx = -np.ones(z.shape, int); ys, xs = np.nonzero(fill); idx[ys, xs] = np.arange(len(ys))
    known = np.isfinite(z) & ~fill
    rows = nd.binary_dilation(fill, iterations=1)
    rr, cc, vv, b = [], [], [], []
    k = 0
    for y, x in zip(*np.nonzero(rows)):
        if y == 0 or x == 0 or y == ny - 1 or x == nx - 1:
            continue
        st = [(y, x, 4.0), (y - 1, x, -1.0), (y + 1, x, -1.0), (y, x - 1, -1.0), (y, x + 1, -1.0)]
        if not all(fill[a, c] or known[a, c] for a, c, _ in st):
            continue                                   # the stencil touches a cell that is neither data nor being filled
        rhs = 0.0
        for a, c, w in st:
            if fill[a, c]:
                rr.append(k); cc.append(idx[a, c]); vv.append(w)
            else:
                rhs -= w * z[a, c]
        b.append(rhs); k += 1
    out = z.copy()
    if k == 0:
        return out
    A = sp.csr_matrix((vv, (rr, cc)), shape=(k, len(ys)))
    sol = lsqr(A, np.array(b), atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    out[ys, xs] = sol
    return out


def robust_clip(fld, note="robust range"):
    """the field clipped to its P_LO..P_HI percentiles (used for the target too, so both images follow one rule)"""
    z = fld.z_mm.astype(float)
    if not np.isfinite(z).any():
        return fld
    lo, hi = np.nanpercentile(z, [P_LO, P_HI])
    out = np.where(np.isfinite(z), np.clip(z, lo, hi), np.nan).astype(np.float32)
    return HeightField(fld.x_mm, fld.y_mm, out, {**fld.meta, "clipped_to": [round(float(lo), 2), round(float(hi), 2)], "clip": note})


def clean_field(fld, width_mm=231.0, crop_y=-67.0, y_top=87.0):
    """raw HeightField (cropped, plane removed) -> cleaned HeightField; meta["cleaning"] says what was done.
    width_mm / crop_y / y_top: the sheet width, the crop line and the free edge in the sheet frame (holes touching them
    are borders, not dropouts, and are not filled)."""
    z = fld.z_mm.astype(float); x, y = fld.x_mm.astype(float), fld.y_mm.astype(float)
    step = float(abs(x[1] - x[0])) if len(x) > 1 else 2.0
    W, ylo, yhi = float(width_mm), float(crop_y), float(y_top)
    med = nd.generic_filter(z, np.nanmedian, size=5, mode="constant", cval=np.nan)
    out = np.isfinite(z) & np.isfinite(med) & (np.abs(z - med) > DESPECKLE_MM)
    zd = np.where(out, np.nan, z)
    X, Y = np.meshgrid(x, y)
    nan = ~np.isfinite(zd); lab, n = nd.label(nan); fill = np.zeros_like(nan)
    for k in range(1, n + 1):
        c = lab == k
        if X[c].min() > -W / 2 + FILL_EDGE_MM and X[c].max() < W / 2 - FILL_EDGE_MM and Y[c].min() > ylo + 4 and Y[c].max() < yhi - 4:
            fill |= c
    if fill.any():
        zd = inpaint_biharmonic(zd, fill)
    s = SIGMA_MM / step; v = np.isfinite(zd)
    num = nd.gaussian_filter(np.where(v, zd, 0.0), s, mode="constant"); den = nd.gaussian_filter(v.astype(float), s, mode="constant")
    zs = np.where(v | (den >= 0.7), num / np.maximum(den, 1e-6), np.nan)
    zs[Y < ylo] = np.nan                                            # never put data back into the cropped band
    sm = HeightField(fld.x_mm, fld.y_mm, zs.astype(np.float32), dict(fld.meta))
    res = z - zs
    cl = robust_clip(sm, "cleaned field clipped to its robust range")
    cl.meta["cleaning"] = {"despeckle_mm": DESPECKLE_MM, "despeckled_cells": int(out.sum()), "inpainted_cells": int(fill.sum()),
                           "sigma_mm": SIGMA_MM, "percentiles": [P_LO, P_HI], "removed_rms_mm": round(float(np.nanstd(res)), 2),
                           "raw_zrange": [round(float(np.nanmin(z)), 2), round(float(np.nanmax(z)), 2)] if np.isfinite(z).any() else None}
    return cl
