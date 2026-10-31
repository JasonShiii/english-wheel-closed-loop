"""MPC training pairs for the real sheet (2026-10-06).
From each trajectory (record.json + cycles.npz, clamped states) take cuts (s, t) with max |z_t - z_s| >= TAU
(stratified by horizon, capped), render CURRENT and TARGET the way deployment renders them
(frames.field_from_points -> mask_clamp_band -> render.field_png: sheet centred in the 300 mm frame, band and
every unseen cell white, nothing interpolated across gaps) and write messages JSONL for train_mpc.py.
Robustness (training side only, the CURRENT image; the target stays clean):
  --aug  : random edge erosion (0-2 cells), 0-3 holes, an occlusion strip next to the wheel line (p 0.3),
           Gaussian noise sigma U(0, 0.4) mm (cell-level noise after the grid median), grid shift +-2 mm  (what real captures do to the field)
  --fake : with probability p_fake the current field comes from the fake camera rig (ew_loop/synthetic)
           through the real scan_to_field pipeline (0.8 mm point noise, gripper bar, flying pixels; the upper
           wheel only when cfg.SCAN_POSITION is False)
  --aug-real (2026-10-30): degrade-then-clean, the deployed loop's input rule: CURRENT = sim field -> degrade_real (correlated
           noise, outliers, glare dropouts, pinholes, lower corners, per-camera bias + waviness, fitted to sheet 1) -> plane
           removal -> ew_loop.clean.clean_field -> image + prompt z range of the cleaned field; TARGET clipped to its own
           p0.5..p99.5 (ew_loop.clean.robust_clip), otherwise clean. --match-cuts <pairs dir> reproduces another pairs dir's
           (source, cut) rows and split (an evaluation variant of the same samples). Without these flags nothing changes.
Schema v7 (2026-10-30, data of gen_realsheet.py; chosen by the dataset's manifest "prompt_schema"): the v5facts prompt with
the zigzag orientation as rot_deg (integer degrees, CCW in the sheet frame; 0 / 90 = the old rot) and the trajectory's growth
factor g in the REQUEST (and explained in the PROCESS FACTS); the label's zigzag is {"rot_deg": int, "N": int}. The sheet /
dose of the prompt follow the dataset's profile (cfg.apply_manifest), so old datasets render and prompt exactly as before.
Usage: python pairs.py --data data_mini --out pairs_mini --cap 30 --val-frac 0.2 --aug --fake 0.25"""
import argparse, glob, json, os
from multiprocessing import Pool
import numpy as np
import cfg
from ew_loop import frames as F
import render as R          # the ONE renderer of the new model (no interpolation across gaps)
from common import slice_actions
HERE = os.path.dirname(os.path.abspath(__file__))
HORIZON_BINS = [(1, 2), (3, 5), (6, 9), (10, 99)]

PROMPT_SYSTEM = """You are a process planner for incremental sheet-metal forming with an English wheel. \
The sheet is {W:.0f} x {L:.0f} mm, {t:.1f} mm thick, held by a gripper along its bottom edge (y = {ylo:.0f} mm). \
Coordinates: the sheet frame has its origin at the sheet centre; x runs across the feed direction (to the right in \
the images, side edges at x = {xlo_s:.1f} and {xhi_s:.1f} mm), y runs along the feed (up in the images, gripped edge \
at y = {ylo:.0f}, free edge at y = {yhi_s:.0f} mm); z is the height above the sheet's own best-fit plane, positive \
toward the side the upper wheel touches. All lengths are in mm. \
Three regions matter: the strip below y = {yc:.0f} mm is hidden by the gripper and is never shown (white); the wheel \
can only reach the rectangle x in [{xlo:.1f}, {xhi:.1f}] mm, y in [{yzlo:.0f}, {yhi:.0f}] mm, so every working area \
must lie inside it; the margins around that rectangle ({mside:.1f} mm at each side, {mtop:.0f} mm at the top) deform \
passively and are shown but never wheeled.

INPUTS (given with every request)
1. Image 1 - CURRENT shape: top view of the sheet height field (x to the right, y up, in mm). \
The image shows the part of the sheet above the hidden strip plus a {m:.0f} mm border: x in [{fx0:.1f}, {fx1:.1f}] mm, \
y in [{fy0:.0f}, {fy1:.0f}] mm, so the bottom edge of the image is {m:.0f} mm below the hidden strip's top; pure white \
pixels are "no data": outside the sheet and parts the cameras could not see. Colors elsewhere are normalized to THIS \
image's own min/max, so read the shape from the image and the scale from the text (the z range of each image is \
given in the request).
2. Image 2 - TARGET shape: same convention.
3. HISTORY - ordered list of wheeling actions already executed (may be empty). \
Each action has three parts. "wa" is the rectangular working area: center_xy = its center [x, y] in mm, \
size_wh = its width and height in mm; it always lies inside the reachable part of the sheet. "zigzag" is \
the toolpath applied inside that area: rot is its orientation (0: strips run along y, stacked along x; \
90: strips run along x, stacked along y), N is the number of strips (density). "hits_done" = how many \
passes of this zigzag were already applied. Because of work hardening, repeated passes over \
already-wheeled regions produce progressively smaller deformation.
{facts}
TASK
Predict the remaining action sequence that transforms CURRENT into TARGET. Answer with ONLY a JSON \
object, no other text:
{{"plan": [{{"wa": {{"center_xy": [x, y], "size_wh": [w, h]}}, "zigzag": {{"rot": 0 or 90, "N": int}}, \
"hits": int}}, ...]}}
Entries are in execution order; hits >= 1. If the last history action is not finished, the first plan \
entry may continue the same wa with the remaining hits."""

_ZZ_V5 = """the toolpath applied inside that area: rot is its orientation (0: strips run along y, stacked along x; \
90: strips run along x, stacked along y), N is the number of strips (density)."""
_ZZ_V7 = """the toolpath applied inside that area: rot_deg is its orientation in integer degrees, counter-clockwise in \
the sheet frame as seen in the images (0: strips run along y, stacked along x; 90: strips run along x, stacked along y; \
any other angle rotates the strips and the working area together about the area centre, e.g. +30 tilts the strips so \
that their upper end moves toward -x), N is the number of strips (density)."""
_JSON_V5, _JSON_V7 = '"zigzag": {{"rot": 0 or 90, "N": int}}', '"zigzag": {{"rot_deg": int, "N": int}}'
assert PROMPT_SYSTEM.count(_ZZ_V5) == 1 and PROMPT_SYSTEM.count(_JSON_V5) == 1
PROMPT_SYSTEM_V7 = PROMPT_SYSTEM.replace(_ZZ_V5, _ZZ_V7).replace(_JSON_V5, _JSON_V7)      # schema v7: continuous angle + growth factor
# Schema v8 (2026-10-30, the final scheme DATASET_D_PLAN_2026-10-30.md): the tilt is a property of the working area ("wa": angle),
# decided with the region before the toolpath; the zigzag keeps rot 0 / 90 (how it lies in the area) and N. The simulator /
# driver rotation is rot + angle (common.read_entry). No growth factor (nominal growth only).
_WA_V5 = "size_wh = its width and height in mm; it always lies inside the reachable part of the sheet."
_WA_V8 = ("size_wh = its width and height in mm before rotation, angle = its rotation about its center in integer degrees, "
          "counter-clockwise in the sheet frame as seen in the images (0: sides along x and y; a positive angle turns the area "
          "so that its upper end moves toward -x; allowed values -45, -30, -15, 0, 15, 30, 45); the rotated area always lies "
          "inside the reachable part of the sheet.")
_ZZ_V8 = """the toolpath applied inside that area: rot is how the zigzag lies in the area (0: strips run along the area's \
height side, stacked along its width; 90: strips run along its width side, stacked along its height), N is the number of \
strips (density)."""
_JSONWA_V5, _JSONWA_V8 = '"size_wh": [w, h]}}', '"size_wh": [w, h], "angle": int}}'
assert PROMPT_SYSTEM.count(_WA_V5) == 1 and PROMPT_SYSTEM.count(_JSONWA_V5) == 1
PROMPT_SYSTEM_V8 = PROMPT_SYSTEM.replace(_WA_V5, _WA_V8).replace(_ZZ_V5, _ZZ_V8).replace(_JSONWA_V5, _JSONWA_V8)
FORCE_FULL_CUT = False   # schema v8: every trajectory keeps its flat -> final cut (0, T) among the capped training cuts

# The per-request part. Our model is trained on PROMPT_SYSTEM + "\n\n" + PROMPT_USER as one user message; the closed
# models get PROMPT_SYSTEM once as the system message and PROMPT_USER (+ the two images) per example / query turn, so
# the invariant text is not repeated k times in a k-shot request. split_prompt() recovers the two parts.
USER_MARK = "\n\nREQUEST\n"
PROMPT_USER = """REQUEST
Image 1 (CURRENT) z range = [{cz0}, {cz1}] mm. Image 2 (TARGET) z range = [{tz0}, {tz1}] mm.
HISTORY:
{history}"""
PROMPT_USER_V7 = """REQUEST
Image 1 (CURRENT) z range = [{cz0}, {cz1}] mm. Image 2 (TARGET) z range = [{tz0}, {tz1}] mm.
Growth factor of this sheet: g = {g:.2f} relative to the nominal {gtop:g} (see PROCESS FACTS).
HISTORY:
{history}"""
SCHEMA = "v5"          # set by main() from the dataset's manifest ("prompt_schema": "v7") / facts.json; v5 and v5facts otherwise
GROWTH_LINE = True     # schema v7: the request states the trajectory's growth factor. main() sets it False for data WITHOUT growth
                       # randomisation (manifest growth_factor_range [1, 1] or p_nominal 1, e.g. data_231x174_t1.0_g6.5e-4_nominal,
                       # the main line of 2026-10-30): then the v7 request is PROMPT_USER (no g line), everything else stays v7.


def growth_randomised(man):
    """does this dataset randomise the growth factor (and therefore state it in the prompt)? v7 data without the fields = yes"""
    r = man.get("growth_factor_range")
    return not ((r is not None and abs(float(r[1]) - float(r[0])) < 1e-12) or float(man.get("p_nominal", 0.0)) >= 1.0)


# Lean prompt (schema v6lean, 2026-10-27): for the fine-tuned model only. The constant process text of v5facts carries no
# label information in SFT (identical in every sample), so only the request is kept, the amplitude is given as one
# number per image (peak-to-valley) instead of [lo, hi], and the label starts with that number read back
# ({"target_pv_mm": ..., "plan": [...]}) so the plan tokens follow an explicit read-out of the dose cue.
PROMPT_LEAN = """Sheet {W:.0f} x {L:.0f} x {t:.1f} mm, gripped along y = {ylo:.0f}; origin at the sheet centre, x across the feed \
(right in the images), y along the feed (up), z = height above the sheet's best-fit plane. Image 1 = CURRENT, image 2 = \
TARGET height field (white = no data, colours scaled per image). Working areas must lie inside x in [{xlo:.1f}, {xhi:.1f}], \
y in [{yzlo:.0f}, {yhi:.0f}] mm.
CURRENT peak-to-valley {cpv:.1f} mm. TARGET peak-to-valley {tpv:.1f} mm.
HISTORY: {history}
Answer with ONLY a JSON object: {{"target_pv_mm": number, "plan": [{{"wa": {{"center_xy": [x, y], "size_wh": [w, h]}}, \
"zigzag": {{"rot": 0 or 90, "N": int}}, "hits": int}}, ...]}}"""


def build_prompt_lean(cz, tz, history):
    s = cfg.SHEET; x_lo, x_hi, y_lo, y_hi = cfg.reachable_rect()
    return PROMPT_LEAN.format(W=s.width_mm, L=s.length_mm, t=s.thickness_mm, ylo=-s.length_mm / 2, xlo=x_lo, xhi=x_hi,
                              yzlo=y_lo, yhi=y_hi, cpv=float(cz[1]) - float(cz[0]), tpv=float(tz[1]) - float(tz[0]),
                              history=json.dumps(history) if history else "[] (the sheet is untouched)")


def label_lean(tz, plan):
    """the assistant text of schema v6lean: the target peak-to-valley read back, then the plan"""
    return json.dumps({"target_pv_mm": round(float(tz[1]) - float(tz[0]), 1), "plan": plan})


def split_prompt(prompt):
    """training prompt -> (system text, per-request text)"""
    i = prompt.rindex(USER_MARK)
    return prompt[:i], prompt[i + 2:]


FACTS_TEXT = None      # set by main() from <data>/facts.json (facts.py); None = schema v5 without the process facts


def facts_block():
    return "\nPROCESS FACTS\n" + "\n".join("- " + f for f in FACTS_TEXT) + "\n" if FACTS_TEXT else ""


def system_prompt():
    s = cfg.SHEET; x_lo, x_hi, y_lo, y_hi = cfg.reachable_rect(); fr = R.frame()
    return {"v7": PROMPT_SYSTEM_V7, "v8": PROMPT_SYSTEM_V8}.get(SCHEMA, PROMPT_SYSTEM).format(W=s.width_mm, L=s.length_mm, t=s.thickness_mm, ylo=-s.length_mm / 2, yc=cfg.crop_y(),
                                xlo_s=-s.width_mm / 2, xhi_s=s.width_mm / 2, yhi_s=s.length_mm / 2,
                                xlo=x_lo, xhi=x_hi, yzlo=y_lo, yhi=y_hi, mside=s.margin_side_mm, mtop=cfg.MARGIN_TOP_MM,
                                m=R.MARGIN_MM, fx0=fr[0], fx1=fr[1], fy0=fr[2], fy1=fr[3], facts=facts_block())


def user_prompt(cz, tz, history, g=None):
    """g: the trajectory's growth factor (schema v7 with GROWTH_LINE only; required there)"""
    hist = json.dumps(history, indent=1) if history else "(empty - the sheet is untouched)"
    if SCHEMA == "v7" and GROWTH_LINE:
        assert g is not None, "schema v7 needs the growth factor"
        return PROMPT_USER_V7.format(cz0=round(float(cz[0]), 2), cz1=round(float(cz[1]), 2), tz0=round(float(tz[0]), 2),
                                     tz1=round(float(tz[1]), 2), g=float(g), gtop=cfg.DOSE["gtop"], history=hist)
    return PROMPT_USER.format(cz0=round(float(cz[0]), 2), cz1=round(float(cz[1]), 2),
                              tz0=round(float(tz[0]), 2), tz1=round(float(tz[1]), 2), history=hist)


def build_prompt(cz, tz, history, g=None):
    """the text our model is trained on: system part + per-request part in ONE user message"""
    return system_prompt() + "\n\n" + user_prompt(cz, tz, history, g)


def label_schema():
    """the entry schema of common.slice_actions / entry_desc for the current prompt schema"""
    return SCHEMA if SCHEMA in ("v7", "v8") else "v2"


def mask_crop(field):
    """white below the crop line (cfg.CROP_MM from the gripped edge); the same rule is applied to measured fields."""
    z = field.z_mm.copy(); z[field.y_mm < cfg.crop_y(), :] = np.nan
    return F.HeightField(field.x_mm, field.y_mm, z, {**field.meta, "crop_mm": cfg.CROP_MM})


def plane_removed(field):
    """heights relative to the best-fit plane of the cells with data (the observed region): the same rule is applied to
    real captures, where the sheet also moves rigidly in the compliant gripper; the free-plate simulation itself has
    no rigid motion, so this changes its fields by at most ~1 mm."""
    z = field.z_mm; m = np.isfinite(z); X, Y = np.meshgrid(field.x_mm, field.y_mm)
    A = np.column_stack([X[m], Y[m], np.ones(m.sum())]); cf = np.linalg.lstsq(A, z[m], rcond=None)[0]
    out = (z - (cf[0] * X + cf[1] * Y + cf[2])).astype(np.float32); out[~m] = np.nan
    return F.HeightField(field.x_mm, field.y_mm, out, {**field.meta, "plane_removed": True})


def field_of(P):
    """simulator vertices (mm) -> HeightField in the sheet frame, cropped below the crop line, best-fit plane removed"""
    return plane_removed(mask_crop(F.field_from_points(P[:, :2], P[:, 2], cfg.SHEET, cfg.STEP_MM)))


def augment(fld, rng):
    z = fld.z_mm.copy(); ny, nx = z.shape; valid = np.isfinite(z)
    # edge erosion 0-2 cells
    k = int(rng.integers(0, 3))
    if k:
        from scipy.ndimage import binary_erosion
        valid = binary_erosion(valid, iterations=k); z[~valid] = np.nan
    # holes
    X, Y = np.meshgrid(fld.x_mm, fld.y_mm)
    for _ in range(int(rng.integers(0, 4))):
        cx, cy = rng.uniform(fld.x_mm[0], fld.x_mm[-1]), rng.uniform(cfg.crop_y(), fld.y_mm[-1])
        a, b = rng.uniform(3, 14), rng.uniform(3, 14)
        z[((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 < 1] = np.nan
    # occlusion strip next to the wheel line (the upper wheel hides the sheet next to it)
    if rng.random() < (0.05 if cfg.SCAN_POSITION else 0.3):
        h = rng.uniform(15, 50); x0 = rng.uniform(fld.x_mm[0], fld.x_mm[-1] - 40); w = rng.uniform(40, 120)
        z[(Y >= cfg.SHEET.wheel_line_y) & (Y < cfg.SHEET.wheel_line_y + h) & (X >= x0) & (X < x0 + w)] = np.nan
    # noise
    sig = rng.uniform(0, 0.4); z = z + rng.normal(0, sig, z.shape).astype(np.float32)   # 0.8 mm point noise -> ~0.3 mm per 2 mm cell after the grid median
    # grid shift +-2 mm (the field is resampled on the same grid, so this moves the picture by ~1-2 px)
    dx, dy = rng.uniform(-2, 2), rng.uniform(-2, 2)
    return F.HeightField(fld.x_mm + dx, fld.y_mm + dy, z.astype(np.float32), {**fld.meta, "aug": dict(erode=k, sigma=round(sig, 2), dx=round(dx, 1), dy=round(dy, 1))})


# ---- degrade-then-clean (--aug-real, 2026-10-30) -----------------------------------------------------------------------
# The deployed loop (real_testbed/ew_loop/clean.py, clean_field) cleans every measured field before the planner sees it:
# despeckle (> 2 mm from the 5x5 NaN-median), biharmonic fill of interior holes, 4 mm NaN-aware Gaussian, clip to
# p0.5..p99.5; the CURRENT image and its prompt range come from that cleaned field, the TARGET is clipped to its own
# p0.5..p99.5 (clean.robust_clip). --aug-real reproduces that chain on simulated fields: field -> realistic degradation
# (fitted to sheet 1's repaired raw fields, 2026-10-29/30: DEGRADE_CLEAN_AUG_2026-10-30.md) -> plane removal -> the SAME
# clean_field() -> image + range. Numbers of the fit (9 raw fields, flat + 8 passes):
#   residual around a 4 mm NaN-aware Gaussian (after the 2 mm despeckle) 0.55-0.66 mm rms (median 0.58), its ACF at
#   2 / 4 / 6 / 8 mm 0.44 / 0.10 / -0.07 / -0.10; residual around 10 mm 0.76 mm rms. Fitted noise: 30 % of the variance
#   Gaussian-filtered white noise with kernel sigma 1 mm + 70 % with kernel sigma 3 mm (marginal sigma = rms4 / 0.688;
#   underlying 1/e correlation length ~4.5 mm); 43-174 cells > 2 mm from the 5x5 median (0-17 > 4 mm), about half
#   positive, in small clusters (up to 21 cells), concentrated at the left / top edges; glare dropout up to 60 x 32 mm at the
#   sheet centre with a bright rim (raw up to +12.8 mm); a few 1-4 cell pinholes; the two lower corners missing
#   (~20 x 18 mm, hardware); ragged borders (outer cell line 9-45 % missing); per-camera low-frequency bias 1-2 mm at the left edge (loop vs repaired calibration);
#   flat sheet: raw span 6.2, robust 4.6, cleaned 2.3 mm.
AUGREAL = dict(sigma4=(0.3, 0.8), res4_per_marginal=0.688, short_frac=0.3, k_short_mm=1.0, k_long_mm=3.0,
               outliers=(40, 160), drop_p=(0.55, 0.33, 0.12), drop_w=(12.0, 60.0), drop_h=(8.0, 32.0),
               bias_p=0.7, bias_amp=2.0, wavy_rms=(0.05, 0.3), wavy_mm=(10.0, 25.0), shift_mm=2.0, corner_p=0.8)


def _unit_noise(shape, k_cells, rng):
    import scipy.ndimage as nd
    w = rng.normal(size=shape)
    n = nd.gaussian_filter(w, k_cells, mode="wrap") if k_cells > 0 else w
    return n / n.std()


def _blob(X, Y, cx, cy, a, b, rng, p=None):
    """superellipse |dx/a|^p + |dy/b|^p < 1 with a ragged rim (random radial jitter)"""
    p = rng.uniform(2.0, 4.0) if p is None else p
    r = np.abs((X - cx) / a) ** p + np.abs((Y - cy) / b) ** p
    return r < 1.0 + 0.25 * rng.uniform(-1, 1, X.shape)


def degrade_real(fld, rng):
    """simulated (cropped, plane-removed) field -> a raw field as the real measurement chain delivers it (cropped,
    plane-removed, NaN = no data), fitted to sheet 1 (see the block above / AUGREAL). Order: low-frequency bias + waviness,
    correlated cell noise, sparse outliers (clusters, edge-weighted), glare dropouts with a bright rim, pinholes, the lower
    corners and edge loss, then the best-fit plane is removed again (as the chain does on the raw field)."""
    import scipy.ndimage as nd
    A = AUGREAL; s = cfg.SHEET
    z = fld.z_mm.astype(float).copy(); x, y = fld.x_mm.astype(float), fld.y_mm.astype(float)
    X, Y = np.meshgrid(x, y); step = float(abs(x[1] - x[0])); W2 = s.width_mm / 2; ytop = s.length_mm / 2; yc = cfg.crop_y()
    valid0 = np.isfinite(z); info = {}
    # 1. low-frequency: per-camera bias ramp at a side edge (left more often: sheet 1's camera 2) + sheet waviness
    bias = np.zeros_like(z)
    if rng.random() < A["bias_p"]:
        side = -1 if rng.random() < 0.65 else 1
        reach = rng.uniform(30.0, 90.0); amp = rng.uniform(-A["bias_amp"], A["bias_amp"]); beta = rng.uniform(-0.5, 0.5)
        d = (X + W2) if side < 0 else (W2 - X)
        t = np.clip(1.0 - d / reach, 0.0, 1.0); bias += amp * t * t * (3 - 2 * t) * (1 + beta * Y / ytop)
        info["bias"] = dict(side="left" if side < 0 else "right", amp=round(amp, 2), reach=round(reach))
    wr = rng.uniform(*A["wavy_rms"]); wl = rng.uniform(*A["wavy_mm"])
    bias += wr * _unit_noise(z.shape, wl / step, rng)
    info["wavy"] = dict(rms=round(wr, 2), mm=round(wl))
    # 2. correlated cell noise (sigma4 = its rms around a 4 mm smooth, the measured quantity)
    s4 = rng.uniform(*A["sigma4"]); marg = s4 / A["res4_per_marginal"]
    n = np.sqrt(A["short_frac"]) * _unit_noise(z.shape, A["k_short_mm"] / step, rng) + np.sqrt(1 - A["short_frac"]) * _unit_noise(z.shape, A["k_long_mm"] / step, rng)
    z = z + bias + marg * n / n.std()
    info["sigma4"] = round(s4, 2)
    # 3. sparse outliers: clusters of 1-8 cells, same sign per cluster; 45 % in a 12 mm band at the
    #    left / right / top edges (left weighted), the rest anywhere (|dev| 1.8 + Exp(0.7) mm, clusters <= 8 cells)
    n_target = int(rng.integers(A["outliers"][0], A["outliers"][1] + 1)); vy, vx = np.nonzero(valid0); placed = 0
    edge = valid0 & (((X + W2) < 12) | ((W2 - X) < 12) | ((ytop - Y) < 12))
    ey, ex = np.nonzero(edge); wts = np.where(X[ey, ex] < 0, 2.0, 1.0); wts /= wts.sum()
    while placed < n_target and len(vy):
        if rng.random() < 0.45 and len(ey):
            i = rng.choice(len(ey), p=wts); cy0, cx0 = ey[i], ex[i]
        else:
            i = rng.integers(len(vy)); cy0, cx0 = vy[i], vx[i]
        size = min(int(rng.geometric(0.5)), 8); sign = 1.0 if rng.random() < 0.5 else -1.0; mag = 1.8 + rng.exponential(0.7)
        cells = [(cy0, cx0)]
        for _ in range(size - 1):
            cyy, cxx = cells[rng.integers(len(cells))]; cells.append((int(np.clip(cyy + rng.integers(-1, 2), 0, z.shape[0] - 1)), int(np.clip(cxx + rng.integers(-1, 2), 0, z.shape[1] - 1))))
        for cyy, cxx in set(cells):
            if valid0[cyy, cxx]:
                z[cyy, cxx] += sign * mag * rng.uniform(0.8, 1.2); placed += 1
    info["outliers"] = placed
    # 4. glare dropouts (0-2, up to ~60 x 32 mm, central) with a bright rim; pinholes
    holes = np.zeros(z.shape, bool); k = int(rng.choice(3, p=A["drop_p"]))
    for _ in range(k):
        a, b = rng.uniform(*A["drop_w"]) / 2, rng.uniform(*A["drop_h"]) / 2
        h = _blob(X, Y, rng.uniform(-60, 60), rng.uniform(yc + 15, ytop - 20), a, b, rng)
        holes |= h
        if rng.random() < 0.7:                                      # glare rim: a few bright cells next to the dropout
            rim = nd.binary_dilation(h, iterations=2) & ~h & valid0; ry, rx = np.nonzero(rim)
            if len(ry):
                sel = rng.choice(len(ry), size=min(len(ry), int(rng.integers(3, 16))), replace=False)
                z[ry[sel], rx[sel]] += rng.uniform(2.0, 6.0, len(sel)) + (rng.random(len(sel)) < 0.1) * rng.uniform(0, 4.0, len(sel))
    for _ in range(int(rng.integers(1, 7))):
        holes |= _blob(X, Y, rng.uniform(-W2 + 10, W2 - 10), rng.uniform(yc + 6, ytop - 6), rng.uniform(0.6, 3.0), rng.uniform(0.6, 3.0), rng, p=2.0)
    info["dropouts"] = k
    # 5. edges: the two lower corners (hardware, ~20 x 18 mm, sometimes stepped), ragged borders (the outermost 1-3 lines of
    #    cells at the left / right / top edge missing in runs: sheet 1 cleaned 9-45 % of the outer line, 2-17 % of the
    #    second), an occasional bite at the top edge
    if rng.random() < A["corner_p"]:
        for sgn in (-1, 1):
            if rng.random() < 0.9:
                w, hh = rng.uniform(10, 24), rng.uniform(10, 20)
                holes |= ((sgn * X) > W2 - w) & (Y < yc + hh)
                if rng.random() < 0.5:                                  # a step: a narrower part reaching higher
                    holes |= ((sgn * X) > W2 - w * rng.uniform(0.3, 0.7)) & (Y < yc + hh + rng.uniform(2, 8))
    rows = np.nonzero(y >= yc)[0]
    for side in ("left", "right", "top"):
        p = rng.uniform(0.05, 0.45)
        for depth in range(3):
            if side == "top":
                line = (np.full(nx := len(x), len(y) - 1 - depth), np.arange(nx))
            else:
                line = (rows, np.full(len(rows), depth if side == "left" else len(x) - 1 - depth))
            u = nd.gaussian_filter1d(rng.normal(size=len(line[0])), rng.uniform(0.5, 2.0))
            miss = u > np.quantile(u, 1 - p)
            holes[line[0][miss], line[1][miss]] = True
            p *= rng.uniform(0.1, 0.5) if depth == 0 else rng.uniform(0.0, 0.6)
    if rng.random() < 0.3:
        x0 = rng.uniform(-W2, W2 - 20); holes |= (X > x0) & (X < x0 + rng.uniform(5, 20)) & (Y > ytop - rng.uniform(2, 5))
    z[holes] = np.nan; z[~valid0] = np.nan
    out = plane_removed(F.HeightField(fld.x_mm, fld.y_mm, z.astype(np.float32), {**fld.meta}))
    out.meta["aug_real"] = info
    return out


def clean_like_loop(fld):
    """the deployed cleaning (ew_loop.clean.clean_field, imported read-only) with the loop's geometry arguments"""
    from ew_loop import clean as CL
    return CL.clean_field(fld, width_mm=cfg.SHEET.width_mm, crop_y=cfg.crop_y(), y_top=cfg.SHEET.length_mm / 2)


def clip_like_loop(fld):
    """the loop's target rule: clipped to its own p0.5..p99.5 (ew_loop.clean.robust_clip)"""
    from ew_loop import clean as CL
    return CL.robust_clip(fld, "target clipped to its robust range (the rule of the cleaned current)")


def shifted_field_of(P, rng):
    """field_of with the sheet content shifted by +-AUGREAL['shift_mm'] in x and y on the FIXED grid (registration error of
    the measured sheet frame; the old --aug moved the grid itself)"""
    import scipy.ndimage as nd
    d = rng.uniform(-AUGREAL["shift_mm"], AUGREAL["shift_mm"], 2)
    Q = P.copy(); Q[:, :2] = Q[:, :2] + d
    f = field_of(Q); f.meta["shift_mm"] = [round(float(d[0]), 2), round(float(d[1]), 2)]
    # the measured sheet fills the grid up to its outline; the simulated one does not: formed states contract in-plane by
    # ~0.06 mm, so the outermost grid column and the top row fall outside the mesh hull (white in every sim field), and the
    # shift moves one more line off the grid. Fill every cell above the crop line by its nearest value (ragged borders are
    # then made by degrade_real, as measured).
    lost = (f.y_mm[:, None] >= cfg.crop_y()) & ~np.isfinite(f.z_mm)
    if lost.any():
        iy, ix = nd.distance_transform_edt(~np.isfinite(f.z_mm), return_distances=False, return_indices=True)
        f.z_mm[lost] = f.z_mm[iy[lost], ix[lost]]
    return f


_REF = {}
def fake_capture_field(P, rng):
    """CURRENT through the fake camera rig and the real measurement pipeline."""
    from ew_loop import synthetic as SY, scan_to_field as S
    from scipy.interpolate import CloughTocher2DInterpolator
    sheet = cfg.SHEET; seed = int(rng.integers(0, 10**6))
    key = seed % 8                                            # a few cached references (rig poses)
    if key not in _REF:
        r = SY.FakeRig(sheet, seed=key)
        if cfg.SCAN_POSITION:                                 # sheet moved clear of the upper wheel: no wheel in the capture
            r._occluded = lambda P, C, ax: np.zeros(len(P), bool)
            r._wheel_points = lambda C, ax: np.zeros((0, 3))
        _REF[key] = (r, S.make_reference(r.capture(SY.flat_surface, r.contact_ref), sheet, r.feed_dir_cam(error_deg=10.0)))
    r, ref = _REF[key]
    interp = CloughTocher2DInterpolator(P[:, :2], P[:, 2], fill_value=np.nan)   # C1-smooth surface between the 4 mm mesh vertices
    surf = lambda X, Y: interp(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    contact = np.array([rng.uniform(-40, 40), sheet.wheel_line_y])
    # scan_to_field masks at the wheel line internally (frames.mask_clamp_band); the deployment pipeline will mask at
    # the crop line instead, so emulate that here: swap the mask for the duration of the call.
    orig = F.mask_clamp_band
    F.mask_clamp_band = lambda field, sh: mask_crop(field)
    try:
        fld = S.scan_to_field(r.capture(surf, contact), ref, contact_xy=contact, step=cfg.STEP_MM)
    finally:
        F.mask_clamp_band = orig
    return plane_removed(mask_crop(fld))


def pairs_for(T, Z, cap, rng):
    valid = [(s, t) for s in range(T) for t in range(s + 1, T + 1) if np.nanmax(np.abs(Z[t] - Z[s])) >= cfg.TAU_PAIR_MM]
    if len(valid) <= cap: return valid
    out = []; per_bin = cap // len(HORIZON_BINS)
    for lo, hi in HORIZON_BINS:
        binp = [p for p in valid if lo <= p[1] - p[0] <= hi]
        if binp:
            idx = rng.choice(len(binp), size=min(per_bin, len(binp)), replace=False); out += [binp[i] for i in idx]
    rest = [p for p in valid if p not in out]
    if len(out) < cap and rest:
        idx = rng.choice(len(rest), size=min(cap - len(out), len(rest)), replace=False); out += [rest[i] for i in idx]
    if FORCE_FULL_CUT and (0, T) in valid and (0, T) not in out:      # schema v8: swap one cut of the fullest horizon bin for (0, T)
        fullest = max(HORIZON_BINS, key=lambda b: sum(1 for p in out if b[0] <= p[1] - p[0] <= b[1]))
        drop = [p for p in out if fullest[0] <= p[1] - p[0] <= fullest[1]]
        out.remove(drop[int(rng.integers(len(drop)))]); out.append((0, T))
    return sorted(out)


def process_traj(args):
    tdir, outdir, cap, seed, aug, p_fake = args[:6]
    aug_real = len(args) > 6 and args[6]                      # --aug-real: degrade-then-clean CURRENT, loop-clipped TARGET
    cuts = args[7] if len(args) > 7 else None                 # --match-cuts: {source: [[s, t], ...]} of a reference pairs dir
    rec = json.load(open(os.path.join(tdir, "record.json"))); npz = os.path.join(tdir, "cycles.npz")
    if not os.path.exists(npz): return rec["id"], []
    xyz = np.load(npz)["xyz"].astype(float)                   # (T+1, n, 3) mm, clamped states
    Z = xyz[:, :, 2]
    if aug_real or cuts is not None:                          # new paths: a stable seed (str hash() changes per process)
        import zlib
        rng = np.random.default_rng(seed * 1000003 + zlib.crc32(rec["id"].encode()))
    else:
        rng = np.random.default_rng(seed + hash(rec["id"]) % 10**6)
    tag0 = rec["id"].replace("/", "_"); samples = []
    todo = [tuple(c) for c in cuts.get(rec["id"], [])] if cuts is not None else pairs_for(rec["T"], Z, cap, rng)
    for s, t in todo:
        tag = f"{tag0}__cut{s:02d}-{t:02d}"
        extra = {}
        if aug_real:
            ft = clip_like_loop(field_of(xyz[t]))
            if p_fake > 0 and rng.random() < p_fake:
                try: fc = fake_capture_field(xyz[s], rng); how = "fake_camera_clean"
                except Exception as e: fc = degrade_real(shifted_field_of(xyz[s], rng), rng); how = f"aug_real(fake failed: {type(e).__name__})"
            else:
                fc = degrade_real(shifted_field_of(xyz[s], rng), rng); how = "aug_real"
                extra = {"aug_real": {**fc.meta["aug_real"], "shift_mm": fc.meta.get("shift_mm")}}
            fc = clean_like_loop(fc)
            extra["cleaning"] = {k: fc.meta["cleaning"][k] for k in ("despeckled_cells", "inpainted_cells", "removed_rms_mm")}
        else:
            ft = field_of(xyz[t]); fc = field_of(xyz[s]); how = "clean"
            if p_fake > 0 and rng.random() < p_fake:
                try: fc = fake_capture_field(xyz[s], rng); how = "fake_camera"
                except Exception as e: how = f"clean(fake failed: {type(e).__name__})"
            elif aug:
                fc = augment(fc, rng); how = "aug"
        pc, pt = os.path.join(outdir, "images", f"{tag}_cur.png"), os.path.join(outdir, "images", f"{tag}_tgt.png")
        R.field_png(fc, pc); R.field_png(ft, pt)
        hist, plan = slice_actions(rec["was"], s, t, schema=label_schema())
        prompt = build_prompt(fc.zrange(), ft.zrange(), hist, rec.get("growth_factor"))
        meta = {"source": rec["id"], "category": rec["category"], "cut": [s, t], "current": how, **extra,
                **({k: rec[k] for k in ("subtype", "split") if k in rec})}
        if SCHEMA == "v7": meta["g"] = rec["growth_factor"]
        samples.append({"meta": meta,
                        "messages": [{"role": "user", "content": [{"type": "image", "image": f"images/{tag}_cur.png"},
                                                                  {"type": "image", "image": f"images/{tag}_tgt.png"},
                                                                  {"type": "text", "text": prompt}]},
                                     {"role": "assistant", "content": [{"type": "text", "text": json.dumps({"plan": plan})}]}]})
    return rec["id"], samples


def main(a):
    datadir, outdir = os.path.join(HERE, a.data), os.path.join(HERE, a.out)
    os.makedirs(os.path.join(outdir, "images"), exist_ok=True)
    man = json.load(open(os.path.join(datadir, "manifest.json"))); rng = np.random.default_rng(a.seed)
    prof = cfg.apply_manifest(man)                            # sheet / dose of THIS dataset (before the pool forks)
    global FACTS_TEXT, SCHEMA, GROWTH_LINE
    fp = os.path.join(datadir, "facts.json")
    FACTS_TEXT = json.load(open(fp))["facts"] if os.path.exists(fp) else None
    if man.get("prompt_schema") == "v8":
        assert FACTS_TEXT, "schema v8 needs <data>/facts.json (facts.py)"
        global FORCE_FULL_CUT
        SCHEMA = schema = "v8"; GROWTH_LINE = False; FORCE_FULL_CUT = True
        assert not growth_randomised(man), "schema v8 data has nominal growth only"
        if any("growth factor g" in f for f in FACTS_TEXT):
            raise SystemExit(f"{fp} states the growth factor g: rerun facts.py --data {a.data}")
    elif man.get("prompt_schema") == "v7":
        assert FACTS_TEXT, "schema v7 needs <data>/facts.json (facts.py)"
        SCHEMA = schema = "v7"
        GROWTH_LINE = growth_randomised(man)
        if not GROWTH_LINE:
            print("no growth randomisation in this dataset (growth_factor_range / p_nominal): v7 prompt WITHOUT the growth-factor line")
            if any("growth factor g" in f for f in FACTS_TEXT):
                raise SystemExit(f"{fp} states the growth factor g but this dataset's prompt has no g line: rerun facts.py --data {a.data}")
    else:
        schema = "v5facts" if FACTS_TEXT else "v5"
    print(f"prompt schema {schema}" + (f" ({fp})" if FACTS_TEXT else "") + f"; profile {prof['name']}: {cfg.SHEET.thickness_mm} mm, dose {cfg.DOSE}")
    split_of = {m["id"]: m["split"] for m in man["cases"]} if all("split" in m for m in man["cases"]) else None
    val_ids = set()
    if split_of:                                              # the manifest's train / fresh / test split (dataset D, 2026-10-30)
        val_ids = {i for i, sp in split_of.items() if sp == "test"}
        print(f"manifest split: " + str({sp: sum(1 for v in split_of.values() if v == sp) for sp in ("train", "fresh", "test")})
              + "; train cuts capped at --cap, fresh / test keep every cut; test -> val.jsonl, fresh -> fresh.jsonl")
    else:
        for cat in sorted({m["category"] for m in man["cases"]}):
            ids = [m["id"] for m in man["cases"] if m["category"] == cat]
            val_ids |= set(rng.choice(ids, size=max(1, int(round(len(ids) * a.val_frac))), replace=False))
    cuts = None; a.aug_real = getattr(a, "aug_real", False); a.match_cuts = getattr(a, "match_cuts", None)
    if a.match_cuts:                                          # the cut list (and split) of a reference pairs dir, row for row
        ref = os.path.join(HERE, a.match_cuts); cuts = {}
        for name in ("train", "fresh", "val"):
            if not os.path.exists(os.path.join(ref, f"{name}.jsonl")): continue
            for line in open(os.path.join(ref, f"{name}.jsonl")):
                m = json.loads(line)["meta"]; cuts.setdefault(m["source"], []).append(m["cut"])
        ref_meta = json.load(open(os.path.join(ref, "pairs_meta.json")))
        assert set(ref_meta["val_ids"]) == {str(v) for v in val_ids}, f"--match-cuts {a.match_cuts}: its validation split differs (seed / val-frac)"
        if split_of: assert ref_meta.get("splits") == split_of, f"--match-cuts {a.match_cuts}: its train / fresh / test split differs"
    new = a.aug_real or cuts is not None
    cap_of = lambda tid: a.cap if not split_of or split_of[tid] == "train" else 10**6
    jobs = [(os.path.join(datadir, m["id"]), outdir, cap_of(m["id"]), a.seed, a.aug, a.fake) + ((a.aug_real, cuts) if new else ()) for m in man["cases"]]
    tr, va, fr, percat, how = [], [], [], {}, {}
    with Pool(a.workers) as pool:
        for tid, samples in pool.imap_unordered(process_traj, jobs):
            (fr if split_of and split_of[tid] == "fresh" else va if tid in val_ids else tr).extend(samples)
            for x in samples:
                percat[x["meta"]["category"]] = percat.get(x["meta"]["category"], 0) + 1
                h = x["meta"]["current"].split("(")[0]; how[h] = how.get(h, 0) + 1
            print(f"{tid}: {len(samples)} pairs", flush=True)
    rng.shuffle(tr)
    for name, rows in (("train", tr), ("val", va)) + ((("fresh", fr),) if split_of else ()):
        with open(os.path.join(outdir, f"{name}.jsonl"), "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    json.dump({"sheet": cfg.SHEET.to_dict(), "dose": cfg.DOSE, "profile": cfg.PROFILE, "decay": cfg.DECAY, "tau_mm": cfg.TAU_PAIR_MM, "aug": a.aug, "fake": a.fake,
               "prompt_schema": schema, "crop_mm": cfg.CROP_MM, "reachable": cfg.reachable_zones(),
               "val_ids": sorted(val_ids), "train": len(tr), "val": len(va), "per_category": percat, "current_images": how,
               **({"splits": split_of, "fresh": len(fr), "val_is": "the manifest's test split (fresh.jsonl = its fresh split)",
                   "cuts": f"train capped at {a.cap} per trajectory, horizon-stratified, flat -> final always kept; fresh / test every cut"}
                  if split_of else {}),
               **({"growth_line": GROWTH_LINE} if schema == "v7" else {}),
               **({"aug_real": True, "aug_real_params": AUGREAL, "target": "clipped to p0.5..p99.5 (ew_loop.clean.robust_clip)",
                   "current": "degrade_real -> ew_loop.clean.clean_field"} if a.aug_real else {}),
               **({"match_cuts": a.match_cuts} if a.match_cuts else {})},
              open(os.path.join(outdir, "pairs_meta.json"), "w"), indent=1)
    print(f"\ntrain {len(tr)} / val {len(va)}" + (f" / fresh {len(fr)}" if split_of else "") + f"; per category {percat}; current images {how}; val {sorted(val_ids)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=30); ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--aug", action="store_true"); ap.add_argument("--fake", type=float, default=0.0, help="probability of a fake-camera CURRENT image")
    ap.add_argument("--aug-real", action="store_true", help="degrade-then-clean (2026-10-30): CURRENT = degrade_real(sim) -> ew_loop.clean.clean_field "
                    "(image + prompt range from the cleaned field), TARGET clipped to its p0.5..p99.5 as the loop does; with --fake p a share "
                    "of fake-camera captures, also cleaned (off by default: after cleaning they are far too clean)")
    ap.add_argument("--match-cuts", help="pairs dir whose (source, cut) list and split are reproduced row for row (instead of pairs_for)")
    main(ap.parse_args())
