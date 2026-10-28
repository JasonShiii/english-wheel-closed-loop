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

# The per-request part. Our model is trained on PROMPT_SYSTEM + "\n\n" + PROMPT_USER as one user message; the closed
# models get PROMPT_SYSTEM once as the system message and PROMPT_USER (+ the two images) per example / query turn, so
# the invariant text is not repeated k times in a k-shot request. split_prompt() recovers the two parts.
USER_MARK = "\n\nREQUEST\n"
PROMPT_USER = """REQUEST
Image 1 (CURRENT) z range = [{cz0}, {cz1}] mm. Image 2 (TARGET) z range = [{tz0}, {tz1}] mm.
HISTORY:
{history}"""


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
    return PROMPT_SYSTEM.format(W=s.width_mm, L=s.length_mm, t=s.thickness_mm, ylo=-s.length_mm / 2, yc=cfg.crop_y(),
                                xlo_s=-s.width_mm / 2, xhi_s=s.width_mm / 2, yhi_s=s.length_mm / 2,
                                xlo=x_lo, xhi=x_hi, yzlo=y_lo, yhi=y_hi, mside=s.margin_side_mm, mtop=cfg.MARGIN_TOP_MM,
                                m=R.MARGIN_MM, fx0=fr[0], fx1=fr[1], fy0=fr[2], fy1=fr[3], facts=facts_block())


def user_prompt(cz, tz, history):
    hist = json.dumps(history, indent=1) if history else "(empty - the sheet is untouched)"
    return PROMPT_USER.format(cz0=round(float(cz[0]), 2), cz1=round(float(cz[1]), 2),
                              tz0=round(float(tz[0]), 2), tz1=round(float(tz[1]), 2), history=hist)


def build_prompt(cz, tz, history):
    """the text our model is trained on: system part + per-request part in ONE user message"""
    return system_prompt() + "\n\n" + user_prompt(cz, tz, history)


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
    return sorted(out)


def process_traj(args):
    tdir, outdir, cap, seed, aug, p_fake = args
    rec = json.load(open(os.path.join(tdir, "record.json"))); npz = os.path.join(tdir, "cycles.npz")
    if not os.path.exists(npz): return rec["id"], []
    xyz = np.load(npz)["xyz"].astype(float)                   # (T+1, n, 3) mm, clamped states
    Z = xyz[:, :, 2]
    rng = np.random.default_rng(seed + hash(rec["id"]) % 10**6)
    tag0 = rec["id"].replace("/", "_"); samples = []
    for s, t in pairs_for(rec["T"], Z, cap, rng):
        tag = f"{tag0}__cut{s:02d}-{t:02d}"
        ft = field_of(xyz[t]); fc = field_of(xyz[s]); how = "clean"
        if p_fake > 0 and rng.random() < p_fake:
            try: fc = fake_capture_field(xyz[s], rng); how = "fake_camera"
            except Exception as e: how = f"clean(fake failed: {type(e).__name__})"
        elif aug:
            fc = augment(fc, rng); how = "aug"
        pc, pt = os.path.join(outdir, "images", f"{tag}_cur.png"), os.path.join(outdir, "images", f"{tag}_tgt.png")
        R.field_png(fc, pc); R.field_png(ft, pt)
        hist, plan = slice_actions(rec["was"], s, t, schema="v2")
        prompt = build_prompt(fc.zrange(), ft.zrange(), hist)
        samples.append({"meta": {"source": rec["id"], "category": rec["category"], "cut": [s, t], "current": how},
                        "messages": [{"role": "user", "content": [{"type": "image", "image": f"images/{tag}_cur.png"},
                                                                  {"type": "image", "image": f"images/{tag}_tgt.png"},
                                                                  {"type": "text", "text": prompt}]},
                                     {"role": "assistant", "content": [{"type": "text", "text": json.dumps({"plan": plan})}]}]})
    return rec["id"], samples


def main(a):
    datadir, outdir = os.path.join(HERE, a.data), os.path.join(HERE, a.out)
    os.makedirs(os.path.join(outdir, "images"), exist_ok=True)
    man = json.load(open(os.path.join(datadir, "manifest.json"))); rng = np.random.default_rng(a.seed)
    global FACTS_TEXT
    fp = os.path.join(datadir, "facts.json")
    FACTS_TEXT = json.load(open(fp))["facts"] if os.path.exists(fp) else None
    schema = "v5facts" if FACTS_TEXT else "v5"; print(f"prompt schema {schema}" + (f" ({fp})" if FACTS_TEXT else ""))
    val_ids = set()
    for cat in sorted({m["category"] for m in man["cases"]}):
        ids = [m["id"] for m in man["cases"] if m["category"] == cat]
        val_ids |= set(rng.choice(ids, size=max(1, int(round(len(ids) * a.val_frac))), replace=False))
    jobs = [(os.path.join(datadir, m["id"]), outdir, a.cap, a.seed, a.aug, a.fake) for m in man["cases"]]
    tr, va, percat, how = [], [], {}, {}
    with Pool(a.workers) as pool:
        for tid, samples in pool.imap_unordered(process_traj, jobs):
            (va if tid in val_ids else tr).extend(samples)
            for x in samples:
                percat[x["meta"]["category"]] = percat.get(x["meta"]["category"], 0) + 1
                h = x["meta"]["current"].split("(")[0]; how[h] = how.get(h, 0) + 1
            print(f"{tid}: {len(samples)} pairs", flush=True)
    rng.shuffle(tr)
    for name, rows in (("train", tr), ("val", va)):
        with open(os.path.join(outdir, f"{name}.jsonl"), "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
    json.dump({"sheet": cfg.SHEET.to_dict(), "dose": cfg.DOSE, "tau_mm": cfg.TAU_PAIR_MM, "aug": a.aug, "fake": a.fake,
               "prompt_schema": schema, "crop_mm": cfg.CROP_MM, "reachable": cfg.reachable_zones(),
               "val_ids": sorted(val_ids), "train": len(tr), "val": len(va), "per_category": percat, "current_images": how},
              open(os.path.join(outdir, "pairs_meta.json"), "w"), indent=1)
    print(f"\ntrain {len(tr)} / val {len(va)}; per category {percat}; current images {how}; val {sorted(val_ids)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=30); ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--aug", action="store_true"); ap.add_argument("--fake", type=float, default=0.0, help="probability of a fake-camera CURRENT image")
    main(ap.parse_args())
