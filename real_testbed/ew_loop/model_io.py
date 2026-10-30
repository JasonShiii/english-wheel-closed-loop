"""What the loop must share exactly with the trained model and the simulator data generation.

Copied (not imported) so that the ew_loop folder runs on its own on the robot PC with numpy
only. tests/test_model_io.py checks every item against its original in this repository:

  prompt text, history/plan schema   experiments/mpc_end2end/common.py
  height-field rendering (heatmap)   experiments/mpc_end2end/common.py
  working area -> zigzag inscription experiments/category_tests/category_tests.py
  zigzag corner points               utils/process_rotation_data.py (zigzag_vertices)
  shape metrics                      experiments/mpc_end2end/eval_execute.py (field_metrics)

If the model is retrained with another prompt, schema or toolpath parametrisation, update
this file together with the checkpoint.
"""
import json
import math

import numpy as np

# ---- prompt and plan schema (experiments/mpc_end2end/common.py) -------------------------


def wa_desc(wa):
    """WA = working area GEOMETRY only (schema v2, 2026-08-28)."""
    return {"center_xy": [wa["cx"], wa["cy"]], "size_wh": [wa["wx"], wa["wy"]]}


def zz_desc(wa, schema=None):
    """zigzag parameters, separated from the WA geometry (schema v2). Schema v7 (2026-10-30, continuous angle): the
    orientation is 'rot_deg' in integer degrees (CCW in the sheet frame; 0 and 90 are the old rot values)."""
    if schema == "v7":
        return {"rot_deg": int(round(float(wa["rot_deg"] if "rot_deg" in wa else wa["rot"]))), "N": wa["N"]}
    return {"rot": wa["rot"], "N": wa["N"]}


SCHEMA_DEFAULT = 'v2'


def entry_desc(wa, n, key="hits", schema=SCHEMA_DEFAULT):
    if schema == "v1":
        return {"wa": {**wa_desc(wa), **zz_desc(wa)}, key: n}
    return {"wa": wa_desc(wa), "zigzag": zz_desc(wa, schema), key: n}


def read_entry(e):
    """Tolerant reader: accepts schema v2 {wa:{geometry}, zigzag:{rot,N}, hits}
    AND legacy v1 {wa:{geometry, rot, N}, hits}. Returns a flat dict."""
    wa = e["wa"]
    zz = e.get("zigzag", wa)
    rot = int(round(float(zz["rot_deg"] if "rot_deg" in zz else zz["rot"])))      # schema v7 'rot_deg' or the old 'rot' (0 / 90)
    return dict(cx=float(wa["center_xy"][0]), cy=float(wa["center_xy"][1]),
                wx=float(wa["size_wh"][0]), wy=float(wa["size_wh"][1]),
                rot=rot, rot_deg=rot, N=int(zz["N"]),
                n=int(e.get("hits", e.get("hits_done", 0))))


PROMPT_TEMPLATE_V1 = 'You are a process planner for incremental sheet-metal forming with an English wheel on a flat 300 x 300 mm panel.\n\nINPUTS\n1. Image 1 - CURRENT shape: top view of the panel height field (x to the right, y up, in mm). The image covers the full panel: x in [-150, 150] mm, y in [-150, 150] mm (300 x 300 mm, center at the origin). Pure white pixels near the border mean "outside the current panel footprint" - the sheet pulls inward as it deforms, so its top-view projection can be smaller than the frame; white carries no height information. Colors elsewhere are normalized to THIS image\'s own min/max, so read the shape from the image and the scale from the text: current z range = [{cz0}, {cz1}] mm.\n2. Image 2 - TARGET shape: same convention; target z range = [{tz0}, {tz1}] mm.\n3. HISTORY - ordered list of wheeling actions already executed (may be empty):\n{history}\n   Each action: "wa" is a rectangular working area, center_xy = its center [x, y] in mm, size_wh = its width and height in mm; rot is the zigzag orientation (0: strips run along y, stacked along x; 90: strips run along x, stacked along y); N is the number of strips (density). "hits_done" = how many passes of this zigzag were already applied. Because of work hardening, repeated passes over already-wheeled regions produce progressively smaller deformation.\n\nTASK\nPredict the remaining action sequence that transforms CURRENT into TARGET. Answer with ONLY a JSON object, no other text:\n{{"plan": [{{"wa": {{"center_xy": [x, y], "size_wh": [w, h], "rot": 0 or 90, "N": int}}, "hits": int}}, ...]}}\nEntries are in execution order; hits >= 1. If the last history action is not finished, the first plan entry may continue the same wa with the remaining hits.'


PROMPT_TEMPLATE_V2 = 'You are a process planner for incremental sheet-metal forming with an English wheel on a flat 300 x 300 mm panel.\n\nINPUTS\n1. Image 1 - CURRENT shape: top view of the panel height field (x to the right, y up, in mm). The image covers the full panel: x in [-150, 150] mm, y in [-150, 150] mm (300 x 300 mm, center at the origin). Pure white pixels near the border mean "outside the current panel footprint" - the sheet pulls inward as it deforms, so its top-view projection can be smaller than the frame; white carries no height information. Colors elsewhere are normalized to THIS image\'s own min/max, so read the shape from the image and the scale from the text: current z range = [{cz0}, {cz1}] mm.\n2. Image 2 - TARGET shape: same convention; target z range = [{tz0}, {tz1}] mm.\n3. HISTORY - ordered list of wheeling actions already executed (may be empty):\n{history}\n   Each action has three parts. "wa" is the rectangular working area: center_xy = its center [x, y] in mm, size_wh = its width and height in mm. "zigzag" is the toolpath applied inside that area: rot is its orientation (0: strips run along y, stacked along x; 90: strips run along x, stacked along y), N is the number of strips (density). "hits_done" = how many passes of this zigzag were already applied. Because of work hardening, repeated passes over already-wheeled regions produce progressively smaller deformation.\n\nTASK\nPredict the remaining action sequence that transforms CURRENT into TARGET. Answer with ONLY a JSON object, no other text:\n{{"plan": [{{"wa": {{"center_xy": [x, y], "size_wh": [w, h]}}, "zigzag": {{"rot": 0 or 90, "N": int}}, "hits": int}}, ...]}}\nEntries are in execution order; hits >= 1. If the last history action is not finished, the first plan entry may continue the same wa with the remaining hits.'


def build_prompt(cz, tz, history, schema=SCHEMA_DEFAULT):
    hist = json.dumps(history, indent=1) if history else "   (empty - the panel is untouched)"
    tpl = PROMPT_TEMPLATE_V1 if schema == "v1" else PROMPT_TEMPLATE_V2
    return tpl.format(cz0=round(float(cz[0]), 2), cz1=round(float(cz[1]), 2),
                                  tz0=round(float(tz[0]), 2), tz1=round(float(tz[1]), 2),
                                  history=hist)


# ---- rendering of a height field as the model saw it in training (common.py) -------------
# The model is sensitive to small rendering differences (a 1-pixel white border flipped the
# planned zigzag orientation on a test case), so fields are drawn with this exact code.
GRID = 256


def heatmap_png(P, z, path):
    """Smooth per-image-normalized height map over spatial (x,y); numbers go to text (D16)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    x, y = P[:, 0] * 1e3, P[:, 1] * 1e3
    g = np.linspace(-150, 150, GRID)
    X, Y = np.meshgrid(g, g)
    H = griddata((x, y), z, (X, Y), method="linear")
    # Outside the projected panel footprint (the sheet pulls inward as it deforms)
    # there is NO data. Convention (B2, 2026-08-24): paint it WHITE — outside the
    # viridis gamut (no confusion with low-z dark purple) and matches the VLM's
    # white-background prior. Mask boundary is morphologically smoothed (geometry
    # only; the <=2 px band added by smoothing is filled with nearest EDGE values,
    # interior heights are never fabricated).
    from scipy.ndimage import gaussian_filter
    valid = ~np.isnan(H)
    support = gaussian_filter(valid.astype(float), 1.2) >= 0.5
    Hn = griddata((x, y), z, (X, Y), method="nearest")
    H = np.where(support, np.where(valid, H, Hn), np.nan)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    fig = plt.figure(figsize=(3.2, 3.2), dpi=140)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(np.ma.masked_invalid(H), origin="lower", cmap=cmap,
              extent=(-150, 150, -150, 150))
    fig.savefig(path); plt.close(fig)


# ---- working area -> zigzag (experiments/category_tests/category_tests.py) ---------------
W = 10.0            # strip (wheel contact) width used by the data generation, mm


def strip_stack_dims(wx, wy, rot):
    return (wy, wx) if rot == 0 else (wx, wy)


def inscribe(wx, wy, rot, N):
    a, b = strip_stack_dims(wx, wy, rot)
    lv = a - W
    return lv, math.degrees(math.atan((b - W) / ((N - 2) * lv)))


# ---- zigzag corner points (utils/process_rotation_data.py) -------------------------------
def zigzag_vertices(lv_mm: float, alpha_deg: float, N: int,
                    rotation_deg: float = 0.0) -> np.ndarray:
    """
    Return (N+1, 3) corner vertices of zigzag toolpath, centred at origin.
    If rotation_deg != 0, rotate the 2D pattern around origin.
    rotation_deg: CCW positive, CW negative (from x-axis).
    """
    assert N >= 2
    y_top = +lv_mm / 2.0
    y_bot = -lv_mm / 2.0
    w = lv_mm * np.tan(np.radians(alpha_deg))
    x0 = -(N - 2) / 2.0 * w
    verts = []
    for k in range(N + 1):
        y = y_bot if (k % 2 == 0) else y_top
        if k == 0:
            x = x0
        elif k == N:
            x = x0 + (N - 2) * w
        else:
            x = x0 + (k - 1) * w
        verts.append([x, y, 0.0])
    verts = np.array(verts)

    # Apply rotation if needed
    if rotation_deg != 0.0:
        theta = np.radians(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([[cos_t, -sin_t],
                               [sin_t,  cos_t]])
        verts[:, :2] = (rot_matrix @ verts[:, :2].T).T

    return verts


# ---- shape metrics (experiments/mpc_end2end/eval_execute.py) -----------------------------
def field_metrics(z_pred, z_gt):
    zp, zg = z_pred - z_pred.mean(), z_gt - z_gt.mean()
    corr = float(np.corrcoef(zp, zg)[0, 1])
    return dict(pattern_corr=round(corr, 4),
                span_pred=round(float(np.ptp(z_pred)), 2), span_gt=round(float(np.ptp(z_gt)), 2),
                amp_ratio=round(float(np.ptp(z_pred) / max(np.ptp(z_gt), 1e-9)), 3),
                resid_mean=round(float(np.abs(zp - zg).mean()), 2),
                resid_max=round(float(np.abs(zp - zg).max()), 2))
