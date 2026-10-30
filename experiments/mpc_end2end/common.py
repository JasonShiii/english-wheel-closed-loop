"""Shared MPC-dataset assets: WA schema, prompt template, heatmap rendering.
Single source of truth — make_mpc_samples.py (preview) and derive_pairs.py import here."""
import json
import numpy as np

GRID = 256
TAU_PAIR_MM = 2.0                    # aligned with deployment amplitude-band floor (07 doc)


def wa_desc(wa):
    """WA = working area GEOMETRY only (schema v2, 2026-08-28)."""
    return {"center_xy": [wa["cx"], wa["cy"]], "size_wh": [wa["wx"], wa["wy"]]}


def zz_desc(wa, schema=None):
    """zigzag parameters, separated from the WA geometry (schema v2). Schema v7 (2026-10-30, continuous angle): the
    orientation is 'rot_deg' in integer degrees (CCW in the sheet frame; 0 and 90 are the old rot values)."""
    if schema == "v7":
        return {"rot_deg": int(round(float(wa["rot_deg"] if "rot_deg" in wa else wa["rot"]))), "N": wa["N"]}
    return {"rot": wa["rot"], "N": wa["N"]}


SCHEMA_DEFAULT = "v2"          # v2 = wa/zigzag separated (2026-08-28). A checkpoint must be
                               # prompted with the schema it was TRAINED on (current one: v1).


def entry_desc(wa, n, key="hits", schema=SCHEMA_DEFAULT):
    if schema in ("v1", "v1facts"):          # v1facts = v1 entries + PROCESS_FACTS in the prompt
        return {"wa": {**wa_desc(wa), **zz_desc(wa)}, key: n}
    return {"wa": wa_desc(wa), "zigzag": zz_desc(wa, schema), key: n}


def read_entry(e):
    """Tolerant reader: accepts schema v2 {wa:{geometry}, zigzag:{rot,N}, hits}, legacy v1 {wa:{geometry, rot, N}, hits}
    AND schema v7 {wa:{geometry}, zigzag:{rot_deg,N}, hits}. Returns a flat dict whose 'rot' and 'rot_deg' both hold the
    orientation in integer degrees (rot 0 / 90 of the old schemas map to rot_deg 0 / 90)."""
    wa = e["wa"]
    zz = e.get("zigzag", wa)
    rot = int(round(float(zz["rot_deg"] if "rot_deg" in zz else zz["rot"])))
    return dict(cx=float(wa["center_xy"][0]), cy=float(wa["center_xy"][1]),
                wx=float(wa["size_wh"][0]), wy=float(wa["size_wh"][1]),
                rot=rot, rot_deg=rot, N=int(zz["N"]),
                n=int(e.get("hits", e.get("hits_done", 0))))


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


PROMPT_TEMPLATE_V1 = """You are a process planner for incremental sheet-metal forming with an \
English wheel on a flat 300 x 300 mm panel.

INPUTS
1. Image 1 - CURRENT shape: top view of the panel height field (x to the right, y up, \
in mm). The image covers the full panel: x in [-150, 150] mm, y in [-150, 150] mm \
(300 x 300 mm, center at the origin). Pure white pixels near the border mean \
"outside the current panel footprint" - the sheet pulls inward as it deforms, so \
its top-view projection can be smaller than the frame; white carries no height \
information. Colors elsewhere are normalized to THIS image's own min/max, so read \
the shape from the image and the scale from the text: current \
z range = [{cz0}, {cz1}] mm.
2. Image 2 - TARGET shape: same convention; target z range = [{tz0}, {tz1}] mm.
3. HISTORY - ordered list of wheeling actions already executed (may be empty):
{history}
   Each action: "wa" is a rectangular working area, center_xy = its center [x, y] in mm, \
size_wh = its width and height in mm; rot is the zigzag orientation (0: strips run \
along y, stacked along x; 90: strips run along x, stacked along y); N is the number of \
strips (density). \
"hits_done" = how many passes of this zigzag were already applied. Because of work \
hardening, repeated passes over already-wheeled regions produce progressively smaller \
deformation.

TASK
Predict the remaining action sequence that transforms CURRENT into TARGET. Answer with \
ONLY a JSON object, no other text:
{{"plan": [{{"wa": {{"center_xy": [x, y], "size_wh": [w, h], "rot": 0 or 90, "N": int}}, \
"hits": int}}, ...]}}
Entries are in execution order; hits >= 1. If the last history action is not finished, \
the first plan entry may continue the same wa with the remaining hits."""




PROMPT_TEMPLATE_V2 = """You are a process planner for incremental sheet-metal forming with an \
English wheel on a flat 300 x 300 mm panel.

INPUTS
1. Image 1 - CURRENT shape: top view of the panel height field (x to the right, y up, \
in mm). The image covers the full panel: x in [-150, 150] mm, y in [-150, 150] mm \
(300 x 300 mm, center at the origin). Pure white pixels near the border mean \
"outside the current panel footprint" - the sheet pulls inward as it deforms, so \
its top-view projection can be smaller than the frame; white carries no height \
information. Colors elsewhere are normalized to THIS image's own min/max, so read \
the shape from the image and the scale from the text: current \
z range = [{cz0}, {cz1}] mm.
2. Image 2 - TARGET shape: same convention; target z range = [{tz0}, {tz1}] mm.
3. HISTORY - ordered list of wheeling actions already executed (may be empty):
{history}
   Each action has three parts. "wa" is the rectangular working area: center_xy = its \
center [x, y] in mm, size_wh = its width and height in mm. "zigzag" is the toolpath \
applied inside that area: rot is its orientation (0: strips run along y, stacked along x; \
90: strips run along x, stacked along y), N is the number of strips (density). \
"hits_done" = how many passes of this zigzag were already applied. Because of work \
hardening, repeated passes over already-wheeled regions produce progressively smaller \
deformation.

TASK
Predict the remaining action sequence that transforms CURRENT into TARGET. Answer with \
ONLY a JSON object, no other text:
{{"plan": [{{"wa": {{"center_xy": [x, y], "size_wh": [w, h]}}, \
"zigzag": {{"rot": 0 or 90, "N": int}}, "hits": int}}, ...]}}
Entries are in execution order; hits >= 1. If the last history action is not finished, \
the first plan entry may continue the same wa with the remaining hits."""


# Process facts inside the prompt (decision 2026-10-08, closed-model comparison): the same text goes to
# our model (training prompt, schema "v1facts") and to the closed models. The numbers are statistics
# of the training trajectories (GPT-astra_comparison/fix_conditions.py) -> RECOMPUTE when the data is
# regenerated; the wording may change once the real-testbed setting is confirmed.
PROCESS_FACTS = """\
- The English wheel only adds deformation: a panel formed past the target cannot be formed back, so \
under-forming is recoverable and over-forming is not.
- Orientation sets the curvature direction: rot 0 makes the height change mainly along x (ridge \
parallel to y), rot 90 mainly along y.
- Larger N and longer strips deform more per pass. On a flat panel, a first pass with one working \
area adds 10-30 mm of height range (median 18 mm); each further pass over the same area adds about \
0.75 times what the previous pass added.
- Values in the reference plans: center x from -121 to 121 mm, y from -127 to 127 mm; sizes 36-282 mm; \
N 4-20 (rot 0: 5-14); hits 1-6 per entry; 1-7 entries per plan."""

PROMPT_TEMPLATE_V1_FACTS = PROMPT_TEMPLATE_V1.replace(
    "\n\nTASK\n", "\n\nPROCESS FACTS (measured on the simulator that produced the reference plans)\n"
    + PROCESS_FACTS + "\n\nTASK\n")
assert PROMPT_TEMPLATE_V1_FACTS != PROMPT_TEMPLATE_V1
PROMPT_TEMPLATES = {"v1": PROMPT_TEMPLATE_V1, "v1facts": PROMPT_TEMPLATE_V1_FACTS, "v2": PROMPT_TEMPLATE_V2}


def build_prompt(cz, tz, history, schema=SCHEMA_DEFAULT):
    hist = json.dumps(history, indent=1) if history else "   (empty - the panel is untouched)"
    tpl = PROMPT_TEMPLATES[schema]
    return tpl.format(cz0=round(float(cz[0]), 2), cz1=round(float(cz[1]), 2),
                                  tz0=round(float(tz[0]), 2), tz1=round(float(tz[1]), 2),
                                  history=hist)


def slice_actions(was, start, target, schema=SCHEMA_DEFAULT):
    """history (prefix) + plan (suffix) for a cycle-level cut. Mid-WA cuts allowed."""
    bounds = np.cumsum([0] + [w["n"] for w in was])
    hist, plan = [], []
    for k, wa in enumerate(was):
        done = int(np.clip(start - bounds[k], 0, wa["n"]))
        upto = int(np.clip(target - bounds[k], 0, wa["n"]))
        if done > 0:
            hist.append(entry_desc(wa, done, "hits_done", schema))
        if upto - done > 0:
            plan.append(entry_desc(wa, upto - done, "hits", schema))
    return hist, plan
