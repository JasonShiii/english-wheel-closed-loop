"""Category tests, ortho = -0.5 (sign corrected 2026-08-23 after the group meeting:
the principal deformation axis was 90 deg off with ortho=+0.5).

Categories (10 samples each):
  C1 single-curvature arch : one center-ish WA, one zigzag repeated
  C2 arch + edge bands     : center WA (rot0) then two edge bands (rot90)
  C3 even crown            : concentric shrinking SQUARE bboxes, same orientation,
                             each zigzag repeated 2-3x; then rotate 90 deg and repeat
                             the shrinking series ("cross wheel") -> even crown.
                             (nested/overlapping by design)

Parameterization = converged design (05_hierarchy_design.md): closed-form inscription
lv = a - w, tan(alpha) = (b - w)/((N-2)*lv); free per-WA params = rot{0,90} + N;
w / profile / gtop / ortho frozen.

Usage:
  python category_tests.py sample --seed 0
  python category_tests.py run --workers 20 --cores 8
  python category_tests.py analyze          # metrics + montages + per-case figures
  python category_tests.py figures          # per-case figures only
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import time
from multiprocessing import Pool

import numpy as np

BIN = "/home/xiangyu/English_wheel/PSM2025_recurring/bin/shell"
ROOT = os.path.dirname(os.path.abspath(__file__))
BATCH = "2026-08-23_ortho-0.5"

# ---- environment constants ---------------------------------------------------
W = 10.0
GTOP, ORTHO = 0.0015, -0.5                 # ORTHO sign corrected
TAN_MIN, TAN_MAX = math.tan(math.radians(2.86)), math.tan(math.radians(9.13))
N_MIN, N_MAX = 4, 14
PANEL_HALF, PLACE_MAX, GAP = 150.0, 146.0, 6.0
CPK = {"mode": "center_peak", "top_end_ratio": 0.4, "top_profile_power": 2.0}
HARDENING = {"model": "voce_decay", "history_variable": "hit_count",
             "beta": 0.223143551, "floor": 0.0, "scope": "face_shared",
             "count_rule": "plastic_hits"}
CATS = ["C1", "C2", "C3", "C4", "C3_lightfirst"]


# ---- inscription rule (rot=0: strips along y, stacking along x) ----------------
def strip_stack_dims(wx, wy, rot):
    return (wy, wx) if rot == 0 else (wx, wy)


def feasible_N(wx, wy, rot, nmax=N_MAX):
    a, b = strip_stack_dims(wx, wy, rot)
    lv = a - W
    if lv <= 0 or b <= W:
        return []
    lo = max(N_MIN, 2 + math.ceil((b - W) / (lv * TAN_MAX) - 1e-9))
    hi = min(nmax, 2 + math.floor((b - W) / (lv * TAN_MIN) + 1e-9))
    return list(range(lo, hi + 1))


def inscribe(wx, wy, rot, N):
    a, b = strip_stack_dims(wx, wy, rot)
    lv = a - W
    return lv, math.degrees(math.atan((b - W) / ((N - 2) * lv)))


def wa_op(idx, wa):
    lv, alpha = inscribe(wa["wx"], wa["wy"], wa["rot"], wa["N"])
    op = {"type": "zigzag", "lv_mm": round(lv, 2), "alpha_deg": round(alpha, 3),
          "n_strips": wa["N"], "width_mm": W,
          "center_uv_mm": [wa["cx"], wa["cy"]], "shift_uv_mm": [0.0, 0.0],
          "shift_frame": "material", "rotation_deg": float(wa["rot"]),
          "gtop": GTOP, "gbot": 0.0, "ortho": ORTHO}
    return {"id": f"WA{idx}", "enabled": True, "repeat": wa["n"], "operation": op}


def spec_for(was):
    return {"schema_version": 1,
            "units": {"length": "mm", "angle": "deg", "growth": "engineering_strain"},
            "hardening": HARDENING,
            "defaults": {"start_mode": "left_bottom_up", "profile": CPK},
            "toolpaths": [wa_op(i, wa) for i, wa in enumerate(was)]}


# ---- sampling helpers --------------------------------------------------------
def fits_panel(wa):
    return (abs(wa["cx"]) + wa["wx"] / 2 <= PLACE_MAX and
            abs(wa["cy"]) + wa["wy"] / 2 <= PLACE_MAX)


def overlaps(w1, w2, gap=GAP):
    return (abs(w1["cx"] - w2["cx"]) < (w1["wx"] + w2["wx"]) / 2 + gap and
            abs(w1["cy"] - w2["cy"]) < (w1["wy"] + w2["wy"]) / 2 + gap)


def mk_wa(rng, wx, wy, rot, cx, cy, n, nmax=N_MAX, **extra):
    Ns = feasible_N(wx, wy, rot, nmax)
    if not Ns:
        return None
    wa = dict(wx=round(wx, 1), wy=round(wy, 1), rot=int(rot), cx=round(cx, 1),
              cy=round(cy, 1), N=int(rng.choice(Ns)), n=int(n), **extra)
    return wa if fits_panel(wa) else None


def sample_C1(rng):
    for _ in range(50):
        rot = int(rng.choice([0, 90]))
        a, b = rng.uniform(110, 240), rng.uniform(70, 190)
        wx, wy = (b, a) if rot == 0 else (a, b)
        wa = mk_wa(rng, wx, wy, rot, rng.uniform(-15, 15), rng.uniform(-15, 15),
                   rng.integers(2, 7))
        if wa:
            return [wa]


def sample_C2(rng):
    for _ in range(80):
        center = mk_wa(rng, rng.uniform(140, 200), rng.uniform(120, 150), 0,
                       rng.uniform(-8, 8), rng.uniform(-8, 8), rng.integers(2, 5))
        if not center:
            continue
        wyb, wxb = rng.uniform(36, 64), rng.uniform(180, 260)
        cyb = PLACE_MAX - wyb / 2 - rng.uniform(0, 4)
        nb = int(rng.integers(2, 5))
        bp = mk_wa(rng, wxb, wyb, 90, rng.uniform(-5, 5), cyb, nb)
        bm = mk_wa(rng, wxb, wyb, 90, rng.uniform(-5, 5), -cyb, nb)
        if bp and bm and not overlaps(center, bp) and not overlaps(center, bm):
            return [center, bp, bm]


def sample_C3(rng):
    """Even crown: shrinking concentric squares (phase 1, rot r1) then the same
    procedure rotated 90 deg (phase 2, cross wheel)."""
    r1 = int(rng.choice([0, 90])); r2 = 90 - r1
    s0 = rng.uniform(240, 285)
    for _ in range(50):
        was = []
        for phase, rot in ((1, r1), (2, r2)):
            K = int(rng.integers(2, 5))                     # 2-4 squares
            s = s0
            for k in range(K):
                if k > 0:
                    s = s * rng.uniform(0.6, 0.8)
                wa = mk_wa(rng, s, s, rot, 0.0, 0.0, rng.integers(2, 4), phase=phase)
                if wa is None:
                    break
                was.append(wa)
        if len(was) >= 4:
            return was


def sample_C4(rng):
    """Dome intent (corrected 2026-08-23): center WA rot0 (x-curvature, ridge || y) +
    two side WAs at x = +-edge with rot90 -> their principal deformation axis is
    ORTHOGONAL to z1's: they add y-curvature at the left/right sides. Side boxes are
    narrow-x tall-y; rot90 stacks strips along y, so their N uses a raised cap (20)
    while alpha stays inside the validated envelope."""
    for _ in range(200):
        center = mk_wa(rng, rng.uniform(120, 150), rng.uniform(140, 200), 0,
                       rng.uniform(-8, 8), rng.uniform(-8, 8), rng.integers(2, 5))
        if not center:
            continue
        wxb = rng.uniform(48, 64)
        wyb = rng.uniform(80, 140)
        cxb = PLACE_MAX - wxb / 2 - rng.uniform(0, 4)
        nb = int(rng.integers(2, 5))
        bl = mk_wa(rng, wxb, wyb, 90, -cxb, rng.uniform(-8, 8), nb, nmax=20)
        br = mk_wa(rng, wxb, wyb, 90, cxb, rng.uniform(-8, 8), nb, nmax=20)
        if bl and br and not overlaps(center, bl) and not overlaps(center, br):
            return [center, bl, br]


def sample_C3_lightfirst(rng):
    """Even crown, light-first schedule (artisan heuristic + hardening): phase 1 extends
    along x (rot0), 3 shrinking squares x 1/1/(1|2); phase 2 rot90, 3 squares x 2/3/3."""
    s0 = rng.uniform(240, 285)
    for _ in range(50):
        was = []
        for phase, rot, reps in ((1, 0, [1, 1, int(rng.choice([1, 2]))]), (2, 90, [2, 3, 3])):
            s = s0
            for k, n in enumerate(reps):
                if k > 0:
                    s = s * rng.uniform(0.6, 0.8)
                wa = mk_wa(rng, s, s, rot, 0.0, 0.0, n, phase=phase)
                if wa is None:
                    break
                was.append(wa)
        if len(was) == 6:
            return was


SAMPLERS = {"C1": sample_C1, "C2": sample_C2, "C3": sample_C3, "C4": sample_C4,
            "C3_lightfirst": sample_C3_lightfirst}


# ---- I/O helpers -------------------------------------------------------------
def read_vtp(path):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    r = vtk.vtkXMLPolyDataReader(); r.SetFileName(path); r.Update()
    pd = r.GetOutput()
    pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    tris = vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    return pts, tris


def case_dirs(outdir):
    for cat in CATS:
        for s in sorted(os.listdir(os.path.join(outdir, cat))):
            d = os.path.join(outdir, cat, s)
            if os.path.isdir(d):
                yield cat, s, d


def cycle_files(d):
    init = sorted(glob.glob(os.path.join(d, "*_initial.vtp")))
    finals = sorted(glob.glob(os.path.join(d, "*_final.vtp")))
    files = [("initial (flat)", init[0])] if init else []
    for f in finals:
        m = re.search(r"cycle_(\d+)_(WA\d+)_r(\d+)_final", f)
        files.append((f"c{int(m.group(1)):02d} {m.group(2)} hit{int(m.group(3))}", f))
    return files


# ---- commands: sample / run --------------------------------------------------
def cmd_sample(seed, per_cat=10, cats=None):
    rng = np.random.default_rng(seed)
    outdir = os.path.join(ROOT, BATCH)
    mpath = os.path.join(outdir, "manifest.json")
    prev = json.load(open(mpath))["cases"] if os.path.exists(mpath) else []
    cats = cats or list(SAMPLERS)
    manifest = [m for m in prev if m["category"] not in cats]
    for cat, fn in SAMPLERS.items():
        if cat not in cats:
            continue
        for s in range(per_cat):
            was = fn(rng)
            assert was, f"{cat} s{s}: sampler failed"
            sid = f"s{s:02d}"
            d = os.path.join(outdir, cat, sid); os.makedirs(d, exist_ok=True)
            json.dump(spec_for(was), open(os.path.join(d, "sequence.json"), "w"), indent=1)
            rec = dict(id=f"{cat}/{sid}", category=cat, was=was,
                       derived=[dict(zip(("lv", "alpha"), (round(v, 3) for v in
                                     inscribe(w["wx"], w["wy"], w["rot"], w["N"]))))
                                for w in was],
                       T=sum(w["n"] for w in was))
            json.dump(rec, open(os.path.join(d, "record.json"), "w"), indent=1)
            manifest.append(dict(id=rec["id"], dir=f"{cat}/{sid}", category=cat,
                                 T=rec["T"], nWA=len(was)))
    manifest.sort(key=lambda m: (CATS.index(m["category"]), m["id"]))
    json.dump(dict(seed=seed, env=dict(w=W, gtop=GTOP, ortho=ORTHO, profile="center_peak(0.4,2.0)",
                                       res=0.02, tol=3e-12), cases=manifest),
              open(mpath, "w"), indent=1)
    Ts = [m["T"] for m in manifest]
    print(f"sampled {len(manifest)} cases -> {outdir}; cycles {sum(Ts)} (T {min(Ts)}-{max(Ts)})")
    for cat in CATS:
        print(cat, "T:", [m["T"] for m in manifest if m["category"] == cat])


def run_one(args):
    d, cores_start, cores_n = args
    if os.path.exists(os.path.join(d, "done.flag")):
        return d, 0, 0.0
    cmd = ["taskset", "-c", f"{cores_start}-{cores_start+cores_n-1}", BIN,
           "-sim", "bilayer_growth", "-case", "custom", "-geometry", "rectangle",
           "-lx", "0.15", "-ly", "0.15", "-res", "0.02", "-h_total", "0.0005",
           "-growth_type", "zigzag_sequence", "-cycle_file", "sequence.json",
           "-enable_passE", "false", "-nsteps", "1", "-export_stl", "false", "-tol", "3e-12"]
    t0 = time.time()
    with open(os.path.join(d, "run.log"), "w") as logf:
        p = subprocess.run(cmd, cwd=d, stdout=logf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if p.returncode == 0:
        open(os.path.join(d, "done.flag"), "w").write(f"{wall:.1f}\n")
    return d, p.returncode, wall


def cmd_run(workers, cores):
    outdir = os.path.join(ROOT, BATCH)
    man = json.load(open(os.path.join(outdir, "manifest.json")))
    # heaviest first so the long C3 runs start immediately
    cases = sorted(man["cases"], key=lambda m: -m["T"])
    jobs = [(os.path.join(outdir, m["dir"]), (i % workers) * cores, cores) for i, m in enumerate(cases)]
    t0 = time.time(); ok = fail = 0
    with Pool(workers) as pool:
        for d, rc, wall in pool.imap_unordered(run_one, jobs):
            ok, fail = ok + (rc == 0), fail + (rc != 0)
            print(f"{os.path.relpath(d, outdir)} rc={rc} {wall:.0f}s  [{ok} ok/{fail} fail/{len(jobs)}; "
                  f"{time.time()-t0:.0f}s]", flush=True)
    print(f"batch done: {ok} ok, {fail} fail, {time.time()-t0:.0f}s wall")


# ---- figures -----------------------------------------------------------------
def zigzag_polyline(lv, alpha_deg, N, rot_deg, cx, cy):
    """Exact replica of ZigZagGrowth.hpp buildZigZagStrips (mm, panel frame)."""
    p = lv * math.tan(math.radians(alpha_deg))
    Ninc = N - 2
    yb, yt = -lv / 2, lv / 2
    xL, xR = -Ninc * p / 2, Ninc * p / 2
    pts = [(xL, yb), (xL, yt)]
    for j in range(1, Ninc + 1):
        pts.append((xL + j * p, yb if j % 2 == 1 else yt))
    pts.append((xR, yb if abs(pts[-1][1] - yt) < 1e-12 else yt))
    r = math.radians(rot_deg)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    return (R @ np.array(pts).T).T + np.array([cx, cy])


def strip_polys(P, w):
    polys = []
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        dv = b - a; L = np.hypot(*dv)
        if L < 1e-9:
            continue
        nrm = np.array([-dv[1], dv[0]]) / L * (w / 2)
        polys.append(np.array([a + nrm, b + nrm, b - nrm, a - nrm]))
    return polys


COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#e377c2", "#17becf"]


def draw_layout(d, cat, s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    rec = json.load(open(os.path.join(d, "record.json")))
    seq = json.load(open(os.path.join(d, "sequence.json")))
    nested = cat.startswith("C3")
    K = len(rec["was"])
    if nested:   # one panel per phase + labels listed at the side
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6))
    else:
        fig, ax = plt.subplots(figsize=(7.4, 8.0)); axes = [ax]
    for ax in axes:
        ax.add_patch(Rectangle((-PANEL_HALF, -PANEL_HALF), 2 * PANEL_HALF, 2 * PANEL_HALF,
                               fill=False, lw=1.6, ec="k"))
        ax.set_xlim(-PANEL_HALF - 12, PANEL_HALF + 12); ax.set_ylim(-PANEL_HALF - 12, PANEL_HALF + 12)
        ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    for k, (wa, tp) in enumerate(zip(rec["was"], seq["toolpaths"])):
        c = COLORS[k % len(COLORS)]
        op = tp["operation"]
        ax = axes[wa.get("phase", 1) - 1] if nested else axes[0]
        P = zigzag_polyline(op["lv_mm"], op["alpha_deg"], op["n_strips"], op["rotation_deg"],
                            op["center_uv_mm"][0], op["center_uv_mm"][1])
        for poly in strip_polys(P, op["width_mm"]):
            ax.add_patch(Polygon(poly, closed=True, fc=c, ec="none", alpha=0.18 if nested else 0.30))
        ax.plot(P[:, 0], P[:, 1], color=c, lw=0.9)
        ax.plot(P[0, 0], P[0, 1], "o", color=c, ms=5)
        ax.add_patch(Rectangle((wa["cx"] - wa["wx"] / 2, wa["cy"] - wa["wy"] / 2), wa["wx"], wa["wy"],
                               fill=False, ec=c, lw=1.0, ls="--"))
        lab = f"z{k+1} × {wa['n']}"
        sub = f"rot{int(op['rotation_deg'])}  N={op['n_strips']}  lv={op['lv_mm']:.0f}  α={op['alpha_deg']:.1f}°"
        if nested:   # concentric: label at the top edge of each square, right-aligned text column
            top = wa["cy"] + wa["wy"] / 2
            ax.text(-wa["wx"] / 2 + 2, top - 2, f"{lab}  ({sub})", color=c, ha="left", va="top",
                    fontsize=8, fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1))
        else:
            top = wa["cy"] + wa["wy"] / 2
            if top + 22 <= PANEL_HALF + 10:
                y1, y2, va = top + 3, top + 13, "bottom"
            else:
                bot = wa["cy"] - wa["wy"] / 2
                y1, y2, va = bot - 3, bot - 13, "top"
            ax.text(wa["cx"], y1, lab, color=c, ha="center", va=va, fontsize=11, fontweight="bold")
            ax.text(wa["cx"], y2, sub, color=c, ha="center", va=va, fontsize=7)
    if nested:
        axes[0].set_title(f"{cat}/{s} — phase 1 (concentric shrinking squares)", fontsize=10)
        axes[1].set_title(f"{cat}/{s} — phase 2 (cross wheel: rotated 90°, executed after phase 1)",
                          fontsize=10)
    else:
        axes[0].set_title(f"{cat}/{s} — zigzag layout on 300×300 panel (top view)", fontsize=11)
    fig.text(0.5, 0.012,
             "z_k = k-th toolpath in execution order, × n = repeats (hits) · dot = start point (left-bottom-up) · "
             "shaded band = painted footprint (w = 10 mm, to scale) ·\ndashed box = WA bbox · "
             f"gtop {GTOP}, ortho {ORTHO}, center_peak profile, voce hardening",
             ha="center", va="bottom", fontsize=7.5, color="0.3")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(os.path.join(d, "zigzag_layout.png"), dpi=130)
    plt.close(fig)


def render_cycles_fixed(d, cat, s):
    """Per-cycle 3D surface + top-view heatmap with ONE fixed z scale (3D z-limits
    and colormap range) for the whole trajectory; x/y axes labelled to match the
    layout figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    rec = json.load(open(os.path.join(d, "record.json")))
    files = cycle_files(d)
    frames = [(lab, *read_vtp(f)) for lab, f in files]
    zall = np.concatenate([P[:, 2] for _, P, _ in frames]) * 1e3
    zmin, zmax = float(zall.min()), float(zall.max())
    pad = max(0.05 * (zmax - zmin), 0.5)
    zlo, zhi = zmin - pad, zmax + pad
    n = len(frames); ncols = min(n, 5); nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(4.3 * ncols, 7.4 * nrows))
    for i, (lab, P, F) in enumerate(frames):
        x, y, z = P[:, 0] * 1e3, P[:, 1] * 1e3, P[:, 2] * 1e3
        row = i // ncols; col = i % ncols
        ax = fig.add_subplot(nrows * 2, ncols, row * 2 * ncols + col + 1, projection="3d")
        tri = mtri.Triangulation(x, y, F)
        ax.plot_trisurf(tri, z, cmap="viridis", vmin=zlo, vmax=zhi, linewidth=0.05, antialiased=True)
        ax.set_xlim(-150, 150); ax.set_ylim(-150, 150); ax.set_zlim(zlo, zhi)
        ax.set_box_aspect((1, 1, 0.5))
        ax.set_xticks([-150, 0, 150]); ax.set_yticks([-150, 0, 150])
        ax.set_xlabel("x (mm)", fontsize=8, labelpad=-2); ax.set_ylabel("y (mm)", fontsize=8, labelpad=-2)
        ax.set_zlabel("z (mm)", fontsize=8, labelpad=-2)
        ax.tick_params(labelsize=6, pad=-1)
        ax.view_init(24, -60)
        ax.set_title(f"{lab}\nz-span {np.ptp(z):.2f} mm", fontsize=9)
        ax2 = fig.add_subplot(nrows * 2, ncols, row * 2 * ncols + ncols + col + 1)
        sc = ax2.scatter(x, y, c=z, s=4, cmap="viridis", vmin=zlo, vmax=zhi)
        plt.colorbar(sc, ax=ax2, label="z (mm)", fraction=0.046)
        ax2.set_aspect("equal"); ax2.set_xlim(-155, 155); ax2.set_ylim(-155, 155)
        ax2.set_xticks([-150, 0, 150]); ax2.set_yticks([-150, 0, 150]); ax2.tick_params(labelsize=7)
        ax2.set_xlabel("x (mm)", fontsize=8); ax2.set_ylabel("y (mm)", fontsize=8)
        ax2.set_title("top view (fixed color scale)", fontsize=8)
    lines = [f"WA{k}: rot{wa['rot']}  N={wa['N']}  lv={dv['lv']:.0f}mm  alpha={dv['alpha']:.2f}deg  w=10  "
             f"center=({wa['cx']:.0f},{wa['cy']:.0f})  bbox {wa['wx']:.0f}x{wa['wy']:.0f}  repeat x{wa['n']}"
             + (f"  [phase {wa['phase']}]" if "phase" in wa else "")
             for k, (wa, dv) in enumerate(zip(rec["was"], rec["derived"]))]
    exag = 0.5 * 300.0 / max(zhi - zlo, 1e-9)
    fig.suptitle(f"{cat}/{s}  [gtop={GTOP} gbot=0 ortho={ORTHO} profile=center_peak voce-hardening]   "
                 f"fixed z-scale [{zlo:.1f}, {zhi:.1f}] mm for all frames (3D z exaggeration ≈ {exag:.1f}×)\n"
                 + "\n".join(lines), fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.012 * (len(lines) + 2)))
    fig.savefig(os.path.join(d, "deformation_cycles.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


def cmd_figures():
    outdir = os.path.join(ROOT, BATCH)
    for cat, s, d in case_dirs(outdir):
        if not os.path.exists(os.path.join(d, "zigzag_layout.png")):
            draw_layout(d, cat, s)
        if glob.glob(os.path.join(d, "*_final.vtp")) and not os.path.exists(os.path.join(d, "deformation_cycles.png")):
            render_cycles_fixed(d, cat, s)
        print(f"{cat}/{s} figures done", flush=True)


# ---- analysis ----------------------------------------------------------------
def bbox_mask(P, wa, pad=0.0):
    x, y = P[:, 0] * 1e3, P[:, 1] * 1e3
    return ((np.abs(x - wa["cx"]) <= wa["wx"] / 2 + pad) & (np.abs(y - wa["cy"]) <= wa["wy"] / 2 + pad))


def profile(P, z, axis, half=8.0, other_lim=25.0):
    """median z along `axis` (0=x,1=y) in a slab |other|<other_lim, 5-mm bins."""
    a = P[:, axis] * 1e3; o = P[:, 1 - axis] * 1e3
    slab = np.abs(o) < other_lim
    edges = np.arange(-145, 146, 5.0); xc = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(a[slab], edges) - 1
    zz = z[slab]
    prof = np.array([np.median(zz[idx == i]) if (idx == i).sum() > 2 else np.nan for i in range(len(xc))])
    return xc, prof


def axis_check(P, z):
    """Which axis carries the curvature? Returns (range along x-profile, along y-profile)."""
    _, px = profile(P, z, 0); _, py = profile(P, z, 1)
    return float(np.nanmax(px) - np.nanmin(px)), float(np.nanmax(py) - np.nanmin(py))


def analyze_case(d, rec):
    finals = sorted(glob.glob(os.path.join(d, "*_final.vtp")))
    if len(finals) != rec["T"]:
        return dict(id=rec["id"], category=rec["category"], status=f"incomplete ({len(finals)}/{rec['T']})")
    P0, _ = read_vtp(finals[0])
    Z = [read_vtp(f)[0][:, 2] * 1e3 for f in finals]
    was = rec["was"]
    out = dict(id=rec["id"], category=rec["category"], status="ok",
               zspan_final=round(float(np.ptp(Z[-1])), 2),
               zspan_by_cycle=[round(float(np.ptp(z)), 2) for z in Z], wa_metrics=[])
    far = np.ones(len(P0), bool)
    for wa in was:
        far &= ~bbox_mask(P0, wa, pad=25.0)
    c0 = 0
    for k, wa in enumerate(was):
        c1 = c0 + wa["n"]
        m = bbox_mask(P0, wa)
        dz = Z[c1 - 1] - (np.zeros(len(P0)) if c0 == 0 else Z[c0 - 1])
        loc = float(np.abs(dz[m]).mean() / (np.abs(dz[far]).mean() + 1e-9)) if far.sum() > 10 else float("nan")
        out["wa_metrics"].append(dict(wa=k, cycles=[c0 + 1, c1], n=wa["n"],
                                      dz_in_mean=round(float(dz[m].mean()), 3),
                                      dz_in_absmean=round(float(np.abs(dz[m]).mean()), 3),
                                      localization=round(loc, 2) if loc == loc else None))
        c0 = c1
    # principal-axis check on the FIRST WA's own deformation (what the professor asked about)
    rx, ry = axis_check(P0, Z[was[0]["n"] - 1])
    out["axis_check_after_WA0"] = dict(rot=was[0]["rot"], range_along_x=round(rx, 2),
                                       range_along_y=round(ry, 2),
                                       curvature_axis="x" if rx > ry else "y")
    if rec["category"] == "C2":
        xc, py = profile(P0, Z[-1], 1, other_lim=30.0)
        at = lambda q: float(np.interp(q, xc, np.nan_to_num(py, nan=0.0)))
        out["c2_yprofile"] = dict(y_m135=round(at(-135), 2), y_m120=round(at(-120), 2), y_0=round(at(0), 2),
                                  y_p120=round(at(120), 2), y_p135=round(at(135), 2))
    if rec["category"] == "C4":
        def side_yarch(z, wa):
            x, y = P0[:, 0] * 1e3, P0[:, 1] * 1e3
            m = np.abs(x - wa["cx"]) <= wa["wx"] / 2
            yy, zz = y[m], z[m]
            yl = wa["wy"] / 2
            c = np.median(zz[np.abs(yy - wa["cy"]) < 0.25 * yl])
            e = np.median(zz[np.abs(np.abs(yy - wa["cy"]) - yl) < 0.25 * yl])
            return float(c - e)                       # >0: locally arched along y at that side
        za, zf = Z[was[0]["n"] - 1], Z[-1]
        xc, px = profile(P0, zf, 0, other_lim=30.0)
        at = lambda q: float(np.interp(q, xc, np.nan_to_num(px, nan=0.0)))
        out["c4_check"] = dict(
            xprofile={"x-135": round(at(-135), 2), "x-120": round(at(-120), 2), "x0": round(at(0), 2),
                      "x+120": round(at(120), 2), "x+135": round(at(135), 2)},
            side_yarch_after_WA0=[round(side_yarch(za, was[1]), 2), round(side_yarch(za, was[2]), 2)],
            side_yarch_final=[round(side_yarch(zf, was[1]), 2), round(side_yarch(zf, was[2]), 2)])
    if rec["category"].startswith("C3"):
        # crown check: center vs edges along both axes, after phase 1 and at the end
        n1 = sum(w["n"] for w in was if w.get("phase", 1) == 1)
        res = {}
        for tag, z in (("after_phase1", Z[n1 - 1]), ("final", Z[-1])):
            xc, px = profile(P0, z, 0); yc, py = profile(P0, z, 1)
            fx = lambda q: float(np.interp(q, xc, np.nan_to_num(px, nan=0.0)))
            fy = lambda q: float(np.interp(q, yc, np.nan_to_num(py, nan=0.0)))
            ax_ = fx(0) - 0.5 * (fx(-140) + fx(140)); ay_ = fy(0) - 0.5 * (fy(-140) + fy(140))
            res[tag] = dict(arch_x=round(ax_, 2), arch_y=round(ay_, 2),
                            evenness=round(min(abs(ax_), abs(ay_)) / (max(abs(ax_), abs(ay_)) + 1e-9), 2),
                            same_sign=bool(ax_ * ay_ > 0))
        out["c3_crown"] = res
    return out


def montage(outdir, man, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    for cat in CATS:
        ms = [m for m in man["cases"] if m["category"] == cat]
        fig, axes = plt.subplots(2, 5, figsize=(22, 9.6))
        for ax, m in zip(axes.ravel(), ms):
            d = os.path.join(outdir, m["dir"])
            finals = sorted(glob.glob(os.path.join(d, "*_final.vtp")))
            rec = json.load(open(os.path.join(d, "record.json")))
            if not finals:
                ax.set_title(f"{m['id']}: no output"); ax.axis("off"); continue
            P, _ = read_vtp(finals[-1])
            sc = ax.scatter(P[:, 0] * 1e3, P[:, 1] * 1e3, c=P[:, 2] * 1e3, s=3, cmap="viridis")
            plt.colorbar(sc, ax=ax, fraction=0.046, label="z (mm)")
            for k, wa in enumerate(rec["was"]):
                ax.add_patch(Rectangle((wa["cx"] - wa["wx"] / 2, wa["cy"] - wa["wy"] / 2), wa["wx"], wa["wy"],
                                       fill=False, ec=COLORS[k % len(COLORS)], lw=1.0, ls="--"))
            r = next(x for x in results if x["id"] == m["id"])
            extra = ""
            if "c3_crown" in r:
                extra = f"  even {r['c3_crown']['final']['evenness']:.2f}" + ("" if r['c3_crown']['final']['same_sign'] else " (opp)")
            ax.set_title(f"{m['id']}  span {r.get('zspan_final', 0):.1f}mm  T={m['T']}{extra}", fontsize=8)
            ax.set_aspect("equal"); ax.set_xlim(-152, 152); ax.set_ylim(-152, 152)
            ax.set_xticks([-150, 0, 150]); ax.set_yticks([-150, 0, 150]); ax.tick_params(labelsize=6)
        fig.suptitle(f"{cat}: final z (top view, x right / y up), dashed = WA bboxes, ortho={ORTHO}")
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f"summary_{cat}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)


def cmd_analyze():
    outdir = os.path.join(ROOT, BATCH)
    man = json.load(open(os.path.join(outdir, "manifest.json")))
    results = []
    for m in man["cases"]:
        d = os.path.join(outdir, m["dir"])
        results.append(analyze_case(d, json.load(open(os.path.join(d, "record.json")))))
        print(f"{m['id']}: {results[-1]['status']}", flush=True)
    json.dump(results, open(os.path.join(outdir, "analysis.json"), "w"), indent=1)
    montage(outdir, man, results)
    cmd_figures()
    for cat in CATS:
        rs = [r for r in results if r["category"] == cat and r["status"] == "ok"]
        if not rs:
            print(f"{cat}: 0 ok"); continue
        spans = [r["zspan_final"] for r in rs]
        axes = [r["axis_check_after_WA0"] for r in rs]
        print(f"{cat}: {len(rs)}/10 ok; zspan med {np.median(spans):.1f}mm; "
              f"curvature axis after WA0: " +
              ", ".join(f"rot{a['rot']}->{a['curvature_axis']}" for a in axes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--seed", type=int, default=0); s.add_argument("--cats", nargs="*")
    r = sub.add_parser("run"); r.add_argument("--workers", type=int, default=20); r.add_argument("--cores", type=int, default=8)
    sub.add_parser("analyze"); sub.add_parser("figures")
    a = ap.parse_args()
    {"sample": lambda: cmd_sample(a.seed, cats=a.cats), "run": lambda: cmd_run(a.workers, a.cores),
     "analyze": cmd_analyze, "figures": cmd_figures}[a.cmd]()
