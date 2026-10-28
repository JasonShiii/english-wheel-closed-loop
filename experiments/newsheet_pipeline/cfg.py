"""Single configuration of the new-sheet training pipeline (2026-10-06; geometry updated 2026-10-20 to the new arm and gripper).
Sheet, pads, reachable zone and crop line follow the user's check with Danish (figure of 2026-10-19, UR5 with force control,
wheel force 300 N): sheet 231 x 174 x 0.6 mm; pads 13 x 18 mm at the two gripped corners; wheel-reachable zone 144.2 x 142 mm
(43.4 mm side margins, 12 mm top margin, bottom edge = the crop line 20 mm from the gripped edge); everything above the crop
line is shown to the model, the margins deform passively.
Growth: placeholder inside the collaborator's prior (the only real run so far, 2026-10-08, had a slipping sheet and no flat
baseline); replace by the fit on Danish's runs of 2026-10-21 (flat baseline; 8 legs x 5 passes, 11 legs x 3 passes).
One gripper is now compliant in translation (Danish, 2026-10-15); the simulator's BC mode has only fully fixed pads, so both
pads are fixed here and the difference goes into the growth calibration."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(ROOT, "real_testbed"), os.path.join(ROOT, "experiments", "category_tests"),
          os.path.join(ROOT, "experiments", "mpc_end2end"), os.path.join(ROOT, "experiments", "newsheet_pilot")):
    if p not in sys.path:
        sys.path.insert(0, p)
from ew_loop.frames import Sheet

# clamp_mm = gripped edge -> bottom of the wheel-reachable zone (= the crop line on this setup); margin_side_mm = the
# 43.4 mm side strips the wheel cannot reach; pads 13 mm wide at the very corners (offsets = pad centres from the left edge).
# NOTE: real_testbed/ew_loop/frames.Sheet DEFAULTS still describe the previous setup (195 x 190, band 100); the
# deployment side takes these values when it moves to the new arm.
SHEET = Sheet(width_mm=231.0, length_mm=174.0, thickness_mm=0.6, clamp_mm=20.0, margin_side_mm=43.4, margin_bottom_mm=0.0,
              pad_offsets_from_left_mm=(6.5, 224.5), pad_diameter_mm=13.0)
MARGIN_TOP_MM = 12.0                              # far-edge strip the wheel cannot reach
WHEEL_FORCE_N = 300.0                             # Danish's setting under force control (informational: the growth is tied to it)
DOSE = dict(gtop=9e-4, gbot=1e-4, ortho=-0.5)     # gtop CALIBRATED 2026-10-22 on Danish's runs of 10-21/22 with the FREE plate (0.9-1.05e-3, dose_fit/DOSE_FIT.md)
BC = "free"                                       # "free": no clamp (the compliant gripper only holds the sheet in space; calibrated 2026-10-22)
                                                  # "pads": two fully fixed pads (zigzag_sequence_BC) -- WRONG for the real gripper, kept for comparison
TOL = "3e-13"                                     # clamped solver: reproducible to ~0.8 mm, 2x faster than 3e-14
PAD_EDGE_GAP_MM = 0.0                             # pads start at the gripped edge (figure 2026-10-19)
PAD_SIZE_MM = (13.0, 18.0)                        # pad contact rectangle: 13 mm across x, 18 mm along the feed. BC v1 needs exactly two.
RES = "0.02"                                      # 1/(nodes per boundary): ~4.6 mm vertex spacing on the 231 mm side
STEP_MM = 2.0                                     # height-field grid
TAU_PAIR_MM = 1.0                                 # pair filter (max |dz| between cut and target); shallow regime
SCAN_POSITION = True                              # captures are taken with the sheet moved clear of the upper wheel (protocol
                                                  # decision 2026-10-12): the fake camera then has no wheel occlusion and the
                                                  # in-place occlusion strip is only a rare augmentation
CROP_MM = 20.0                                    # model observation region starts here (from the gripped edge) = what the
                                                  # gripper hides; on this setup it coincides with the bottom of the reachable zone


def crop_y():
    """sheet-frame y of the crop line (below it the model sees white, in simulation and on the machine alike)"""
    return -SHEET.length_mm / 2 + CROP_MM
BIN = os.path.join(ROOT, "PSM2025_recurring", "bin", "shell")


# A second wheel-reachable rectangle, (x_lo, x_hi, y_lo, y_hi) in sheet mm; None = single zone.
REACHABLE_EXTRA = None


def reachable_zones():
    """list of reachable rectangles; a work area is legal if it lies inside ANY of them"""
    return [reachable_rect()] + ([tuple(REACHABLE_EXTRA)] if REACHABLE_EXTRA else [])


def reachable_rect():
    """(x_lo, x_hi, y_lo, y_hi) in sheet/material mm: side margins, bottom of the zone (clamp_mm from the gripped edge
    + margin_bottom), top margin. 2026-10-19 figure: x +-72.1, y -67..75 (144.2 x 142 mm)."""
    s = SHEET
    return (-s.width_mm / 2 + s.margin_side_mm, s.width_mm / 2 - s.margin_side_mm,
            s.wheel_line_y + s.margin_bottom_mm, s.length_mm / 2 - MARGIN_TOP_MM)


def bc_block():
    return {"enabled": True, "coordinate_system": "material_uv", "application": "persistent",
            "release_after_final_cycle": True,
            "regions": [{"name": f"pad_{i}", "shape": "rectangle",
                         "center_uv_mm": [round(cx, 2), round(-SHEET.length_mm / 2 + PAD_EDGE_GAP_MM + PAD_SIZE_MM[1] / 2, 2)],
                         "size_uv_mm": list(PAD_SIZE_MM), "rotation_deg": 0.0}
                        for i, (cx, cy) in enumerate(SHEET.pad_centres())]}


def sim_cmd(cores_start, cores_n, bc=None):
    bc = bc or BC
    return ["taskset", "-c", f"{cores_start}-{cores_start+cores_n-1}", BIN, "-sim", "bilayer_growth", "-case", "custom",
            "-geometry", "rectangle", "-lx", f"{SHEET.width_mm/2000:.5f}", "-ly", f"{SHEET.length_mm/2000:.5f}",   # HALF-lengths
            "-res", RES, "-h_total", f"{SHEET.thickness_mm/1000:.4f}",
            "-growth_type", "zigzag_sequence_BC" if bc == "pads" else "zigzag_sequence", "-cycle_file", "sequence.json",
            "-enable_passE", "false", "-nsteps", "1", "-export_stl", "false", "-tol", TOL]


def pack_cycles(d, bc=None):
    """cycle outputs -> cycles.npz (xyz in mm: initial + state after each cycle); pads: + the released state"""
    bc = bc or BC
    if bc == "pads":
        from newsim_bc_compare import pack_cycles_bc
        return pack_cycles_bc(d)
    import glob, numpy as np, category_tests as ct
    init = sorted(glob.glob(os.path.join(d, "*_initial.vtp"))); finals = sorted(glob.glob(os.path.join(d, "*_final.vtp")))
    if not init or not finals:
        return 0
    XYZ = np.stack([ct.read_vtp(f)[0] for f in init[:1] + finals]).astype(np.float32) * 1e3
    np.savez_compressed(os.path.join(d, "cycles.npz"), xyz=XYZ, labels=np.array(["initial"] + [os.path.basename(f) for f in finals]))
    for f in glob.glob(os.path.join(d, "*.vtp")) + glob.glob(os.path.join(d, "*.dat")):
        os.remove(f)
    return len(finals)
