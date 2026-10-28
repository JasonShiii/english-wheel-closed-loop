"""Sheet geometry, frame conventions and the height-field format for the real-testbed loop.

This is the ONLY module that knows how the frames relate. spec.py, predict.py, the WP2
scan post-processing and the state bookkeeping all import it; nothing else may carry a
frame constant.

Frames (all lengths in mm)
--------------------------
sheet   : origin at the sheet centre, x across the width, y along the feed direction.
          Clamped edge at y = -length/2, free edge at y = +length/2. This is also the
          simulator's material (u, v) frame and the model's image frame (x right, y up).
planner : the robot's zigzag_planner frame (Engish_Wheel_T109/code/zigzag_planner.py):
          origin at the middle of the WHEEL CONTACT LINE (TARGET_POS_9), X across the
          width, Y along the feed, Y in [0, length - clamp].
              X_p = PLANNER_X_SIGN * x          Y_p = y + length/2 - clamp
model   : the trained checkpoint's image frame: a fixed square
          [-MODEL_HALF_MM, +MODEL_HALF_MM]^2 (300 x 300 for ckpt-1984) centred at the
          origin. The sheet frame is embedded at the centre; pixels outside the sheet or
          without data are white.

PLANNER_X_SIGN = +1 is an ASSUMPTION (the robot code mirrors x once in plan_zigzag and
once more in _planner_to_tcp3d; which physical side is +x cannot be read off the code).
Confirm it once on the machine: wheel a single short strip at a known +x offset and
check which side it shows up on in the scan; flip the constant here if needed.
"""
import json
from dataclasses import dataclass, asdict

import numpy as np

PLANNER_X_SIGN = +1.0
MODEL_HALF_MM = 150.0       # ckpt-1984 prompt: "x in [-150, 150] mm, y in [-150, 150] mm" (model_io.heatmap_png)


@dataclass(frozen=True)
class Sheet:
    """Physical sheet + fixturing, in the sheet frame. Defaults = oct_2_demo branch (2026-09-28)."""
    width_mm: float = 195.0          # across the feed direction (robot X_DIM_MM)
    length_mm: float = 190.0         # along the feed direction (robot Y_DIM_MM)
    thickness_mm: float = 1.0
    clamp_mm: float = 100.0          # gripped edge -> wheel contact line at the robot's start pose = the band the
                                     # wheel cannot reach (robot SHEET_BOTTOM_TO_WHEEL_LINE_MM, 100 on oct_2_demo,
                                     # 85 before). NOT the gripper size (pads are 14 mm). Unverified for the 190 mm
                                     # sheet: their comment says it was set on a 300 x 200 sheet -> measure on the machine.
    margin_side_mm: float = 12.0     # robot MARGIN_SIDE_MM: kept clear on the left and right edges
    margin_bottom_mm: float = 0.0    # robot MARGIN_BOTTOM_MM: kept clear on the wheel-line side
    margin_top_mm: float = None      # robot MARGIN_TOP_MM (Dobot planner, 2026-10): kept clear on the far edge; None = margin_side
    # gripper pads (robot zigzag_planner.GRIPPER_1_X_MM / GRIPPER_2_X_MM, offsets from the left edge)
    pad_offsets_from_left_mm: tuple = (12.7, 184.15)
    pad_diameter_mm: float = 13.98

    # ---- derived geometry -------------------------------------------------------
    @property
    def wheel_line_y(self):
        """y (sheet frame) of the wheel contact line = planner Y = 0."""
        return -self.length_mm / 2.0 + self.clamp_mm

    @property
    def usable_length_mm(self):
        """robot Y_DIM_EFFECTIVE_MM: feed span reachable by the wheel."""
        return self.length_mm - self.clamp_mm

    def pad_centres(self):
        """(x, y) of the gripper pad centres in the sheet frame (one radius inside the clamped edge)."""
        r = self.pad_diameter_mm / 2.0
        return [(-self.width_mm / 2.0 + off, -self.length_mm / 2.0 + r)
                for off in self.pad_offsets_from_left_mm]

    def safe_zone_planner(self):
        """(x_min, x_max, y_min, y_max) in the planner frame — identical to zigzag_planner.safe_zone_bounds(
        X_DIM_MM, Y_DIM_MM, MARGIN_SIDE_MM, MARGIN_BOTTOM_MM, MARGIN_TOP_MM, SHEET_BOTTOM_TO_WHEEL_LINE_MM)
        (Dobot planner; the older 4-argument form = margin_top None)."""
        top = self.margin_side_mm if self.margin_top_mm is None else self.margin_top_mm
        return (-self.width_mm / 2.0 + self.margin_side_mm,
                self.width_mm / 2.0 - self.margin_side_mm,
                self.margin_bottom_mm,
                self.usable_length_mm - top)

    def safe_zone_sheet(self):
        x0, x1, y0, y1 = self.safe_zone_planner()
        return (x0, x1, y0 + self.wheel_line_y, y1 + self.wheel_line_y)

    # ---- frame conversions -------------------------------------------------------
    def to_planner(self, xy):
        """sheet (x, y) -> planner (X, Y). xy: (..., 2) array-like, mm."""
        xy = np.asarray(xy, dtype=float)
        out = np.empty_like(xy)
        out[..., 0] = PLANNER_X_SIGN * xy[..., 0]
        out[..., 1] = xy[..., 1] - self.wheel_line_y
        return out

    def from_planner(self, XY):
        XY = np.asarray(XY, dtype=float)
        out = np.empty_like(XY)
        out[..., 0] = XY[..., 0] / PLANNER_X_SIGN
        out[..., 1] = XY[..., 1] + self.wheel_line_y
        return out

    # ---- (de)serialisation -------------------------------------------------------
    def to_dict(self):
        d = asdict(self)
        d["pad_offsets_from_left_mm"] = list(self.pad_offsets_from_left_mm)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "pad_offsets_from_left_mm" in d:
            d["pad_offsets_from_left_mm"] = tuple(d["pad_offsets_from_left_mm"])
        return cls(**d)

    @classmethod
    def parse(cls, size, clamp_mm, thickness_mm, **kw):
        """size = 'WxL' string (e.g. '195x190'), width first (across the feed)."""
        w, l = (float(v) for v in size.lower().split("x"))
        return cls(width_mm=w, length_mm=l, clamp_mm=clamp_mm, thickness_mm=thickness_mm, **kw)


# =====================================================================================
# Height field: the ONE state/target format of the loop (what WP2 produces and the
# model consumes). Regular grid in the sheet frame, z = height above the flat sheet,
# NaN = no data (occluded, outside the sheet, rejected by the post-processing).
# =====================================================================================
@dataclass
class HeightField:
    x_mm: np.ndarray      # (nx,)
    y_mm: np.ndarray      # (ny,)
    z_mm: np.ndarray      # (ny, nx), float32, NaN = no data
    meta: dict
    points_mm: np.ndarray = None   # (n, 3) simulator mesh vertices, only for fields made from a
                                   # simulation: drawn from these exactly as the training images were

    @property
    def valid(self):
        return np.isfinite(self.z_mm)

    def zrange(self):
        v = self.z_mm[self.valid]
        if v.size == 0:
            return (0.0, 0.0)
        return (float(v.min()), float(v.max()))

    def points(self):
        """valid samples as (n, 3) mm — for metrics and for the simulator-style renderer."""
        Y, X = np.meshgrid(self.y_mm, self.x_mm, indexing="ij")
        m = self.valid
        return np.stack([X[m], Y[m], self.z_mm[m]], axis=1)

    def save(self, path):
        extra = {} if self.points_mm is None else {"points_mm": self.points_mm.astype(np.float32)}
        np.savez_compressed(path, x_mm=self.x_mm.astype(np.float32), y_mm=self.y_mm.astype(np.float32),
                            z_mm=self.z_mm.astype(np.float32), meta=json.dumps(self.meta), **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"])) if "meta" in d else {}
        return cls(x_mm=d["x_mm"].astype(float), y_mm=d["y_mm"].astype(float),
                   z_mm=d["z_mm"].astype(np.float32), meta=meta,
                   points_mm=d["points_mm"].astype(float) if "points_mm" in d else None)


def grid_for(sheet, step_mm=2.0):
    """grid nodes covering the sheet INCLUDING its edges (so a full flat sheet renders edge to
    edge, as the simulator meshes did in training); ~step_mm spacing."""
    nx = int(round(sheet.width_mm / step_mm)) + 1
    ny = int(round(sheet.length_mm / step_mm)) + 1
    return (np.linspace(-sheet.width_mm / 2.0, sheet.width_mm / 2.0, nx),
            np.linspace(-sheet.length_mm / 2.0, sheet.length_mm / 2.0, ny))


def flat_field(sheet, step_mm=2.0, meta=None):
    x, y = grid_for(sheet, step_mm)
    return HeightField(x, y, np.zeros((len(y), len(x)), np.float32),
                       {"source": "flat", "sheet": sheet.to_dict(), **(meta or {})})


def field_from_points(P_mm, z_mm, sheet, step_mm=2.0, meta=None):
    """Scattered (x, y) mm + z mm (sheet frame) -> HeightField. Linear interpolation inside
    the convex hull of the samples, NaN elsewhere. Used for simulator meshes (targets) and
    for registered scan points (WP2). Points outside the sheet rectangle are dropped."""
    from scipy.interpolate import griddata
    P_mm = np.asarray(P_mm, dtype=float)[:, :2]
    z_mm = np.asarray(z_mm, dtype=float)
    keep = (np.abs(P_mm[:, 0]) <= sheet.width_mm / 2.0 + step_mm) & \
           (np.abs(P_mm[:, 1]) <= sheet.length_mm / 2.0 + step_mm)
    P_mm, z_mm = P_mm[keep], z_mm[keep]
    x, y = grid_for(sheet, step_mm)
    X, Y = np.meshgrid(x, y)
    Z = griddata(P_mm, z_mm, (X, Y), method="linear")
    return HeightField(x, y, Z.astype(np.float32),
                       {"source": "points", "sheet": sheet.to_dict(), "n_points": int(len(z_mm)),
                        **(meta or {})}, points_mm=np.column_stack([P_mm, z_mm]))


def mask_clamp_band(field, sheet):
    """The band between the gripped edge and the wheel line is never wheeled; its shape is a
    passive consequence of the clamping that neither the cameras (gripper, wheel in the way)
    nor the simulator (idealised clamp) capture reliably. It is set to no-data in EVERY field
    the model sees, the target included, so both sides follow the same rule."""
    z = field.z_mm.copy()
    z[field.y_mm < sheet.wheel_line_y, :] = np.nan
    pts = None if field.points_mm is None else field.points_mm[field.points_mm[:, 1] >= sheet.wheel_line_y]
    return HeightField(field.x_mm, field.y_mm, z, {**field.meta, "clamp_band_masked": True}, points_mm=pts)


def field_png(field, path):
    """Draw a HeightField exactly as the model saw its training inputs (model_io.heatmap_png:
    fixed 300 x 300 mm frame, viridis normalised to this image's own min/max, white = no data).
    A field made from a simulation is drawn from its mesh vertices, like the training images; a
    measured field from its grid nodes. (The model reacts to small differences of the white
    margin: a 1 % pixel difference along it flipped the planned zigzag orientation.)"""
    from .model_io import heatmap_png
    if field.points_mm is not None and len(field.points_mm) >= 3:
        heatmap_png(field.points_mm[:, :2] * 1e-3, field.points_mm[:, 2], path)
        return
    Y, X = np.meshgrid(field.y_mm, field.x_mm, indexing="ij")
    m = field.valid
    if m.sum() < 3:
        raise ValueError("field has fewer than 3 cells with data")
    P = np.stack([X[m], Y[m]], 1) * 1e-3                      # heatmap_png takes metres
    heatmap_png(P, field.z_mm[m].astype(float), path)
