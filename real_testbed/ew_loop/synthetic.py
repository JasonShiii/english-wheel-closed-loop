"""A fake camera rig for offline tests: what one rig camera would return for a given sheet
surface and robot position, with the disturbances seen in the real captures
(bridge_checks/inplace_cloud_noise_probe.py): ~450 mm viewing distance, ~30 deg tilt, depth
noise ~0.8 mm, the upper wheel standing on the sheet and hiding part of it, the gripper bar
over the gripped edge, mixed pixels behind the sheet edges, another object in range, floor
far away.

The sheet surface is given as z(x, y) over the sheet frame (projected coordinates, mm); NaN
means no sheet there. The robot position is given as the sheet point under the wheel.
"""
from dataclasses import dataclass

import numpy as np

from .frames import Sheet


@dataclass
class FakeRig:
    sheet: Sheet
    distance_mm: float = 450.0
    tilt_deg: float = 30.0          # sheet normal vs the line of sight (real rig: ~29 deg)
    inplane_deg: float = 270.0      # feed axis in the plane; 270 puts the camera over the free part
    up_sign: float = 1.0            # +1: the sheet's +z faces the camera
    noise_mm: float = 0.8
    wheel_radius_mm: float = 120.0
    wheel_width_mm: float = 60.0
    spacing_mm: float = 0.6          # ~10 points per 2 mm cell, like the real camera at 450 mm
    flying_frac: float = 0.3        # of the sheet-edge samples
    seed: int = 0

    def __post_init__(self):
        t = np.radians(self.tilt_deg)
        self.n = np.array([0.0, -np.sin(t), -np.cos(t)])                     # toward the camera
        a = np.array([1.0, 0.0, 0.0])
        self.e1 = a - (a @ self.n) * self.n; self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(self.n, self.e1)
        c = np.radians(self.inplane_deg)
        yh = np.array([np.cos(c), np.sin(c)])
        xh = self.up_sign * np.array([yh[1], -yh[0]])
        self.R2 = np.column_stack([xh, yh])
        self.t2_ref = np.array([10.0, 5.0])
        centre = np.array([20.0, -30.0, self.distance_mm])                     # near the optical axis
        self.origin = centre - self.t2_ref[0] * self.e1 - self.t2_ref[1] * self.e2
        self.contact_ref = np.array([0.0, self.sheet.wheel_line_y])
        self.rng = np.random.default_rng(self.seed)

    # ---- frames -------------------------------------------------------------------------
    def feed_dir_cam(self, error_deg=0.0):
        """the operator's rough feed direction in the camera frame (true one rotated in-plane)."""
        c = np.radians(error_deg)
        y2 = np.array([[np.cos(c), -np.sin(c)], [np.sin(c), np.cos(c)]]) @ self.R2[:, 1]
        return y2[0] * self.e1 + y2[1] * self.e2

    def t2_for(self, contact_xy):
        return self.t2_ref - self.R2 @ (np.asarray(contact_xy, float) - self.contact_ref)

    def to_cam(self, x, y, z, t2):
        uv = np.stack([x, y], 1) @ self.R2.T + t2
        return self.origin + uv[:, :1] * self.e1 + uv[:, 1:2] * self.e2 + (self.up_sign * z)[:, None] * self.n

    # ---- capture ------------------------------------------------------------------------
    def capture(self, surface, contact_xy):
        s, sp, rng = self.sheet, self.spacing_mm, self.rng
        t2 = self.t2_for(contact_xy)
        xs = np.arange(-s.width_mm / 2 + sp / 2, s.width_mm / 2, sp)
        ys = np.arange(-s.length_mm / 2 + sp / 2, s.length_mm / 2, sp)
        X, Y = np.meshgrid(xs, ys)
        Z = surface(X, Y)
        ok = np.isfinite(Z)
        edge = ok & ~(np.roll(ok, 1, 0) & np.roll(ok, -1, 0) & np.roll(ok, 1, 1) & np.roll(ok, -1, 1))
        edge[[0, -1], :] = ok[[0, -1], :]; edge[:, [0, -1]] = ok[:, [0, -1]]
        gripper = Y < -s.length_mm / 2 + 15.0                          # sheet under the gripper bar
        x, y, z = X[ok & ~gripper], Y[ok & ~gripper], Z[ok & ~gripper]
        P = self.to_cam(x, y, z, t2)
        # upper wheel: resting on the sheet point under the wheel, axis along the sheet x axis
        cx, cy = contact_xy
        zc = float(np.nan_to_num(surface(np.array([cx]), np.array([cy]))[0]))
        Pc = self.to_cam(np.array([cx]), np.array([cy]), np.array([zc]), t2)[0]
        C = Pc + self.wheel_radius_mm * self.n
        ax = self.R2[0, 0] * self.e1 + self.R2[1, 0] * self.e2
        P = P[~self._occluded(P, C, ax)]
        W = self._wheel_points(C, ax)
        # gripper bar top face, 25 mm above the sheet, moving with it
        gx, gy = np.meshgrid(np.arange(-s.width_mm / 2 - 5, s.width_mm / 2 + 5, sp),
                             np.arange(-s.length_mm / 2 - 10, -s.length_mm / 2 + 15, sp))
        G = self.to_cam(gx.ravel(), gy.ravel(), np.full(gx.size, 25.0 * self.up_sign), t2)
        # mixed pixels behind the sheet edges (between the sheet and the background)
        E = self.to_cam(X[edge], Y[edge], Z[edge], t2)
        E = E[rng.random(len(E)) < self.flying_frac]
        E = E + (rng.uniform(5.0, 300.0, len(E)) / np.linalg.norm(E, axis=1))[:, None] * E
        # another object in range beside the sheet (like the wheel stand), and the floor
        ox, oy = np.meshgrid(np.arange(160, 260, 2.0), np.arange(-100, 0, 2.0))
        O = self.origin + ox.ravel()[:, None] * self.e1 + oy.ravel()[:, None] * self.e2 - 45.0 * self.n
        fx, fy = np.meshgrid(np.arange(-600, 600, 8.0), np.arange(-600, 600, 8.0))
        F = np.stack([fx.ravel(), fy.ravel(), np.full(fx.size, 1300.0)], 1)
        out = np.concatenate([P, W, G, E, O, F])
        rays = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out + rng.normal(0.0, self.noise_mm, len(out))[:, None] * rays

    def _occluded(self, P, C, ax):
        """sheet points whose line of sight to the camera passes through the wheel cylinder."""
        dist = np.linalg.norm(P, axis=1)
        d = -P / dist[:, None]
        w = P - C
        wp = w - (w @ ax)[:, None] * ax
        dp = d - (d @ ax)[:, None] * ax
        a, b, c = (dp * dp).sum(1), 2 * (wp * dp).sum(1), (wp * wp).sum(1) - self.wheel_radius_mm ** 2
        disc = b * b - 4 * a * c
        hit = disc > 0
        sq = np.sqrt(np.where(hit, disc, 0.0))
        s1, s2 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
        sa, sb = np.maximum(s1, 0.0), np.minimum(s2, dist)                # between the point and the camera
        hit &= sb > sa
        ax_a, ax_b = (w @ ax) + sa * (d @ ax), (w @ ax) + sb * (d @ ax)
        half = self.wheel_width_mm / 2
        return hit & ~(((ax_a > half) & (ax_b > half)) | ((ax_a < -half) & (ax_b < -half)))

    def _wheel_points(self, C, ax):
        b1 = self.n
        b2 = np.cross(ax, b1)
        ph, al = np.meshgrid(np.linspace(0, 2 * np.pi, 720, endpoint=False),
                             np.arange(-self.wheel_width_mm / 2, self.wheel_width_mm / 2 + 1e-9, 2.0))
        nrm = np.cos(ph.ravel())[:, None] * b1 + np.sin(ph.ravel())[:, None] * b2
        Q = C + self.wheel_radius_mm * nrm + al.ravel()[:, None] * ax
        return Q[(nrm * (-Q)).sum(1) > 0]                              # the side facing the camera


def bent_surface(sheet, amp_mm=12.0):
    """the free part curls away from the wheel line, a little more at the sides."""
    def z(X, Y):
        t = np.clip((Y - sheet.wheel_line_y) / (sheet.length_mm / 2 - sheet.wheel_line_y), 0.0, None)
        return amp_mm * t ** 2 * (1.0 + 0.3 * (2.0 * X / sheet.width_mm) ** 2)
    return z


def flat_surface(X, Y):
    return np.zeros_like(X, dtype=float)
