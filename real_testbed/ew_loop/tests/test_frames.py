"""frames.py checks: frame round trip, safe zone = robot formula, field save/load, rendering."""
import os
import sys
import tempfile

import numpy as np

from ew_loop import frames
from ew_loop.frames import Sheet, HeightField

ROBOT_CODE = os.path.join(os.path.dirname(frames.__file__), "..", "Engish_Wheel_T109", "code")
sys.path.insert(0, os.path.abspath(ROBOT_CODE))
import zigzag_planner as zp


def test_frames():
    s = Sheet()
    xy = np.array([[-50.0, -95.0], [0.0, -5.0], [97.5, 95.0]])
    assert np.allclose(s.from_planner(s.to_planner(xy)), xy)
    assert np.allclose(s.to_planner([0.0, s.wheel_line_y]), [0.0, 0.0])          # wheel line = planner Y 0
    assert np.allclose(s.to_planner([0.0, s.length_mm / 2]), [0.0, s.usable_length_mm])
    assert s.safe_zone_planner() == zp.safe_zone_bounds(s.width_mm, s.usable_length_mm, s.margin_side_mm, s.margin_bottom_mm)
    assert Sheet.from_dict(s.to_dict()) == s and Sheet.parse("195x190", 100.0, 1.0) == s
    px = [p[0] for p in s.pad_centres()]
    assert abs(px[0] + 84.8) < 1e-9 and abs(px[1] - 86.65) < 1e-9
    print("[ok] frames: round trip, wheel line, safe zone == robot formula, pads at x", px)


def test_fields():
    from PIL import Image
    s = Sheet()
    xg, yg = frames.grid_for(s, 2.0)
    assert xg[0] == -97.5 and xg[-1] == 97.5 and yg[0] == -95.0 and yg[-1] == 95.0, "grid includes the edges"
    fl = frames.flat_field(s, 2.0)
    assert fl.z_mm.shape == (len(yg), len(xg)) and fl.zrange() == (0.0, 0.0)
    # synthetic cylinder over the sheet, with the clamped band missing (NaN). NB: clamp 100 of
    # 190 mm puts the wheel line at y = +5 (past the sheet centre) — the usable zone is y in [5, 95].
    rng = np.random.default_rng(0)
    P = np.stack([rng.uniform(-97.5, 97.5, 20000), rng.uniform(-95, 95, 20000)], 1)
    z = 0.002 * P[:, 1] ** 2
    keep = P[:, 1] > s.wheel_line_y
    f = frames.field_from_points(P[keep], z[keep], s, 2.0)
    assert np.isnan(f.z_mm[0, :]).all() and np.isfinite(f.z_mm[-3, 10:-10]).all()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "f.npz"); f.save(p); g = HeightField.load(p)
        assert np.allclose(np.nan_to_num(g.z_mm), np.nan_to_num(f.z_mm)) and g.meta["source"] == "points"
        frames.field_png(g, os.path.join(d, "f.png"))
        im = np.asarray(Image.open(os.path.join(d, "f.png")).convert("RGB"))
        n = im.shape[0]
        c = -150 + (np.arange(n) + 0.5) * 300 / n                  # pixel centres, mm (x by column, y by row from the top)
        white = (im == 255).all(-1)
        rows_y = c[::-1]
        assert white[:, c < -97.5 - 3].all() and white[rows_y < s.wheel_line_y - 3, :].all()
        assert not white[(rows_y > s.wheel_line_y + 5) & (rows_y < 90)][:, (c > -80) & (c < 80)].any()
        fl_png = os.path.join(d, "flat300.png")
        frames.field_png(frames.flat_field(Sheet.parse("300x300", 0.0, 0.5), 2.0), fl_png)
        flat = np.asarray(Image.open(fl_png).convert("RGB"))
        assert len(np.unique(flat.reshape(-1, 3), axis=0)) == 1, "full flat 300 x 300 sheet fills the frame"
    print("[ok] fields: node grid with edges, from points (hull/NaN), npz round trip, training renderer (white outside)")


if __name__ == "__main__":
    test_frames(); test_fields(); print("ALL FRAMES TESTS PASSED")
