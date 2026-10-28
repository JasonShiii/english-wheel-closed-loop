"""Model image renderer for the new-sheet model (training AND deployment must use this one function).
A HeightField (2 mm grid, NaN = no data) -> PNG of the MODEL FRAME: the part of the sheet above the crop line (everything
the model may see) plus a MARGIN_MM border, rasterised at PX_MM per pixel and saved OUT_W_PX wide with the aspect ratio
kept (2026-10-20: the frame is no longer a fixed 300 x 300 mm square with the hidden strip painted white; the pixels go
to the observed region). viridis normalised to this image's own valid min/max, WHITE wherever the field has no data:
outside the sheet and every cell the cameras did not see. Nothing is interpolated across gaps: a pixel is drawn only if
the field cell it falls in has data (nearest-cell lookup, no linear fill). This differs from ew_loop/frames.field_png
(2026-10-06), which draws measured fields with griddata and therefore fills the convex hull of the data."""
import numpy as np
import cfg
MARGIN_MM, PX_MM, OUT_W_PX, DPI = 4.0, 1.0, 448, 140


def frame():
    """(x_lo, x_hi, y_lo, y_hi) in sheet mm: the sheet above the crop line plus the margin. Stated in the prompt."""
    s = cfg.SHEET
    return (-s.width_mm / 2 - MARGIN_MM, s.width_mm / 2 + MARGIN_MM, cfg.crop_y() - MARGIN_MM, s.length_mm / 2 + MARGIN_MM)


def field_to_raster(field):
    x0, x1, y0, y1 = frame()
    gx = np.arange(x0 + PX_MM / 2, x1, PX_MM); gy = np.arange(y0 + PX_MM / 2, y1, PX_MM); X, Y = np.meshgrid(gx, gy)
    dx, dy = field.x_mm[1] - field.x_mm[0], field.y_mm[1] - field.y_mm[0]
    xi = np.round((X - field.x_mm[0]) / dx).astype(int); yi = np.round((Y - field.y_mm[0]) / dy).astype(int)
    # a pixel belongs to the sheet only if it lies within half a cell of a grid node (clean straight edges)
    ok = (X >= field.x_mm[0] - dx / 2) & (X <= field.x_mm[-1] + dx / 2) & (Y >= field.y_mm[0] - dy / 2) & (Y <= field.y_mm[-1] + dy / 2)
    ok &= (xi >= 0) & (xi < len(field.x_mm)) & (yi >= 0) & (yi < len(field.y_mm))
    H = np.full(X.shape, np.nan, np.float32)
    H[ok] = field.z_mm[yi[ok], xi[ok]]
    return H


def field_png(field, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    H = field_to_raster(field); x0, x1, y0, y1 = frame()
    if np.isfinite(H).sum() < 3:
        raise ValueError("field has fewer than 3 cells with data")
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad(color="white")
    w_in = OUT_W_PX / DPI; fig = plt.figure(figsize=(w_in, w_in * (y1 - y0) / (x1 - x0)), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(np.ma.masked_invalid(H), origin="lower", cmap=cmap, extent=(x0, x1, y0, y1))
    fig.savefig(path); plt.close(fig)
    return H
