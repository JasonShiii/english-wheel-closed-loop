"""The rig port (2026-10-28): the measurer reproduces bridge_checks/danish_run2_measure.py on the recorded
10-21 run, the model profile renders and prompts exactly as the training code, the history schema matches,
and a toolpath built from a model-style plan is executable. Needs the recorded run (share/Danish_Run_2).
Run: python -m ew_loop.tests.test_rig_port
"""
import json
import os
import sys
import tempfile

import numpy as np

from ew_loop import frames as F, spec, measure as ME, model_newsheet as MN
RT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(RT, "share", "Danish_Run_2", "2026-10-21_04-29-23")
sys.path.insert(0, os.path.join(RT, "bridge_checks"))
import danish_run2_measure as DM          # noqa: E402
import pairs as NP                        # noqa: E402  (newsheet_pipeline, on sys.path through ew_loop.measure)
import render as R                        # noqa: E402
import planners as PL                     # noqa: E402


def reference_chain(run, k):
    """plan_from_capture.measured_field, i.e. the 10-27 chain on DM's own grid"""
    P0, I0, Fr, corr, info = DM.prepare(run)
    xy0, h0, m0 = DM.field_points(P0, I0, Fr, corr); Z0 = DM.grid(xy0[m0], h0[m0])
    sheet = np.isfinite(Z0) & (np.abs(Z0) < DM.FLAT_TOL)
    GX, GY = np.meshgrid(DM.XS, DM.YS); sheet &= ~((np.abs(GX) > DM.W / 2 - 14) & (GY < -15))
    Z = Z0 if k == 0 else DM.grid(*(lambda xy, h, m: (xy[m], h[m]))(*DM.field_points(*DM.load(run, k), Fr, corr)))
    Z, _ = DM.despeckle(np.where(sheet, Z, np.nan))
    return NP.plane_removed(NP.mask_crop(F.HeightField(DM.XS.astype(float), DM.YS.astype(float), Z.astype(np.float32), {})))


def main():
    # 1. the 10-21 run was a 231 x 124 x 1.0 sheet (handoff section 0): measured with that sheet, the measurer must reproduce
    #    the recalibrated chain (danish_run2_measure_231x124.Chain, orientation-searching rectangle, hardware mask 37 mm)
    s124 = F.Sheet(width_mm=231.0, length_mm=124.0, thickness_mm=1.0, clamp_mm=12.0, margin_side_mm=43.4, margin_bottom_mm=0.0)
    m = ME.RigMeasurer(sheet=s124)
    m.reference(RUN)
    b = m.info["beyond_edge_points"]          # counted BEFORE the frame flip: the side depends on the BLAS, one side must dominate
    assert m.info["inside_fraction"] > 0.7 and max(b.values()) > 5 * min(b.values()), m.info
    ch = ME.D124.Chain(231.0, 124.0, band=45.0, hw_mm=37.0, tag="231x124_rig")
    P0, I0, Fr, corr, info = ch.prepare(RUN)
    xy0, h0, m0 = ch.field_points(P0, I0, Fr, corr); Z0 = ch.grid(xy0[m0], h0[m0])
    sheet = np.isfinite(Z0) & (np.abs(Z0) < ME.D124.FLAT_TOL); GX, GY = np.meshgrid(ch.XS, ch.YS)
    sheet &= ~((np.abs(GX) > 231.0 / 2 - 14) & (GY < -124.0 / 2 + 37.0))
    # the frame must be IDENTICAL (same chain); the grids differ by up to 1 mm in x (ours = the targets' nodes spread over the
    # exact width, the chain's = 2 mm steps from the left edge), so masks and fields are compared within tolerance
    for k in ("c", "ex", "ey", "n"):
        assert np.allclose(m.frame[k], Fr[k]), f"frame {k} differs from the recalibrated chain"
    agree = (sheet == m.sheet_mask).mean()
    print(f"[ok] frame == danish_run2_measure_231x124 (rectangle angle vs min-area rect {m.info['rect_angle_vs_minarearect_deg']} deg, "
          f"offset {m.info['rect_offset_mm']}); sheet masks agree on {agree:.3f} of the cells")
    assert agree > 0.97
    for k in (3, 6):
        Zc, _ = ch.despeckle(np.where(sheet, ch.grid(*(lambda xy, h, mm: (xy[mm], h[mm]))(*ch.field_points(*ch.load(RUN, k), Fr, corr))), np.nan))
        Zm, _ = m.masked(m.raw_grid(ME.pass_dir(RUN, k))[0])
        sc, sm = np.ptp(np.nanpercentile(Zc, [0.5, 99.5])), np.ptp(np.nanpercentile(Zm, [0.5, 99.5]))
        mm = np.isfinite(Zc) & np.isfinite(Zm); r = np.corrcoef(Zc[mm], Zm[mm])[0, 1]
        print(f"[ok] 10-21 pass {k} on the 231 x 124 sheet: span {sm:.2f} vs {sc:.2f} mm (recalibrated chain), corr {r:.4f}, "
              f"coverage {np.isfinite(Zm).mean():.3f} vs {np.isfinite(Zc).mean():.3f}")
        assert abs(sc - sm) < 1.0 and r > 0.99, (sc, sm, r)          # same tolerance as the 174 port of 10-28
    with tempfile.TemporaryDirectory() as d:
        m.save(d); m2 = ME.RigMeasurer(sheet=s124).load(d)
        assert np.array_equal(m2.sheet_mask, m.sheet_mask) and np.allclose(m2.frame["c"], m.frame["c"])
        print("[ok] reference save/load")
    # 2. the real sheets from 2026-10-29: 231 x 174, same grid as the targets
    m174 = ME.RigMeasurer()
    tgt = F.HeightField.load(os.path.join(RT, "figures", "rig", "targets", "S1_t042_c4.npz"))
    assert tgt.z_mm.shape == (len(m174.yg), len(m174.xg)) and np.allclose(tgt.x_mm, m174.xg) and np.allclose(tgt.y_mm, m174.yg), "target and measurement grids differ"
    print(f"[ok] 231 x 174 measurer grid == target grid {tgt.z_mm.shape}, hardware mask {m174.hw_mm} mm above the gripped edge")
    flat = tgt

    prof = MN.NewsheetModel()
    hist = [dict(cx=3.0, cy=15.1, wx=99.5, wy=105.6, rot=0, N=15, n=2), dict(cx=3.0, cy=15.1, wx=99.5, wy=105.6, rot=0, N=15, n=1)]
    h = prof.history(hist)
    assert h == [{"wa": {"center_xy": [3.0, 15.1], "size_wh": [99.5, 105.6]}, "zigzag": {"rot": 0, "N": 15}, "hits_done": 3}], h
    assert prof.build_prompt(flat.zrange(), tgt.zrange(), h) == PL.build_prompt(flat.zrange(), tgt.zrange(), h, schema="v5facts")
    assert "PROCESS FACTS" in prof.build_prompt((0, 0), (0, 1), [])
    with tempfile.TemporaryDirectory() as d:
        prof.render(tgt, os.path.join(d, "a.png")); R.field_png(tgt, os.path.join(d, "b.png"))
        assert open(os.path.join(d, "a.png"), "rb").read() == open(os.path.join(d, "b.png"), "rb").read()
    print("[ok] profile: history schema, prompt == planners.build_prompt (v5facts), render == render.field_png")

    sheet = F.Sheet(width_mm=231.0, length_mm=174.0, thickness_mm=0.6, clamp_mm=12.0, margin_side_mm=12.0, margin_bottom_mm=0.0,
                    pad_offsets_from_left_mm=(6.5, 224.5), pad_diameter_mm=13.0)
    tps, problems = spec.plan_to_toolpaths([dict(cx=3.0, cy=15.1, wx=99.5, wy=105.6, rot=0, N=15, n=1)], sheet)
    assert tps and not problems, problems
    print(f"[ok] S1_t042 recipe -> toolpath: {tps[0]['robot']}")
    print("ALL RIG PORT TESTS PASSED")


if __name__ == "__main__":
    main()
