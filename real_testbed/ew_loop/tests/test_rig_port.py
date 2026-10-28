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
    m = ME.RigMeasurer()
    flat = m.reference(RUN)
    assert m.info["inside_fraction"] > 0.7 and m.info["beyond_edge_points"]["y+"] > m.info["beyond_edge_points"]["y-"], m.info
    assert flat.z_mm.shape == (88, 117) and abs(flat.x_mm[-1] - 115.5) < 1e-6
    tgt = F.HeightField.load(os.path.join(RT, "figures", "rig", "targets", "S1_t042_c4.npz"))
    assert tgt.z_mm.shape == flat.z_mm.shape and np.allclose(tgt.x_mm, flat.x_mm) and np.allclose(tgt.y_mm, flat.y_mm), "target and measurement grids differ"
    print(f"[ok] reference on the 10-21 flat scan: {m.info['inside_fraction']} inside, camera offsets {m.info['camera_flat_offset_mm']}, "
          f"flat z {flat.zrange()[0]:.2f}..{flat.zrange()[1]:.2f} mm, coverage {flat.meta['coverage_observed']}; grid == target grid")
    for k in (3, 6):
        a = m.field(ME.pass_dir(RUN, k)); b = reference_chain(RUN, k)
        # same chain, nodes 0-1 mm apart (arange vs linspace): compare robust spans and the correlation on common cells
        sa, sb = np.nanpercentile(a.z_mm, [0.5, 99.5]), np.nanpercentile(b.z_mm, [0.5, 99.5])
        mm = np.isfinite(a.z_mm) & np.isfinite(b.z_mm)
        r = np.corrcoef(a.z_mm[mm], b.z_mm[mm])[0, 1]
        print(f"[ok] pass {k}: span {np.ptp(sa):.2f} vs {np.ptp(sb):.2f} mm (danish_run2_measure), corr {r:.4f}, coverage {a.meta['coverage_observed']}")
        assert abs(np.ptp(sa) - np.ptp(sb)) < 1.0 and r > 0.97, (sa, sb, r)
    with tempfile.TemporaryDirectory() as d:
        m.save(d); m2 = ME.RigMeasurer().load(d)
        c, c0 = m2.field(ME.pass_dir(RUN, 3)), m.field(ME.pass_dir(RUN, 3))
        assert np.allclose(np.nan_to_num(c.z_mm), np.nan_to_num(c0.z_mm))
        print("[ok] reference save/load")

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
