"""scan_to_field on the fake rig: reference fit, robot-motion pose, outline pose, pose check,
up-sign handling, reference save/load. Run: python -m ew_loop.tests.test_scan_to_field"""
import os
import tempfile

import numpy as np

from ew_loop import frames, scan_to_field as S
from ew_loop.frames import Sheet
from ew_loop.synthetic import FakeRig, bent_surface, flat_surface


def truth_field(sheet, surface, step=2.0):
    xg, yg = frames.grid_for(sheet, step)
    X, Y = np.meshgrid(xg, yg)
    return surface(X, Y)


def errors(fld, truth, sheet):
    m = np.isfinite(fld.z_mm) & np.isfinite(truth) & (fld.y_mm[:, None] >= sheet.wheel_line_y)
    d = fld.z_mm[m] - truth[m]
    return float(np.sqrt(np.mean(d ** 2))), float(np.percentile(np.abs(d), 99)), int(m.sum())


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def run(up_sign):
    # the old 300 mm sheets of the existing real data: wheel line at y = -50, free part 200 mm;
    # the upper wheel hides ~70 mm of it next to the wheel (as in the real captures)
    sheet = Sheet(width_mm=195.0, length_mm=300.0, clamp_mm=100.0)
    rig = FakeRig(sheet, up_sign=up_sign, seed=1)
    P0 = rig.capture(flat_surface, rig.contact_ref)
    ref = S.make_reference(P0, sheet, rig.feed_dir_cam(error_deg=20.0), up_toward_camera=(up_sign > 0))
    # compare in the camera frame (the two plane bases differ)
    axis3 = lambda o, e1, e2, R2, k: R2[0, k] * e1 + R2[1, k] * e2
    centre3 = lambda o, e1, e2, t2: o + t2[0] * e1 + t2[1] * e2
    a_err = angle_deg(axis3(ref.origin, ref.e1, ref.e2, ref.R2, 1), axis3(rig.origin, rig.e1, rig.e2, rig.R2, 1))
    x_err = angle_deg(axis3(ref.origin, ref.e1, ref.e2, ref.R2, 0), axis3(rig.origin, rig.e1, rig.e2, rig.R2, 0))
    t_err = float(np.linalg.norm(centre3(ref.origin, ref.e1, ref.e2, ref.t2) - centre3(rig.origin, rig.e1, rig.e2, rig.t2_ref)))
    n_err = angle_deg(ref.n, rig.n)
    print(f"  reference: feed-axis error {a_err:.2f} deg, x-axis error {x_err:.2f} deg, centre error {t_err:.2f} mm, "
          f"normal error {n_err:.2f} deg, outline {ref.outline}")
    assert a_err < 0.5 and x_err < 0.5 and t_err < 1.5 and n_err < 0.3, (a_err, x_err, t_err, n_err)

    surf = bent_surface(sheet, amp_mm=12.0)
    truth = truth_field(sheet, surf)
    contact = np.array([40.0, sheet.wheel_line_y])            # robot moved: another point under the wheel
    P1 = rig.capture(surf, contact)
    f1 = S.scan_to_field(P1, ref, contact_xy=contact)
    rms, p99, n = errors(f1, truth, sheet)
    print(f"  robot-motion pose: RMS {rms:.2f} mm, p99 {p99:.2f} mm over {n} cells, coverage {f1.meta['coverage_usable']}, "
          f"pose check {f1.meta['pose_check']}")
    assert rms < 0.5 and p99 < 1.5 and f1.meta["coverage_usable"] > 0.4, (rms, p99, f1.meta["coverage_usable"])
    assert f1.meta["pose_check"]["ok"] is True
    f2 = S.scan_to_field(P1, ref, contact_xy=None)
    rms2, p992, n2 = errors(f2, truth, sheet)
    print(f"  outline pose:      RMS {rms2:.2f} mm, p99 {p992:.2f} mm over {n2} cells")
    assert rms2 < 1.0, rms2
    f3 = S.scan_to_field(P1, ref, contact_xy=np.array([-40.0, sheet.wheel_line_y]))   # wrong x sign
    print(f"  wrong x sign:      pose check {f3.meta['pose_check']}")
    assert f3.meta["pose_check"]["ok"] is False
    return ref, f1


def wheel_at_edge():
    """reference taken with the wheel on the sheet's edge (no clamp band, as on the simulator's
    300 mm sheet); later the sheet moves so that the whole wheel stands over it."""
    sheet = Sheet(width_mm=300.0, length_mm=300.0, clamp_mm=0.0)
    rig = FakeRig(sheet, seed=3)
    ref = S.make_reference(rig.capture(flat_surface, rig.contact_ref), sheet, rig.feed_dir_cam(error_deg=20.0))
    surf = bent_surface(sheet, amp_mm=6.0)
    contact = np.array([60.0, -50.0])
    f = S.scan_to_field(rig.capture(surf, contact), ref, contact_xy=contact)
    rms, p99, n = errors(f, truth_field(sheet, surf), sheet)
    print(f"  wheel on the edge at the reference, sheet moved 100+ mm: RMS {rms:.2f} mm, p99 {p99:.2f} mm over {n} cells")
    assert rms < 0.5 and p99 < 1.5, (rms, p99)


def main():
    wheel_at_edge()
    print("up = toward the camera")
    ref, f1 = run(+1.0)
    print("up = away from the camera")
    run(-1.0)
    with tempfile.TemporaryDirectory() as d:
        ref.save(d)
        back = S.Reference.load(d)
        assert np.allclose(back.R2, ref.R2) and np.allclose(back.t2, ref.t2)
        assert np.array_equal(back.obstruction["mask"], ref.obstruction["mask"])
        assert np.allclose(np.nan_to_num(back.baseline.z_mm), np.nan_to_num(ref.baseline.z_mm))
        # PLY round trip in the rig's binary format
        P = np.random.default_rng(0).normal(size=(1000, 3)).astype(np.float32)
        hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        open(os.path.join(d, "device_3_pointcloud_001.ply"), "wb").write(hdr.encode() + P.tobytes())
        assert np.allclose(S.load_capture(d, 3), P)
    print("ALL SCAN_TO_FIELD TESTS PASSED")


if __name__ == "__main__":
    main()
