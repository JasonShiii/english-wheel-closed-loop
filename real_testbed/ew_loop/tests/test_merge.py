"""merge_capture: three cameras with known calibration merge into one consistent cloud."""
import json
import os
import tempfile

import numpy as np

from ew_loop import scan_to_field as S


def rot(axis, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])}[axis]


def write_ply(path, P):
    P = P.astype(np.float32)
    hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    open(path, "wb").write(hdr.encode() + P.tobytes())


def main():
    rng = np.random.default_rng(0)
    X = np.stack([rng.uniform(-100, 100, 4000), rng.uniform(-100, 100, 4000), np.full(4000, 450.0)], 1)   # a flat sheet in the device_1 frame
    T = {1: np.eye(4), 2: np.eye(4), 3: np.eye(4)}
    T[2][:3, :3] = rot("y", 25); T[2][:3, 3] = [300, 20, 40]
    T[3][:3, :3] = rot("x", -20); T[3][:3, 3] = [-250, 60, -30]
    with tempfile.TemporaryDirectory() as d:
        cap = os.path.join(d, "pointcloud_pass1"); os.makedirs(cap)
        for dev in (1, 2, 3):
            Ti = np.linalg.inv(T[dev])
            Xd = X @ Ti[:3, :3].T + Ti[:3, 3]                                 # what camera dev would record
            far = np.array([[0.0, 0.0, 2000.0]] * 50)                          # background beyond the depth window
            write_ply(os.path.join(cap, f"device_{dev}_pointcloud_001.ply"), np.concatenate([Xd, far]))
        cal = os.path.join(d, "calib"); os.makedirs(os.path.join(cal, "icp"))
        json.dump({"source_run": "run_A", "devices": {f"device_{k}": T[k].tolist() for k in T}}, open(os.path.join(cal, "extrinsics.json"), "w"))
        M = S.load_capture(cap, "merged", cal)
        assert len(M) == 3 * 4000, len(M)                                     # background clipped per camera
        for k in range(3):
            assert np.allclose(M[k * 4000:(k + 1) * 4000], X, atol=1e-3), f"camera {k+1} not mapped back"
        # an accepted ICP file for the same run replaces the checkerboard transforms
        T2 = {k: v.copy() for k, v in T.items()}; T2[2][:3, 3] += [1.0, 0.0, 0.0]
        json.dump({"status": "accepted", "calibration_run": "run_A", "devices": {f"device_{k}": T2[k].tolist() for k in T2}},
                  open(os.path.join(cal, "icp", "refined_transforms.json"), "w"))
        _, src = S.load_rig_calibration(cal); assert src.startswith("icp"), src
        M2 = S.load_capture(cap, "merged", cal)
        assert np.allclose(M2[4000:8000, 0] - X[:, 0], 1.0, atol=1e-3)
        # an ICP file from another calibration run is ignored
        json.dump({"status": "accepted", "calibration_run": "run_B", "devices": {f"device_{k}": T2[k].tolist() for k in T2}},
                  open(os.path.join(cal, "icp", "refined_transforms.json"), "w"))
        _, src = S.load_rig_calibration(cal); assert src == "extrinsics.json", src
    print("[ok] merge_capture: per-camera depth clip, calibration applied, ICP refinement used only for its own run")
    print("ALL MERGE TESTS PASSED")


if __name__ == "__main__":
    main()
