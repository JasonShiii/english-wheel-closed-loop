"""model_io.py must stay identical in behaviour to the research-repository originals."""
import itertools

import numpy as np

from ew_loop import add_repo_paths, model_io as M
add_repo_paths()
import common                                         # experiments/mpc_end2end
import category_tests as ct                           # experiments/category_tests
from eval_execute import field_metrics                # experiments/mpc_end2end
from utils.process_rotation_data import zigzag_vertices


def main():
    wa = dict(cx=-12.5, cy=40.0, wx=120.0, wy=70.0, rot=0, N=8, n=3)
    hist = [M.entry_desc(wa, 2, "hits_done", "v1")]
    for schema in ("v1", "v2"):
        assert M.build_prompt((0.0, 3.21), (-1.5, 9.9), hist, schema) == common.build_prompt((0.0, 3.21), (-1.5, 9.9), hist, schema)
        assert M.build_prompt((0, 0), (0, 1), [], schema) == common.build_prompt((0, 0), (0, 1), [], schema)
        assert M.entry_desc(wa, 3, "hits", schema) == common.entry_desc(wa, 3, "hits", schema)
        e = common.entry_desc(wa, 3, "hits", schema)
        assert M.read_entry(e) == common.read_entry(e)
    assert M.W == ct.W
    for wx, wy, rot, N in itertools.product([60.0, 140.0], [50.0, 120.0], [0, 90], [4, 7, 12]):
        assert M.strip_stack_dims(wx, wy, rot) == ct.strip_stack_dims(wx, wy, rot)
        assert M.inscribe(wx, wy, rot, N) == ct.inscribe(wx, wy, rot, N)
    for lv, a, N, r in itertools.product([60.0, 150.0], [2.0, 9.0], [4, 9, 14], [0.0, 90.0, 30.0]):
        assert np.array_equal(M.zigzag_vertices(lv, a, N, r), zigzag_vertices(lv, a, N, r))
    rng = np.random.default_rng(0)
    zp, zg = rng.normal(size=500), rng.normal(size=500)
    assert M.field_metrics(zp, zg) == field_metrics(zp, zg)
    import os, tempfile
    xyz = rng.normal(size=(2601, 3)) * [0.08, 0.08, 0.01]
    with tempfile.TemporaryDirectory() as d:
        M.heatmap_png(xyz, xyz[:, 2] * 1e3, os.path.join(d, "a.png"))
        common.heatmap_png(xyz, xyz[:, 2] * 1e3, os.path.join(d, "b.png"))
        assert open(os.path.join(d, "a.png"), "rb").read() == open(os.path.join(d, "b.png"), "rb").read()
    print("[ok] model_io matches common.py (prompt, schema, renderer), category_tests.py, process_rotation_data.py, eval_execute.py")


if __name__ == "__main__":
    main()
