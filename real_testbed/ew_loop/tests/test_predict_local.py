"""GPU test (slow, ~1-2 min): ckpt-1984 answers for flat -> pilot target through the local
backend, and the HTTP server/client round trip returns the same greedy plan.
Run: CUDA_VISIBLE_DEVICES=7 conda run -n EW python -m ew_loop.tests.test_predict_local
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request

import numpy as np

from ew_loop import frames, predict as P, spec
from ew_loop.frames import Sheet

PROJECT = os.path.abspath(os.path.join(os.path.dirname(frames.__file__), "..", ".."))
CKPT = os.path.join(PROJECT, "experiments/mpc_end2end/runs/full_e8_lr2e-4_r16/checkpoint-1984")
PILOT = os.path.join(PROJECT, "experiments/mpc_end2end/dataset_pilot/C1/t018")


def main():
    sheet = Sheet.parse("300x300", clamp_mm=0.0, thickness_mm=0.5)   # the checkpoint's training panel
    xyz = np.load(os.path.join(PILOT, "cycles.npz"))["xyz"] * 1e3
    rec = json.load(open(os.path.join(PILOT, "record.json")))
    tgt = frames.field_from_points(xyz[-1][:, :2], xyz[-1][:, 2], sheet, 2.0)
    cur = frames.flat_field(sheet, 2.0)
    print("target z range", tgt.zrange(), "| ground-truth plan:", rec["was"])
    planner = P.LocalPlanner(CKPT)
    print(f"model loaded in {planner.load_secs:.0f}s on {planner.model.device}")
    out = tempfile.mkdtemp(prefix="ew_predict_")
    res = P.predict(cur, tgt, [], planner, os.path.join(out, "local"))
    print(f"local: {res['secs']}s -> {res['raw'][:200]}")
    assert res["was"], "no parseable plan from the local backend"
    # the plan must convert to robot toolpaths on the training sheet
    tps, probs = spec.plan_to_toolpaths(res["was"], sheet, allow_rot90=True)
    print(f"  -> {len(tps)} toolpath(s), problems {probs}")
    assert tps, probs
    # history round: pretend the first entry was executed, re-plan
    res2 = P.predict(cur, tgt, [res["was"][0]], planner, os.path.join(out, "local_hist"))
    assert res2["history"] and res2["history"][0]["hits_done"] == res["was"][0]["n"]
    print(f"with history: {res2['secs']}s -> {res2['raw'][:120]}")

    # HTTP round trip on a free port, same planner instance
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    th = threading.Thread(target=P.serve, kwargs=dict(ckpt=CKPT, host="127.0.0.1", port=port, planner=planner), daemon=True)
    th.start()
    http = P.HttpPlanner(f"http://127.0.0.1:{port}")
    for _ in range(50):
        try:
            print("health:", http.health()); break
        except Exception:
            time.sleep(0.2)
    res3 = P.predict(cur, tgt, [], http, os.path.join(out, "http"))
    print(f"http: {res3['secs']}s -> {res3['raw'][:120]}")
    assert res3["plan"] == res["plan"], "HTTP greedy plan differs from local greedy plan"
    print("PREDICT TESTS PASSED;", out)


if __name__ == "__main__":
    main()
