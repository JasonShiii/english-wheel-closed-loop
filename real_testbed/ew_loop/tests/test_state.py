"""state.py: metrics agree with the offline evaluation; stop rules; block size decisions."""
import numpy as np

from ew_loop import add_repo_paths, frames, state as ST
from ew_loop.frames import Sheet, HeightField
add_repo_paths()
from eval_execute import field_metrics


def fld(z):
    sheet = Sheet.parse("300x300", 0.0, 0.5)
    x, y = frames.grid_for(sheet, 10.0)
    return HeightField(x, y, z.astype(np.float32), {})


def main():
    rng = np.random.default_rng(0)
    zt = rng.normal(size=(30, 30)) * 5
    zc = 0.6 * zt + rng.normal(size=(30, 30))
    m = ST.compare(fld(zc), fld(zt), robust=False)
    ref = field_metrics(zc.ravel().astype(np.float32).astype(float), zt.ravel().astype(np.float32).astype(float))
    assert abs(m["pattern_corr"] - ref["pattern_corr"]) < 1e-3 and abs(m["amp_ratio"] - ref["amp_ratio"]) < 2e-3, (m, ref)
    print("[ok] compare() == eval_execute.field_metrics (non-robust)", m["pattern_corr"], m["amp_ratio"])

    wa = dict(cx=0.0, cy=40.0, wx=120.0, wy=70.0, rot=0, N=8, n=5)
    s = ST.LoopState(max_passes=20)
    s.record_initial({"pattern_corr": 0.0, "span_cur": 0.0, "span_tgt": 30.0, "amp_ratio": 0.0})
    assert s.block_size(wa, 3) == 1, "unmeasured area -> one probe pass"
    s.record_pass({"pattern_corr": 0.8, "span_cur": 10.0, "span_tgt": 30.0, "amp_ratio": 0.333}, wa, 1, "TP0", 1)
    k = s.block_size(wa, 3)
    assert k == 3, k                                   # 10 mm/pass * 0.8^j stays below the 20 mm room for 3 passes? 8+6.4+5.1=19.5
    s.record_pass({"pattern_corr": 0.9, "span_cur": 18.0, "span_tgt": 30.0, "amp_ratio": 0.6}, wa, 2, "TP0", 1)
    s.record_pass({"pattern_corr": 0.95, "span_cur": 24.0, "span_tgt": 30.0, "amp_ratio": 0.8}, wa, 2, "TP0", 2)
    k = s.block_size(wa, 3)                            # inc 6, r = 6/8 = 0.75 -> next 4.5 > room 6? 4.5 <= 6 take; then 3.4: 7.9 > 6, closer? 1.9 < 1.5? no -> 1
    assert k == 1, k
    s.record_pass({"pattern_corr": 0.99, "span_cur": 28.6, "span_tgt": 30.0, "amp_ratio": 0.953}, wa, 3, "TP0", 1)
    assert s.check_stop() and s.reason == "converged", s.reason
    s2 = ST.LoopState.from_dict(s.to_dict())
    assert s2.reason == "converged" and s2.exec_passes == 4
    s3 = ST.LoopState(max_passes=50)
    for i in range(6):
        s3.scans.append({"pattern_corr": 0.5, "span_cur": 5.0, "span_tgt": 30.0, "amp_ratio": 0.17})
        stopped = s3.check_stop()
    assert stopped and s3.reason == "no_progress"
    s.record_executed(wa, 2); s.record_executed({**wa, "n": 1}, 1); s.record_executed({**wa, "N": 6}, 1)
    assert [h["n"] for h in s.history] == [3, 1]
    print("[ok] probe pass, decayed block size, converged / no-progress stops, history merging, save/load")
    print("ALL STATE TESTS PASSED")


if __name__ == "__main__":
    main()
