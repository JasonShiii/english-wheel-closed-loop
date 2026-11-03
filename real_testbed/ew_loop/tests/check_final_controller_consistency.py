"""Consistency check of the rig service (ew_loop/final_controller.py) against the paper's closed loop (closedloop_v7.py,
configuration _cfg3_ruleteacher_sharedstop, TEST): every block of the given closed-loop runs is replayed through
FinalController.step with the run's own measured fields (field_bNN.npz), target (target.npz) and executed history; the
service's plan pick, block (entries: N, passes), rule action, g_hat and stop must equal the trace's. Runs on ideas2 (planner
answers and simulations come from the paper's caches: the test root's simchain is the paper root's).

  python -m ew_loop.tests.check_final_controller_consistency --planner-url http://127.0.0.1:18631 --out DIR [--runs g1.0/D1-t065_c0-5,...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from ew_loop import final_controller as FC
from ew_loop.frames import HeightField

PAPER_ROOT = FC.GA / "results_231x174_t1.0_D1-5_final75_closedloop_finalsys_riglike_proj"
CFG = "ours-D1-5-augreal-visfix-e3-ck1170_rulesched_cfg3_ruleteacher_sharedstop"


def merged(ops):
    from ew_loop import state as ST
    st = ST.LoopState(99)
    for o in ops:
        st.record_executed({k: v for k, v in o.items() if k != "n"}, int(o["n"]))
    return st.history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-url", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="g1.0/D1-t065_c0-5,g1.0/D5-t019_c0-10,g1.4/D3-t022_c0-4,g0.7/D2-t031_c0-6")
    ap.add_argument("--core0", type=int, default=120); ap.add_argument("--cores", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not (out / "simchain").exists():
        os.symlink(PAPER_ROOT / "simchain", out / "simchain")     # the paper's warm-start cache (read, and extended)
    ctl = FC.FinalController(a.planner_url, out, core0=a.core0, cores=a.cores)
    rows, ok_all = [], True
    for spec in a.runs.split(","):
        g, tag = spec.split("/")
        d = PAPER_ROOT / f"{CFG}_{g}" / tag
        t = json.load(open(d / "trace.json"))
        tgt = HeightField.load(str(d / "target.npz"))
        tgt.meta = {**(tgt.meta or {}), "source": f"{t['meta']['source']} cycle {t['meta']['cut'][1]} of {t['meta']['cut'][1]}"}
        run_id = f"consistency_{g}_{tag}"
        ops, prev_exec = [], None
        for i, b in enumerate(t["blocks"] + [None]):
            rnd = i + 1
            cur = HeightField.load(str(d / f"field_b{i:02d}.npz"))
            res = ctl.step(run_id, rnd, cur, tgt, merged(ops), executed_block=prev_exec, max_passes=12)
            if b is None:                                           # after the last block: the run's stop
                same = res["stop"] is not None and res["stop"]["reason"] == t["reason"]
                rows.append({"run": spec, "round": rnd, "check": "stop", "service": res["stop"], "paper": t["reason"], "same": same,
                             "timing": res["timing"]})
                ok_all &= same
                break
            if "executed" not in b:                                  # the paper stopped in this block (shared block_size 0)
                same = res["stop"] is not None and res["stop"]["reason"] == t["reason"]
                rows.append({"run": spec, "round": rnd, "check": "stop in block", "service": res["stop"], "paper": t["reason"], "same": same,
                             "pick_paper": (b.get("verifier") or {}).get("picked"), "pick_service": (res.get("samples") or {}).get("picked"),
                             "timing": res["timing"]})
                ok_all &= same and rows[-1]["pick_paper"] == rows[-1]["pick_service"]
                break
            ex_paper = [(int(e["N"]), int(e["n"])) for e in b.get("executed", [])]
            ex_srv = [(int(e["N"]), int(e["n"])) for e in (res["block"] or {}).get("entries", [])]
            pick_p = (b.get("verifier") or {}).get("picked")
            pick_s = (res.get("samples") or {}).get("picked")
            g_p = (t["blocks"][i - 1].get("gain_id") or {}).get("g_hat") if i else 1.0
            g_s = res["g_hat"]["g_hat"]
            act_p, act_s = (b.get("rule") or {}).get("action"), ((res["block"] or {}).get("rule") or {}).get("action")
            same = ex_paper == ex_srv and pick_p == pick_s and act_p == act_s and abs(float(g_p) - float(g_s)) < 1e-3
            rows.append({"run": spec, "round": rnd, "check": "block", "executed_paper": ex_paper, "executed_service": ex_srv,
                         "pick_paper": pick_p, "pick_service": pick_s, "action_paper": act_p, "action_service": act_s,
                         "g_hat_paper": g_p, "g_hat_service": g_s, "amp_service": res["measured"]["amp_ratio"],
                         "amp_paper": (b.get("at") or {}).get("amp_ratio"), "stop_service": res["stop"], "same": same, "timing": res["timing"]})
            print(json.dumps(rows[-1]), flush=True)
            ok_all &= same
            ops = ops + [dict(e) for e in b["executed"]]
            prev_exec = [dict(e) for e in b["executed"]]
        print(json.dumps(rows[-1]), flush=True)
    json.dump({"ok": bool(ok_all), "rows": rows}, open(out / "consistency.json", "w"), indent=1, default=float)
    print("ALL SAME" if ok_all else "DIFFERENCES (see consistency.json)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
