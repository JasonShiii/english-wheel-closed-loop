"""Consistency check of the rig service (ew_loop/dual_controller.py) against the frozen simulator loop (closedloop_dual.py, the N2
main line on the TEST split, 2026-11-13): every round of the given runs is replayed through DualController.step with the run's own
measured (denoised) fields (field_bNN.npz), target (target.npz) and executed history; the service's verifier pick, block (N, passes),
belief (g_hat, a_mu), smallest-step risk and stop reason must equal the trace's. The planner answers come from the paper's answer
cache (same prompt and images), the nominal chains from the frozen runs' warm-start store (symlinked into the test root).

  /home/xiangyu/miniconda3/envs/EW/bin/python -m ew_loop.tests.check_dual_controller_consistency --planner-url http://127.0.0.1:18631 --out DIR
      [--runs g1.0/D1-t022_c0-3,...] [--core0 120 --cores 8]
"""
import argparse
import json
import os
import sys
from pathlib import Path

from ew_loop import dual_controller as DCS
from ew_loop.frames import HeightField

SD = DCS.SD
PAPER_ROOT = SD / "sim_runs" / "test_designc2_spanden"
CFG = "ours-D1-5-augreal-visfix-e3-ck1170_dual_rl_d0.1_designc2_spanden_c2shield2_designc2_spanden_c2shield2"
CHAIN = Path("/home/xiangyu/English_wheel/experiments/GPT-astra_comparison/results_231x174_t1.0_D1-5_fresh_cl30_closedloop_riglike_proj/simchain")


def merged(ops):
    from ew_loop import state as ST
    st = ST.LoopState(99)
    for o in ops:
        st.record_executed({k: v for k, v in o.items() if k != "n"}, int(o["n"]))
    return st.history


def close(x, y, tol=2e-3):
    if x is None or y is None:
        return x is None and y is None
    return abs(float(x) - float(y)) <= tol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-url", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default="g1.0/D1-t022_c0-3,g1.0/D1-t025_c0-5,g1.4/D3-t022_c0-4,g1.2/D2-t031_c0-6")
    ap.add_argument("--core0", type=int, default=120); ap.add_argument("--cores", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not (out / "simchain").exists():
        os.symlink(CHAIN, out / "simchain")                    # the frozen runs' warm-start store (read, and extended)
    ctl = DCS.DualController(a.planner_url, out, core0=a.core0, cores=a.cores)
    rows, ok_all = [], True
    for spec in a.runs.split(","):
        g, tag = spec.split("/")
        d = PAPER_ROOT / f"{CFG}_{g}" / tag
        t = json.load(open(d / "trace.json"))
        tgt = HeightField.load(str(d / "target.npz"))
        tgt.meta = {**(tgt.meta or {}), "source": f"{t['meta']['source']} cycle {t['meta']['cut'][1]} of {t['meta']['cut'][1]}"}
        run_id = f"consistency_{g}_{tag}"
        ops, prev_exec = [], None
        blocks = t["blocks"]
        for i, b in enumerate(blocks + [None]):
            rnd = i + 1
            cur = HeightField.load(str(d / f"field_b{i:02d}.npz"))
            res = ctl.step(run_id, rnd, cur, tgt, merged(ops), executed_block=prev_exec, max_passes=16, denoise=False)
            row = {"run": spec, "round": rnd, "timing": res["timing"], "amp_service": res["measured"]["amp_ratio"], "belief": res.get("belief")}
            if b is None or "forecast" not in b:                 # after the last block (or a plan failure): the run's stop
                same = res["stop"] is not None and res["stop"]["reason"] == t["reason"]
                if t["reason"] == "next_step_risk" and same:
                    same &= close(res["dual"]["p_over_min"], t["stop_decision"]["dual"]["p_over_min"])
                    row.update(p_over_min_paper=t["stop_decision"]["dual"]["p_over_min"], p_over_min_service=res["dual"]["p_over_min"])
                row.update(check="stop", service=res["stop"], paper=t["reason"], same=same)
                if i:                                             # the belief after the last block
                    gp = blocks[i - 1].get("gain_id", {}); ab = blocks[i - 1].get("amp_belief") or {}
                    row.update(g_hat_paper=gp.get("g_hat"), g_hat_service=res["g_hat"]["g_hat"], a_mu_paper=ab.get("a_mu"), a_mu_service=res["belief"]["a_mu"])
                    same &= close(gp.get("g_hat"), res["g_hat"]["g_hat"]) and (not ab or close(ab.get("a_mu"), res["belief"]["a_mu"]))
                    row["same"] = same
                rows.append(row); ok_all &= same
                print(json.dumps(row, default=float), flush=True)
                break
            ex_paper = [(int(e["N"]), int(e["n"])) for e in b.get("executed", [])]
            ex_srv = [(int(e["N"]), int(e["n"])) for e in (res["block"] or {}).get("entries", [])]
            pick_p = (b.get("verifier") or {}).get("picked")
            pick_s = (res.get("samples") or {}).get("picked")
            dp, ds = b.get("dual", {}), res.get("dual", {})
            gp = blocks[i - 1].get("gain_id", {}) if i else {}
            ab = blocks[i - 1].get("amp_belief") if i else None
            same = (ex_paper == ex_srv and pick_p == pick_s and res["stop"] is None and close(dp.get("p_over_min"), ds.get("p_over_min"))
                    and dp.get("action") == ds.get("action") and dp.get("hits_g") == ds.get("hits_g") and dp.get("shield_admissible") == ds.get("shield_admissible")
                    and close(dp.get("a_mu"), ds.get("a_mu")) and close(dp.get("belief_mu"), ds.get("belief_mu"))
                    and (not i or close(gp.get("g_hat"), res["g_hat"]["g_hat"])) and (not ab or close(ab.get("a_mu"), res["belief"]["a_mu"])))
            row.update(check="block", executed_paper=ex_paper, executed_service=ex_srv, pick_paper=pick_p, pick_service=pick_s,
                       action_paper=dp.get("action"), action_service=ds.get("action"), p_over_min_paper=dp.get("p_over_min"), p_over_min_service=ds.get("p_over_min"),
                       hits_g_paper=dp.get("hits_g"), hits_g_service=ds.get("hits_g"), adm_paper=dp.get("shield_admissible"), adm_service=ds.get("shield_admissible"),
                       a_mu_paper=dp.get("a_mu"), a_mu_service=ds.get("a_mu"), g_hat_paper=(gp.get("g_hat") if i else None), g_hat_service=res["g_hat"]["g_hat"],
                       amp_paper=(b.get("at") or {}).get("amp_ratio"), stop_service=res["stop"], same=same)
            rows.append(row); ok_all &= same
            print(json.dumps(row, default=float), flush=True)
            if not same and res["stop"] is not None:
                break
            ops = ops + [dict(e) for e in b["executed"]]
            prev_exec = [dict(e) for e in b["executed"]]
    json.dump({"ok": bool(ok_all), "rows": rows, "service": ctl.info}, open(out / "consistency.json", "w"), indent=1, default=float)
    print("ALL SAME" if ok_all else "DIFFERENCES (see consistency.json)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
