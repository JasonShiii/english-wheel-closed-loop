"""Replay of the ARCHIVED REAL rig runs (11-04 Demo A steel D1/t081 and aluminium D1/t065; 11-02 Demo B) through the RSS main-line
service (ew_loop/dual_controller.py): every round's cleaned real scan, the target and the blocks the operator actually executed are
fed in order; the service's readings (denoised span), beliefs (g_hat, a_mu), stop recommendation and recommended block are
recorded next to what the ICCPS advisory controller said at the time. A sanity check of the frozen controller on real data (what
it WOULD have recommended), not an identity test: nothing was executed from its advice. Needs a planner server (new prompts).

  /home/xiangyu/miniconda3/envs/EW/bin/python -m ew_loop.tests.replay_real_runs_dual --planner-url http://127.0.0.1:18631 --out DIR [--core0 120 --cores 8]
"""
import argparse
import json
import sys
from pathlib import Path

from ew_loop import dual_controller as DCS
from ew_loop.frames import HeightField

RT = Path(__file__).resolve().parents[2]
RUNS = {
    "steel_D1_t081_2026-11-04": RT / "robot_pc_archive_2026-11-04" / "loops" / "demoA_D1_t081_supervised_closedloop",
    "aluminium_D1_t065_2026-11-04": RT / "robot_pc_archive_2026-11-04_aluminium" / "loops" / "demoA_D1_t065_newmaterial_supervised_closedloop",
    "demoB_D5_t019_2026-11-02": RT / "robot_pc_archive_2026-11-02" / "loops" / "demoB_D5_t019_supervised_closedloop",
}


def rounds_of(d):
    rows = [json.loads(l) for l in open(d / "rounds.jsonl")]
    out = []
    for r in rows:
        ex = r.get("executed") or 0
        fld = None
        for m in r.get("measured") or []:
            if m.get("field"):
                fld = d / f"round_{r['round']:02d}" / Path(m["field"].replace("\\", "/")).name
        blk = r.get("executed_block")
        if not blk and ex and r.get("chosen"):
            blk = [{**{k: v for k, v in r["chosen"].items() if k != "n"}, "n": int(ex)}]
        out.append({"round": r["round"], "executed": ex, "block": blk, "field_after": fld, "before": r.get("before"),
                    "advisory": (r.get("final_controller") or {}).get("recommendation") or r.get("controller_stop")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-url", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--core0", type=int, default=120); ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--runs", default=",".join(RUNS))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ctl = DCS.DualController(a.planner_url, out, core0=a.core0, cores=a.cores)
    report = {}
    for name in a.runs.split(","):
        d = RUNS[name]
        tgt = HeightField.load(str(d / "target.npz"))
        rounds = rounds_of(d)
        cur = HeightField.load(str(d / "round_00" / "field_reference.npz"))
        ops, prev_exec, rows = [], None, []
        for i, r in enumerate(rounds + [None]):
            rnd = i + 1
            res = ctl.step(f"real_{name}", rnd, cur, tgt, ops, executed_block=prev_exec, max_passes=16)
            adv = None
            af = d / f"round_{rnd:02d}" / "advisory_final_controller.json"
            if af.exists():
                A = json.load(open(af))
                adv = {"proj_amp": (A.get("measured") or {}).get("amp_ratio"), "g_hat": (A.get("g_hat") or {}).get("g_hat"), "stop": A.get("stop"),
                       "block": [(int(e["N"]), int(e["n"])) for e in ((A.get("block") or {}).get("entries") or [])]}
            row = {"round": rnd, "history_passes": sum(int(o["n"]) for o in ops),
                   "reading": {"span_denoised": res["measured"]["amp_ratio"], "corr_denoised": res["measured"]["pattern_corr"],
                               "span_raw": res["measured_raw"]["amp_ratio"], "corr_raw": res["measured_raw"]["pattern_corr"]},
                   "belief": res["belief"], "g_id": {k: res["g_hat"].get(k) for k in ("ratio", "inc_nominal", "inc_measured", "skipped", "g_hat")},
                   "amp_belief": res.get("amp_belief"), "stop": res["stop"], "dual": {k: (res.get("dual") or {}).get(k) for k in ("p_over_min", "hits_g", "shield_admissible", "rule", "action")},
                   "recommended": [(int(e["N"]), int(e["n"])) for e in ((res.get("block") or {}).get("entries") or [])],
                   "plan_first": ((res.get("plan") or {}).get("was") or [None])[0], "verifier_pick": (res.get("samples") or {}).get("picked"),
                   "iccps_advisory": adv, "timing": res["timing"]}
            if r is not None:
                row["operator_executed"] = [(int(e["N"]), int(e["n"])) for e in (r["block"] or [])]
            rows.append(row)
            print(f"{name} r{rnd}: reading span_den {row['reading']['span_denoised']:.3f} (raw {row['reading']['span_raw']:.3f}) corr {row['reading']['corr_denoised']:.3f} | "
                  f"g_hat {res['belief']['g_hat']:.3f} a_mu {res['belief']['a_mu']:.3f} +- {res['belief']['a_sd']:.3f} | stop {res['stop']} | rec {row['recommended']} "
                  f"(p_min {row['dual']['p_over_min']}) | ICCPS {adv} | operator {row.get('operator_executed')}", flush=True)
            if r is None or not r["executed"] or r["field_after"] is None or not r["field_after"].exists():
                break
            ops = ops + [dict(e) for e in r["block"]]
            prev_exec = [dict(e) for e in r["block"]]
            cur = HeightField.load(str(r["field_after"]))
        report[name] = rows
    json.dump({"service": ctl.info, "runs": report}, open(out / "replay_real_runs.json", "w"), indent=1, default=str)
    print("written", out / "replay_real_runs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
