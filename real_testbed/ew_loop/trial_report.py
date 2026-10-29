"""TRIAL.md for one loop folder: per round what was measured, what the model planned, what the controller decided
and why, what the robot was given (toolpath, his run folder) and what came back, with the images.

    python -m ew_loop.trial_report <loop folder>          # writes <loop folder>/TRIAL.md
Reads rounds.jsonl (written by run_loop) and falls back to round_NN/{response,toolpath,robot/robot_run}.json + state.json.
"""
import glob
import json
import os
import sys


def _wa(e):
    return f"({e['cx']:.1f},{e['cy']:.1f}) {e['wx']:.0f}x{e['wy']:.0f} rot {e['rot']} N {e['N']} hits {e['n']}"


def _rel(root, p):
    try:
        return os.path.relpath(p, root).replace("\\", "/")
    except ValueError:
        return p


def load_rounds(root):
    p = os.path.join(root, "rounds.jsonl")
    if os.path.exists(p):
        return [json.loads(l) for l in open(p) if l.strip()]
    out = []                                                    # older folders: rebuild what can be rebuilt
    st = json.load(open(os.path.join(root, "state.json")))["state"] if os.path.exists(os.path.join(root, "state.json")) else {}
    for rd in sorted(glob.glob(os.path.join(root, "round_[0-9][0-9]"))):
        r = int(os.path.basename(rd)[-2:])
        if r == 0:
            continue
        rec = {"round": r, "before": next((s for s in reversed(st.get("scans", [])) if s["round"] < r), None)}
        if os.path.exists(os.path.join(rd, "response.json")):
            res = json.load(open(os.path.join(rd, "response.json"))); rec.update({"plan": res.get("was"), "model_secs": res.get("secs")})
            rec["chosen"] = res["was"][0] if res.get("was") else None
        if os.path.exists(os.path.join(rd, "toolpath.json")):
            tp = json.load(open(os.path.join(rd, "toolpath.json")))["toolpaths"][0]
            rec["toolpath"] = {"id": tp["id"], "passes": tp["passes"], "robot": tp["robot"], "ours": tp.get("ours"), "n_pivots": len(tp["pivots_xy_mm"])}
            rec["block_passes"] = tp["passes"]
        if os.path.exists(os.path.join(rd, "robot", "robot_run.json")):
            rec["robot"] = json.load(open(os.path.join(rd, "robot", "robot_run.json")))
        rec["measured"] = [s for s in st.get("scans", []) if s["round"] == r]
        out.append(rec)
    return out


def main(root):
    root = os.path.abspath(root)
    st = json.load(open(os.path.join(root, "state.json"))) if os.path.exists(os.path.join(root, "state.json")) else {}
    s = st.get("state", {})
    rounds = load_rounds(root)
    L = [f"# Trial `{os.path.basename(root)}`", ""]
    tgt = json.load(open(os.path.join(root, "target.npz").replace(".npz", ".json"))) if os.path.exists(os.path.join(root, "target.json")) else None
    L.append(f"Folder: `{root}`  ")
    if s:
        last = s["scans"][-1] if s.get("scans") else {}
        L.append(f"Stop: **{s.get('reason')}** after {s.get('exec_passes')} pass(es); last amp {last.get('amp_ratio')} corr {last.get('pattern_corr')} "
                 f"(span {last.get('span_cur')} / target {last.get('span_tgt')} mm)  ")
        if s.get("scans"):
            L.append(f"Flat scan: amp {s['scans'][0]['amp_ratio']} corr {s['scans'][0]['pattern_corr']}  ")
    for p in ("target.png", "round_00/field_reference.png", "round_00/reference/reference_preview.png", "summary.png"):
        if os.path.exists(os.path.join(root, p)):
            L.append(f"![{p}]({p})")
    L.append("")
    L.append("## Rounds")
    L.append("")
    L.append("| round | measured before (amp / corr) | model plan (first entry executed) | controller | toolpath given to the robot | robot run | measured after (capture: amp / corr / z) |")
    L.append("|---|---|---|---|---|---|---|")
    for rec in rounds:
        b = rec.get("before") or {}
        plan = rec.get("plan") or []
        plan_txt = "<br>".join(("**" if i == 0 else "") + _wa(e) + ("**" if i == 0 else "") for i, e in enumerate(plan)) or "(none)"
        ctrl = f"{rec.get('block_passes', '?')} pass(es)<br><small>{rec.get('block_reason', '')}</small>"
        tp = rec.get("toolpath") or {}
        rb = tp.get("robot") or {}
        tp_txt = (f"{tp.get('id')} x{tp.get('passes')}: N {rb.get('zigzag_N')}, lv {rb.get('zigzag_lv_mm')}, alpha {rb.get('zigzag_alpha_deg')}, "
                  f"cx {rb.get('center_x_mm')}, start {rb.get('start_mm')}, {tp.get('n_pivots')} pivots") if tp else "(none)"
        ro = rec.get("robot") or {}
        ro_txt = (f"`{os.path.basename(str(ro.get('output_dir') or '').rstrip(chr(92) + '/'))}` captures {sorted(ro.get('captures', {}))}"
                  + (f" {ro.get('secs')} s" if ro.get("secs") else "")) if ro else "(none)"
        meas = "<br>".join(f"{m.get('capture', m.get('pass'))}: {m.get('amp_ratio')} / {m.get('pattern_corr')} / {m.get('z_min', '')}..{m.get('z_max', '')}"
                           + (" **(block)**" if m.get("block_measurement") else "") for m in rec.get("measured") or []) or "(none)"
        stop = f"<br>stop: {rec['stop']}" if rec.get("stop") else ""
        L.append(f"| {rec['round']} | {b.get('amp_ratio')} / {b.get('pattern_corr')} | {plan_txt} | {ctrl}{stop} | {tp_txt} | {ro_txt} | {meas} |")
    L.append("")
    for rec in rounds:
        r = rec["round"]; rd = os.path.join(root, f"round_{r:02d}")
        L.append(f"## Round {r}")
        if rec.get("block_reason"):
            L.append(f"Controller: {rec['block_reason']}  ")
        if rec.get("history_given") is not None:
            L.append(f"History given to the model: `{json.dumps(rec['history_given'])}`  ")
        imgs = [os.path.join(rd, "preview.png")] + sorted(glob.glob(os.path.join(rd, "robot", "run_*", "zigzag_preview.png"))) \
             + [os.path.join(rd, "cur.png"), os.path.join(rd, "tgt.png")] + sorted(glob.glob(os.path.join(rd, "field_*.png")))
        L.append(" ".join(f"![{os.path.basename(p)}]({_rel(root, p)})" for p in imgs if os.path.exists(p)))
        ro = rec.get("robot") or {}
        ri = os.path.join(ro.get("local_copy") or "", "run_info.txt") if ro.get("local_copy") else None
        if ri and os.path.exists(ri):
            L.append(f"His run folder: `{_rel(root, ro['local_copy'])}` (run_info.txt, force CSVs, point clouds)  ")
        L.append("")
    open(os.path.join(root, "TRIAL.md"), "w").write("\n".join(L))
    print(os.path.join(root, "TRIAL.md"), f"({len(rounds)} rounds)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
