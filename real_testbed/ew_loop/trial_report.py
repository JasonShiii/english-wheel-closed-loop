"""TRIAL.md for one loop folder: per round what was measured, what the model planned, what the controller decided
and why, what the robot was given (toolpath, his run folder) and what came back, with the images.

    python -m ew_loop.trial_report <loop folder>          # writes <loop folder>/TRIAL.md
Reads rounds.jsonl (written by run_loop) and falls back to round_NN/{response,toolpath,robot/robot_run}.json + state.json.
"""
import glob
import json
import ntpath
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
    live = sorted(glob.glob(os.path.join(root, "round_[0-9][0-9]", "round.json")))
    if live:                                                    # written before execution and updated after (2026-10-29)
        return [json.load(open(p)) for p in live]
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


def scan_table(root, rounds):
    """the per-scan report (2026-10-30): after every scan, corr and amp ratio of the CLEANED field against the target
    (what the controller used), the cleaned z range (what the planner was told), and on the RAW field the sagitta across
    x over a 200 mm chord, the zone curvature and radius, the principal sagitta and the ridge axis; plus the wheel load"""
    rows, done = [], 0
    fm = os.path.join(root, "round_00", "flat_metrics.json")
    if os.path.exists(fm):
        f = json.load(open(fm)); c, k = f.get("compare_cleaned") or {}, f.get("calib_raw") or {}
        rows.append((0, "flat scan", c, k, None, None))
    for rec in rounds:
        for m in rec.get("measured") or []:
            if not m.get("calib"):
                continue
            done = (rec.get("executed") or rec.get("block_passes") or 0) + done if m.get("block_measurement") else done
            rows.append((done, f"round {rec['round']} capture {m.get('capture')}", m, m.get("calib") or {}, (rec.get("robot") or {}).get("wheel_load"),
                         f"{m.get('z_min')}..{m.get('z_max')}"))
    if len(rows) <= 1:
        return []
    L = ["## Per scan", "",
         "corr / amp: the cleaned field against the target (the controller's numbers). z: the cleaned field's range (the planner's). "
         "Sagitta, curvature: fits on the raw field (the noise averages out). Load: set threshold and the load-cell force while rolling.", "",
         "| passes | scan | corr | amp ratio | span (mm) | z planner (mm) | sagitta x 200 (mm) | kxx (1/m) | radius x (mm) | principal sagitta 200 (mm) | ridge axis (deg) | load set / rolling median (N) | rolling p5-p95 / max (N) |",
         "|" + "---|" * 13]
    for done, lab, c, k, wl, zr in rows:
        roll = (wl or {}).get("rolling") or {}
        L.append(f"| {done} | {lab} | {c.get('pattern_corr')} | {c.get('amp_ratio')} | {c.get('span_cur')} | {zr or ''} | {k.get('sag_x_200_mm')} | {k.get('kxx_per_m')} | "
                 f"{k.get('radius_x_mm')} | {k.get('sag_pmax_200_mm')} | {k.get('ridge_axis_deg')} | "
                 + (f"{(wl or {}).get('threshold_N')} / {roll.get('median_N')} | {roll.get('p5_N')}-{roll.get('p95_N')} / {roll.get('max_N')} |" if roll else " | |"))
    return L + [""]


def main(root, quiet=False):
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
    for p in ("target.png", "round_00/scan_flat.png", "round_00/field_reference.png", "round_00/reference/reference_preview.png", "summary.png"):
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
        dirs = " ".join("L>R" if d == "l2r" else "R>L" for d in (tp.get("pass_directions") or []))
        tp_txt = (f"{tp.get('id')} x{tp.get('passes')} [{dirs}]: N {rb.get('zigzag_N')}, lv {rb.get('zigzag_lv_mm')}, alpha {rb.get('zigzag_alpha_deg')}, "
                  f"cx {rb.get('center_x_mm')}, start {rb.get('start_mm')}, {tp.get('n_pivots')} pivots") if tp else "(none)"
        ro = rec.get("robot") or {}
        ro_txt = (f"`{ntpath.basename(str(ro.get('output_dir') or '').rstrip(chr(92) + '/'))}` captures {sorted(ro.get('captures', {}))}"
                  + (f" {ro.get('secs')} s" if ro.get("secs") else "")) if ro else "(none)"
        wl = (ro.get("wheel_load") or {}) if ro else {}
        if wl.get("rolling"):
            ro_txt += (f"<br>load {wl.get('threshold_N')} N set; rolling median {wl['rolling']['median_N']} N "
                       f"(p5-p95 {wl['rolling']['p5_N']}-{wl['rolling']['p95_N']}, max {wl['rolling']['max_N']})")
        meas = "<br>".join(f"{m.get('capture', m.get('pass'))}: {m.get('amp_ratio')} / {m.get('pattern_corr')} / {m.get('z_min', '')}..{m.get('z_max', '')}"
                           + (" **(block)**" if m.get("block_measurement") else "") for m in rec.get("measured") or []) or "(none)"
        stop = (f"<br>stop: {rec['stop']}" if rec.get("stop") else "") + (f"<br><i>{rec['status']}</i>" if rec.get("status") else "")
        L.append(f"| {rec['round']} | {b.get('amp_ratio')} / {b.get('pattern_corr')} | {plan_txt} | {ctrl}{stop} | {tp_txt} | {ro_txt} | {meas} |")
    L.append("")
    L += scan_table(root, rounds)
    for rec in rounds:
        r = rec["round"]; rd = os.path.join(root, f"round_{r:02d}")
        L.append(f"## Round {r}")
        if rec.get("block_reason"):
            L.append(f"Controller: {rec['block_reason']}  ")
        if rec.get("history_given") is not None:
            L.append(f"History given to the model: `{json.dumps(rec['history_given'])}`  ")
        imgs = [os.path.join(rd, "replan.png"), os.path.join(rd, "preview.png")] + sorted(glob.glob(os.path.join(rd, "robot", "run_*", "zigzag_preview.png"))) \
             + sorted(glob.glob(os.path.join(rd, "scan_*.png")))
        L.append(" ".join(f"![{os.path.basename(p)}]({_rel(root, p)})" for p in imgs if os.path.exists(p)))
        ro = rec.get("robot") or {}
        ri = os.path.join(ro.get("local_copy") or "", "run_info.txt") if ro.get("local_copy") else None
        if ri and os.path.exists(ri):
            L.append(f"His run folder: `{_rel(root, ro['local_copy'])}` (run_info.txt, force CSVs, point clouds)  ")
        L.append("")
    open(os.path.join(root, "TRIAL.md"), "w").write("\n".join(L))
    if not quiet:
        print(os.path.join(root, "TRIAL.md"), f"({len(rounds)} rounds)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
