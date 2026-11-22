"""rigloop --controller server-final (2026-11-02): the rig loop driven by the paper's FINAL controller, which runs as a service on
ideas2 (ew_loop/final_controller.py, `serve-fields --controller final`). The default rigloop (state.py sizing, loop.run_loop) is
untouched; this module only adds a second loop that reuses its measurement, cleaning, records and figures.

Per round:
  1. the cleaned field, the target, the executed history and the block executed since the last request go to the service, which
     returns the verifier-picked plan (8 samples with their predicted amplitudes), the rule's block for the plan's first entry
     (passes b, last-pass strip scale s and N', possibly as two parts: n_full passes at the plan's N, then 1 pass at N'), the shared
     stop check, the projection progress a, corr, g_hat (+ support), the forecast per pass, and the timing;
  2. HUMAN IN THE LOOP: the operator (a) accepts, (b) edits the block (passes, N, another sample or another entry of the plan),
     (c) skips the block and re-scans, or (d) stops; a free-text reason is asked every time. With --yes the recommendation is
     accepted (the service's stop ends the run);
  3. the EXECUTED block (not the recommended one) goes into the history and to the service at the next request; the operator's
     choice and reason are kept next to the recommendation in round_NN/round.json, rounds.jsonl and TRIAL.md.

Plants: the rig's (file / replay / dobot, as rigloop) and, for offline tests on ideas2, SimRigPlant (the paper's plant: the C++
simulator at a growth factor with the warm-start chain, every scan the rig-like rule of closedloop_v7.scan_riglike).
"""
import json
import os
import shutil
import time
from dataclasses import replace

import numpy as np

from . import loop as L, spec, state as ST
from .frames import HeightField


# ---------------------------------------------------------------------------------------------------------------------------
def _entry_txt(e):
    return f"({e['cx']:.1f},{e['cy']:.1f}) {e['wx']:.0f}x{e['wy']:.0f} {spec.orient_txt(e)} N {e['N']} x{e['n']}"


def show_recommendation(resp, log):
    m, g, st = resp["measured"], resp["g_hat"], resp.get("state", {})
    dual = "belief" in resp                                    # 2026-11-13: the RSS main-line service (dual_controller.py)
    if dual:
        b, r = resp["belief"], resp.get("measured_raw") or {}
        log(f"  [main line] reading (denoised span) {m['amp_ratio']:.3f} (raw {r.get('amp_ratio', float('nan')):.3f}), corr {m['pattern_corr']:.4f}; "
            f"amplitude belief {b['a_mu']:.3f} +- {b['a_sd']:.3f}; growth belief g {b['g_hat']:.3f} (log-sd {b['belief_sd']:.3f}, {b['n_ratio_obs']} ratio(s))")
    log(f"  [final controller] measured: progress a {m['amp_ratio']:.3f} ({'denoised span' if dual else 'projection'}), corr {m['pattern_corr']:.4f}; "
        f"g_hat {g['g_hat']:.3f} (support {g['support']}{', last ratio %.3f' % g['ratio'] if 'ratio' in g else ''}"
        f"{'; ' + g['skipped'] if g.get('skipped') else ''}); passes {st.get('exec_passes')}/{st.get('max_passes')}")
    plan = (resp.get("plan") or {}).get("was") or []
    for i, e in enumerate(plan):
        log(f"    plan {'>' if i == 0 else ' '} {i}: {_entry_txt(e)}")
    sm = resp.get("samples") or {}
    if sm.get("candidates"):
        log("    verifier samples (a_pred = predicted whole-sheet amplitude; pick rule |a-0.98| + 3 max(0, a-1.03)):")
        for c, w in zip(sm["candidates"], resp.get("samples_was") or [None] * len(sm["candidates"])):
            first = _entry_txt(w[0]) if w else (c.get("problem") or "unparsable")
            log(f"      {'*' if c['draw'] == sm.get('picked') else ' '} sample {c['draw']}: a_pred {c['a_pred']}  first entry {first}"
                f"{'  (+%d more entries)' % (len(w) - 1) if w and len(w) > 1 else ''}")
    if resp.get("stop"):
        log(f"  [final controller] RECOMMENDS STOP: {resp['stop']['reason']}  ({resp['stop']['rule']})")
    else:
        b = resp["block"]
        fc = b["forecast"]
        log(f"  [final controller] RECOMMENDED BLOCK: {' then '.join(_entry_txt(e) for e in b['entries'])}  "
            f"(b {b['b']}, s {b['s']}, N' {b['N']} of plan N {b['N_plan']}; rule: {b['rule']['rule']} plan {b['rule']['plan']}"
            f"{'; ' + '; '.join(b['rule']['conversions']) if b['rule'].get('conversions') else ''})")
        log(f"    forecast (nominal x {'E[g]' if dual else 'g_hat^kappa'} = {fc['g_used']}): predicted a after 0..{fc['k_fc']} full passes {fc['pred_amp_full_passes']}"
            + (f"; P(over) of the smallest step {b_['p_over_min']}, shield admissible {b_['shield_admissible']}" if dual and (b_ := b['rule']) else ""))
    log(f"    service timing: {resp.get('timing')}")


def _ask(prompt, default=""):
    try:
        s = input(prompt).strip()
    except EOFError:
        s = ""
    return s or default


def operator_decision(resp, sheet, max_angle, confirm, log):
    """-> {"choice": accept|edit|skip|stop, "reason": str, "block": [entries] or None, "edit": {...}}"""
    rec_block = None if resp.get("stop") else [dict(e) for e in resp["block"]["entries"]]
    if not confirm:
        return {"choice": "accept", "reason": "auto (--yes)", "block": rec_block}
    while True:
        if rec_block is None:
            ans = _ask("  controller recommends STOP: [x] stop (accept) / [e] run a block anyway / [s] skip + re-scan ? ", "x").lower()
            ans = {"a": "x", "accept": "x"}.get(ans, ans)
        else:
            ans = _ask("  [a] accept / [e] edit / [s] skip + re-scan / [x] stop ? ", "").lower()
        if ans not in ("a", "e", "s", "x"):
            print("  please type a, e, s or x"); continue
        reason = _ask("  reason (free text, Enter = none): ")
        if ans == "a":
            return {"choice": "accept", "reason": reason, "block": rec_block}
        if ans == "x":
            return {"choice": "stop", "reason": reason, "block": None}
        if ans == "s":
            return {"choice": "skip", "reason": reason, "block": None}
        # edit
        plans = resp.get("samples_was") or []
        picked = (resp.get("samples") or {}).get("picked", 0)
        si = _ask(f"    sample (0-{max(len(plans) - 1, 0)}, Enter = picked {picked}): ", str(picked))
        try:
            si = int(si); plan = plans[si] if plans else (resp.get("plan") or {}).get("was")
        except (ValueError, IndexError):
            print("    no such sample"); continue
        if not plan:
            print("    that sample has no executable plan"); continue
        ej = _ask(f"    entry of that plan (0-{len(plan) - 1}, Enter = 0): ", "0")
        try:
            e = dict(plan[int(ej)])
        except (ValueError, IndexError):
            print("    no such entry"); continue
        same = rec_block is not None and si == picked and int(ej) == 0
        dp = sum(x["n"] for x in rec_block) if same else 1
        dN = rec_block[-1]["N"] if same else e["N"]
        try:
            n = int(_ask(f"    passes (Enter = {dp}): ", str(dp)))
            N = int(_ask(f"    strip count N, all passes (Enter = {dN}{'; the recommendation thins only the last pass' if same and len(rec_block) > 1 else ''}): ", str(dN)))
        except ValueError:
            print("    passes and N must be integers"); continue
        if same and n == dp and N == dN and len(rec_block) > 1:
            block = rec_block                                    # unchanged: keep the two-part recommendation
        else:
            block = [{**e, "N": N, "n": n}]
        bad = None
        for x in block:
            _, probs = spec.plan_to_toolpaths([x], sheet, allow_rot90=False, max_angle=max_angle)
            bad = bad or (probs[0][1] if probs else None)
        if bad or n <= 0:
            print(f"    not executable: {bad or 'passes must be > 0'}"); continue
        log(f"    edited block: {' then '.join(_entry_txt(x) for x in block)}")
        if _ask("    run this block? [y/N] ", "n").lower() != "y":
            continue
        return {"choice": "edit", "reason": reason, "block": block, "edit": {"sample": si, "entry": int(ej), "passes": n, "N": N}}


# ---------------------------------------------------------------------------------------------------------------------------
class SimRigPlant:
    """offline test plant (ideas2 only): the paper's plant = the C++ simulator from flat at growth `growth` (gtop only) with the
    warm-start chain, every scan the rig-like rule (closedloop_v7.scan_riglike: registration shift, noise / dropout / bias,
    clean_like_loop). The loop hands it the entry it executes (pending_entry). Its true whole-sheet amp / corr against the
    dataset target state is kept per block (record)."""
    def __init__(self, workdir, target_meta, growth=1.0, core0=136, cores=8, seed_tag="rig"):
        import zlib
        from . import final_controller as FC
        import sys
        sys.path.insert(0, str(FC.GA))
        import closedloop_v7 as V7
        import argparse
        self.V7, self.growth = V7, float(growth)
        V7.WARM, V7.MEASURE = True, "riglike"
        a = argparse.Namespace(pairs=FC.FROZEN["pairs"], data=FC.FROZEN["data"], tag="rig_simplant", root=os.path.abspath(workdir), prompt_schema="auto")
        V7.setup(a)
        tgt, target_meta = (target_meta, target_meta.meta) if isinstance(target_meta, HeightField) else (None, target_meta)
        self.meta = FC.meta_of_target(target_meta, tgt, V7.P.DATA)
        if not self.meta or not self.meta.get("source"):
            raise ValueError(f"simrig needs a dataset target (its trajectory is the plant's reference): {self.meta}")
        V7.CL._BLOCK = (int(core0), int(cores))
        self.rng = np.random.default_rng(zlib.crc32(f"{self.meta['source']}|g{self.growth}|{seed_tag}".encode()))
        self.workdir, self.ops, self.k, self.next_dir, self.pending_entry, self.true = os.path.abspath(workdir), [], 0, "l2r", None, []

    def baseline(self, round_dir):
        gt = np.load(self.V7.P.DATA / self.meta["source"] / "cycles.npz")["xyz"].astype(float)
        fld = self.V7.scan_riglike(gt[0], self.rng)
        return L.Capture(None, 0, None, field=_as_hf(fld))

    def execute(self, doc, round_dir):
        e = dict(self.pending_entry)
        st = ST.LoopState(99); st.history = [dict(o) for o in self.ops]; st.record_executed({k: v for k, v in e.items() if k != "n"}, int(e["n"]))
        self.ops = st.history
        d = os.path.join(round_dir, "sim_plant")
        res = self.V7.plant_run_v7(([dict(o) for o in self.ops], d, self.meta, self.growth))
        if res.get("rc") != 0:
            raise RuntimeError(f"simulator failed: {res}")
        xyz = np.load(os.path.join(d, "cycles.npz"))["xyz"]
        fld = self.V7.scan_riglike(xyz[-1].astype(float), self.rng)
        self.k += 1
        self.true.append({"passes": int(sum(o["n"] for o in self.ops)), "exec": res.get("exec")})
        json.dump(self.true[-1], open(os.path.join(round_dir, "sim_plant_true.json"), "w"), indent=1, default=float)
        return [L.Capture(doc["toolpaths"][0]["id"], self.k, None, field=_as_hf(fld))], int(e["n"])

    def state(self):
        return {"k": self.k, "ops": self.ops, "true": self.true}

    def restore(self, d):
        self.k, self.ops, self.true = d.get("k", 0), d.get("ops", []), d.get("true", [])


def _as_hf(fld):
    if isinstance(fld, HeightField):
        return fld
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as t:
        name = t.name
    fld.save(name); h = HeightField.load(name); os.remove(name)
    return h


def _bump_xy(dz, X, Y, cx, cy, r=45.0, min_peak=0.4):
    """weighted centroid of the increment above half its maximum within r of (cx, cy); None if no clear bump"""
    win = np.isfinite(dz) & (np.hypot(X - cx, Y - cy) < r)
    if not win.any():
        return None, 0.0
    mx = float(np.nanmax(np.where(win, dz, np.nan)))
    if not np.isfinite(mx) or mx < min_peak:
        return None, mx
    m = win & (dz > 0.5 * mx); w = dz[m]
    return (float((X[m] * w).sum() / w.sum()), float((Y[m] * w).sum() / w.sum())), mx


def _detrended_increment(after, before, cx, cy):
    X, Y = np.meshgrid(after.x_mm, after.y_mm); d = (after.z_mm - before.z_mm).astype(float)
    far = np.isfinite(d) & (np.hypot(X - cx, Y - cy) > 55.0)
    if far.sum() > 50:
        A = np.c_[X[far], Y[far], np.ones(int(far.sum()))]; c, *_ = np.linalg.lstsq(A, d[far], rcond=None)
        d = d - (c[0] * X + c[1] * Y + c[2])
    return d, X, Y


def _target_bump_near(target, cx, cy, r=40.0):
    """where the TARGET wants this area's bump: the maximum of the target with its whole-sheet quadratic bend removed, within
    r mm of the (model's) area centre"""
    X, Y = np.meshgrid(target.x_mm, target.y_mm); z = target.z_mm.astype(float); m = np.isfinite(z)
    A = np.c_[np.ones(int(m.sum())), X[m], Y[m], X[m] ** 2, X[m] * Y[m], Y[m] ** 2]; c, *_ = np.linalg.lstsq(A, z[m], rcond=None)
    zb = z - (c[0] + c[1] * X + c[2] * Y + c[3] * X ** 2 + c[4] * X * Y + c[5] * Y ** 2)
    win = m & (np.hypot(X - cx, Y - cy) < r)
    if not win.any():
        return float(cx), float(cy)
    i = int(np.nanargmax(np.where(win, zb, -np.inf))); iy, ix = np.unravel_index(i, zb.shape)
    return float(X[iy, ix]), float(Y[iy, ix])


def _placement_probe_update(placement_offsets, probed, e, e_robot, before, after, target, log, max_step=25.0, apply=True):
    """DEVICE LAYER (2026-11-21, user): after the first pass of the first block on an area, compare where its bump landed with
    where the target wants it and move that angle's placement offset by the difference (clamped to max_step per axis).
    Changes the robot command only; the controller's block, history and inputs are untouched."""
    th = int(spec.orientation(e)[1])
    tb = _target_bump_near(target, float(e["cx"]), float(e["cy"]))
    d, X, Y = _detrended_increment(after, before, float(e_robot["cx"]), float(e_robot["cy"]))
    landed, peak = _bump_xy(d, X, Y, float(e_robot["cx"]), float(e_robot["cy"]))
    old = placement_offsets.get(th); old = (float(old[0]), float(old[1])) if old else (0.0, 0.0)
    info = {"angle": th, "target_bump": [round(tb[0], 1), round(tb[1], 1)], "landed_bump": None if landed is None else [round(landed[0], 1), round(landed[1], 1)],
            "peak_mm": round(float(peak), 2), "robot_area_centre": [e_robot["cx"], e_robot["cy"]], "offset_before": list(old), "applied": False}
    probed.add(th)
    if landed is None:
        log(f"  placement probe ({th:+} deg): no clear bump in the pass-1 increment (peak {peak:.2f} mm); offset kept at {old}")
        return info
    dx = max(-max_step, min(max_step, tb[0] - landed[0])); dy = max(-max_step, min(max_step, tb[1] - landed[1]))
    new = (round(old[0] + dx, 1), round(old[1] + dy, 1))
    info.update({"correction": [round(dx, 1), round(dy, 1)], "offset_if_applied": list(new)})
    if apply:
        placement_offsets[th] = new; info["applied"] = True
    log(f"  placement probe ({th:+} deg): pass-1 bump at ({landed[0]:.1f}, {landed[1]:.1f}) vs target bump ({tb[0]:.1f}, {tb[1]:.1f}), peak {peak:.2f} mm; "
        f"offset {old} {'->' if apply else 'would become'} {new} (plane-detrended increment: treat as an ESTIMATE, the operator decides)")
    return info


# ---------------------------------------------------------------------------------------------------------------------------
def run_loop_final(out, sheet, target_path, client, plant, measurer=None, profile=None, max_passes=12, max_angle=0.0, confirm=True,
                   retakes=2, run_id=None, allow_no_sheet=False, resume=False, log=None, test_physical_area=None,
                   placement_offsets=None, placement_probe=False, probe_first_part=False):
    """test_physical_area (2026-11-18, LOOP TESTS ONLY, never for a scored sheet): {cx, cy, wx, wy} replaces the area the
    ROBOT rolls (N, passes, rot of the controller's block are kept) so a scrap panel with its clamps in the way can carry a
    full planner -> controller -> robot -> scan loop; the controller's own block stays in the model history.
    placement_offsets (2026-11-21, cell calibration, as loop.run_loop): {angle_deg: (dx, dy)} mm, sheet frame; the area the
    ROBOT rolls = the controller's area shifted by the offset of its tilt angle (11-04 calibration of the sheet-turn
    rotation-centre error). Robot command only: the planner, the scheduler, the history sent to the service and every
    figure of the model's plan keep the controller's own area; the shift is recorded in round.json and toolpath.json."""
    log = log or (lambda *a: print(*a, flush=True))
    placement_offsets = {int(k): (float(v[0]), float(v[1])) for k, v in (placement_offsets or {}).items()}
    probed = set()
    if placement_offsets:
        log(f"placement calibration (robot command only, the controller is not told): {placement_offsets}")
    if placement_probe:
        log("placement probe ON (device layer, user 2026-11-21): the first block on each area is capped to 1 pass; after its scan the landing "
            "is logged as an estimate and the operator may correct that angle's offset (stop at the gate, -Resume); the controller plans every "
            "next block from the scan as usual")
    if test_physical_area:
        log(f"!!! TEST MODE: the robot rolls the area {test_physical_area} instead of the controller's area (history unchanged)")
    out = os.path.abspath(out); os.makedirs(out, exist_ok=True)
    rig = measurer is not None
    L._SHEET_FOR_FIGS[0] = sheet
    target_source = HeightField.load(target_path)
    target = L._rig_tools()[1].robust_clip(target_source, "target clipped to its robust range (the rule of the cleaned current)")
    run_id = run_id or f"{os.path.basename(out)}_{time.strftime('%Y%m%d_%H%M%S')}"
    st_path = os.path.join(out, "state.json")
    log(f"final controller service: {json.dumps(client.health())[:400]}")
    if resume:
        saved = json.load(open(st_path))
        state = ST.LoopState.from_dict(saved["state"]); state.stopped, state.reason = False, None
        plant.restore(saved.get("plant", {})); cur = HeightField.load(saved["last_field"])
        r, run_id, last_exec, last_field = saved["round"] + 1, saved["run_id"], saved.get("last_exec"), saved["last_field"]
        if rig and os.path.isdir(os.path.join(out, "round_00", "reference")):
            measurer.load(os.path.join(out, "round_00", "reference"))
        log(f"resuming run {run_id} after round {saved['round']}: {state.exec_passes} passes done")
    else:
        r0 = os.path.join(out, "round_00"); os.makedirs(r0, exist_ok=True)
        base = plant.baseline(r0)
        if base.field is not None:                                  # simulator plant: the rig-like cleaned field directly
            cur, cur_raw, flat_check = base.field, None, {"ok": True, "sim": True}
        else:
            cur_raw, pdir, flat_check = L._measure_with_retakes(lambda d: measurer.reference(os.path.dirname(d)), base.pass_dir, plant, r0,
                                                                sheet, retakes, log, "flat scan")
            if not flat_check["ok"] and hasattr(plant, "retake") and retakes > 0 and not allow_no_sheet:
                raise RuntimeError(f"the flat scan keeps a glare hole after {len(flat_check['takes'])} takes; start again")
            cur = L._clean(cur_raw, sheet)
            measurer.save(os.path.join(r0, "reference")); measurer.preview(os.path.join(r0, "reference", "reference_preview.png"), cur)
            log(f"reference: {measurer.info}")
        L._save_field(cur, os.path.join(r0, "field_reference"), profile)
        L._save_field(target, os.path.join(out, "target"), profile)
        target_source.save(os.path.join(out, "target_source.npz"))
        if cur_raw is not None:
            L._save_field(cur_raw, os.path.join(r0, "field_reference_raw"), profile)
        json.dump({"compare_cleaned_span": L._compare(cur, target, True, allow_no_sheet, log), "scan_check": flat_check, "run_id": run_id},
                  open(os.path.join(r0, "flat_metrics.json"), "w"), indent=1, default=str)
        state = ST.LoopState(max_passes, True)
        state.record_initial(L._compare(cur, target, True, allow_no_sheet, log))
        r, last_exec, last_field = 1, None, os.path.join(r0, "field_reference.npz")
    while True:
        rd = os.path.join(out, f"round_{r:02d}"); os.makedirs(rd, exist_ok=True)
        if profile is not None:
            profile.render(cur, os.path.join(rd, "cur.png")); profile.render(target, os.path.join(rd, "tgt.png"))
        t0 = time.time()
        resp = client.step(run_id, r, cur, target, state.history, executed_block=last_exec, max_passes=max_passes)
        resp["client_wait_s"] = round(time.time() - t0, 1)
        json.dump(resp, open(os.path.join(rd, "final_controller_response.json"), "w"), indent=1, default=float)
        log(f"round {r}: (service answered in {resp['client_wait_s']} s)")
        show_recommendation(resp, log)
        plan = (resp.get("plan") or {}).get("was")
        rec = {"round": r, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "before": state.scans[-1], "plan": plan,
               "plan_raw": (resp.get("plan") or {}).get("plan"), "model_secs": (resp.get("timing") or {}).get("plan_s"),
               "history_given": [dict(o) for o in state.history], "scripted": False, "controller": "server-final",
               "final_controller": {"recommendation": {"stop": resp.get("stop"), "block": (resp.get("block") or {}).get("entries"),
                                                       "b": (resp.get("block") or {}).get("b"), "s": (resp.get("block") or {}).get("s"),
                                                       "N": (resp.get("block") or {}).get("N"), "rule": (resp.get("block") or {}).get("rule"),
                                                       "forecast": (resp.get("block") or {}).get("forecast")},
                                    "measured": resp.get("measured"), "g_hat": resp.get("g_hat"), "picked_sample": (resp.get("samples") or {}).get("picked"),
                                    "timing": resp.get("timing"), "client_wait_s": resp["client_wait_s"]}}
        _save_partial = lambda status: L._save_round(out, rd, rec, status)      # noqa: E731
        _save_partial("recommendation received; awaiting the operator")
        dec = operator_decision(resp, sheet, max_angle, confirm, log)
        rec["final_controller"]["operator"] = dec
        log(f"  operator: {dec['choice']}" + (f" ({dec['reason']})" if dec.get("reason") else "")
            + (f" -> {' then '.join(_entry_txt(x) for x in dec['block'])}" if dec.get("block") else ""))
        if dec["choice"] == "stop" or (dec["choice"] == "accept" and dec["block"] is None):
            state.stopped = True
            state.reason = (resp["stop"]["reason"] if dec["choice"] == "accept" and resp.get("stop") else "stopped_by_operator")
            rec["stop"] = state.reason
            L._save_round(out, rd, rec, f"stopped ({state.reason})"); L._record(out, rec)
            break
        if dec["choice"] == "skip":                                 # no passes; re-scan (scan pose only) and ask again
            rec["executed"] = 0
            if hasattr(plant, "retake") and not isinstance(plant, SimRigPlant):
                pdir = plant.retake(rd, 1)
                fld_raw, pdir, scan_check = L._measure_with_retakes(measurer.field, pdir, plant, rd, sheet, retakes, log, f"round {r} re-scan")
                cur = L._clean(fld_raw, sheet)
                name = os.path.join(rd, "field_rescan"); L._save_field(cur, name, profile); L._save_field(fld_raw, name + "_raw", profile)
                last_field = name + ".npz"
                rec["measured"] = [{"rescan": True, "field": last_field, **L._compare(cur, target, True, allow_no_sheet, log)}]
            else:
                log("  (this plant cannot re-scan: the last field is planned on again)")
            last_exec = []
            L._save_round(out, rd, rec, "skipped: re-scanned, no passes"); L._record(out, rec)
            json.dump({"round": r, "last_field": last_field, "state": state.to_dict(), "plant": plant.state(), "run_id": run_id,
                       "last_exec": last_exec}, open(st_path, "w"), indent=1)
            r += 1
            continue
        block = dec["block"]
        if placement_probe and not test_physical_area and int(spec.orientation(block[0])[1]) not in probed:
            th0 = int(spec.orientation(block[0])[1]); probed.add(th0)
            if len(block) > 1 or int(block[0]["n"]) > 1:
                keep = dict(block[0]) if (probe_first_part and len(block) > 1) else {**block[0], "n": 1}
                rec["placement_probe_cap"] = {"recommended_block": [dict(x) for x in block], "executed_block": [keep], "angle": th0,
                                              "first_part_kept": bool(probe_first_part and len(block) > 1),
                                              "note": "device-layer placement probe (user, 2026-11-21): the first block on a new area is capped to 1 pass; "
                                                      "the operator checks where its bump landed and may correct that angle's offset before the next block, "
                                                      "which the controller plans from the scan as usual"}
                log(f"  placement probe: first block on the {th0:+} deg area capped to {_entry_txt(keep)} (controller recommended "
                    f"{' then '.join(_entry_txt(x) for x in block)}); the landing is checked at the next scan")
                block = [keep]
        rec.update({"chosen": block[0], "block_passes": sum(int(x["n"]) for x in block),
                    "block_reason": f"final controller ({dec['choice']} by the operator)" + (f": {dec['reason']}" if dec.get("reason") else "")})
        # ---- execute the block (one toolpath per part: the robot side takes one toolpath per call)
        parts = [rd] if len(block) == 1 else [os.path.join(rd, f"part{i + 1}") for i in range(len(block))]
        caps_all, executed_all, executed_ops, stop_now = [], 0, [], None
        for i, (e, pd) in enumerate(zip(block, parts)):
            os.makedirs(pd, exist_ok=True)
            e_robot = e
            if test_physical_area:
                e_robot = {**e, **{k: float(test_physical_area[k]) for k in ("cx", "cy", "wx", "wy")}}
                rec["test_physical_area"] = {"robot_entry": e_robot, "model_entry": e}
                log(f"  !!! TEST MODE: robot rolls {_entry_txt(e_robot)} (the model's block {_entry_txt(e)} is what the history records)")
            off = None
            if placement_offsets:
                e_phys, off = L._physical_wa(e_robot, placement_offsets)
                rec.setdefault("placement_offsets", []).append({"part": i + 1, "offset_mm": off, "model_entry": dict(e),
                                                                "robot_entry": dict(e_phys)})
                if off:
                    log(f"  placement calibration: robot area centre ({e_robot['cx']}, {e_robot['cy']}) -> ({e_phys['cx']}, {e_phys['cy']}), offset {off} mm")
                    e_robot = e_phys
                else:
                    log(f"  placement calibration: no offset for this area's angle ({_entry_txt(e_robot)})")
            e_exec, ex_probe = e, 0
            tps, problems = spec.plan_to_toolpaths([e_robot], sheet, allow_rot90=False, max_angle=max_angle)
            if problems:
                stop_now = f"not_executable: {problems[0][1]}"; break
            spec.set_directions(tps[0], getattr(plant, "next_dir", "l2r"))
            doc = spec.toolpath_doc(sheet, tps, note=f"round {r} part {i + 1}/{len(block)}: final controller, {dec['choice']}, {e_exec['n']} pass(es)"
                                    + (f"; placement offset {off} mm applied to the controller's area ({e['cx']}, {e['cy']})" if off else ""))
            spec.write_toolpath_json(os.path.join(pd, "toolpath.json"), doc)
            spec.preview_png(doc, os.path.join(pd, "preview.png"))
            rec.setdefault("toolpaths", []).append({"part": i + 1, "id": tps[0]["id"], "passes": int(e_exec["n"]), "probe_passes": ex_probe, "robot": tps[0]["robot"],
                                                    "pass_directions": tps[0]["pass_directions"], "dir": pd})
            if i == 0:
                rec["toolpath"] = {"id": tps[0]["id"], "passes": int(e["n"]), "robot": tps[0]["robot"], "ours": tps[0].get("ours"),
                                   "start_direction": tps[0]["start_direction"], "pass_directions": tps[0]["pass_directions"],
                                   "n_pivots": len(tps[0]["pivots_xy_mm"])}
                if pd != rd:                                      # two-part block: the figure reads round_NN/preview.png (2026-11-21)
                    shutil.copyfile(os.path.join(pd, "preview.png"), os.path.join(rd, "preview.png"))
                try:
                    L._replan_png(rd, {**rec, "block_passes": rec["block_passes"]}, doc, cur, target)
                except Exception as ex:                               # figures never stop the loop
                    log(f"  (replan figure not written: {ex})")
            L._save_round(out, rd, rec, f"executing part {i + 1}/{len(block)} on the robot")
            if hasattr(plant, "pending_entry"):
                plant.pending_entry = e_exec
            log(f"  >>> part {i + 1}/{len(block)}: {_entry_txt(e_exec)}  passes {' '.join('L>R' if d == 'l2r' else 'R>L' for d in tps[0]['pass_directions'])}")
            caps, executed = plant.execute(doc, pd)
            state.record_executed({k: v for k, v in e.items() if k != "n"}, executed)
            caps_all += caps; executed_all += executed
            if executed + ex_probe > 0:
                executed_ops.append({**e, "n": int(executed + ex_probe)})
            if executed < int(e_exec["n"]):
                stop_now = f"robot executed {executed}/{e_exec['n']} pass(es) of part {i + 1}"; break
        rec.update({"robot": getattr(plant, "last_run", None), "executed": executed_all, "executed_block": executed_ops, "measured": []})
        if caps_all:
            cap = caps_all[-1]
            if cap.field is not None:
                fld, fld_raw, scan_check = cap.field, None, {"ok": True, "sim": True}
            else:
                fld_raw, pdir, scan_check = L._measure_with_retakes(measurer.field, cap.pass_dir, plant, rd, sheet, retakes, log,
                                                                    f"round {r} capture {cap.pass_index}")
                fld = L._clean(fld_raw, sheet)
            name = os.path.join(rd, f"field_{cap.toolpath_id}_capture{cap.pass_index}")
            L._save_field(fld, name, profile)
            if fld_raw is not None:
                L._save_field(fld_raw, name + "_raw", profile)
                L._scan_png(os.path.join(rd, f"scan_{cap.toolpath_id}_capture{cap.pass_index}.png"),
                            measurer.full_view(cap.pass_dir) if hasattr(measurer, "full_view") else None, fld, sheet,
                            f"round {r}: block measurement after {executed_all} pass(es)")
            if placement_probe and not test_physical_area and rec.get("placement_probe_cap"):
                e0 = block[0]; rob = (rec.get("placement_offsets") or [{}])[-1].get("robot_entry") or e0
                info = _placement_probe_update(dict(placement_offsets), set(), e0, rob, cur, fld, target, log, apply=False)
                rec.setdefault("placement_probe", []).append({"part": 1, "from_block_scan": True, "estimate_only": True, **info})
            mt = L._compare(fld, target, True, allow_no_sheet, log)
            state.record_block(mt, executed_ops[-1] if executed_ops else block[-1], r, cap.toolpath_id, cap.pass_index, max(executed_all, 1))
            cur, last_field = fld, name + ".npz"
            true = (getattr(plant, "true", None) or [None])[-1] if isinstance(plant, SimRigPlant) else None
            rec["measured"].append({"capture": cap.pass_index, "block_measurement": True, "field": name + ".npz", "scan_check": scan_check,
                                    "z_min": round(float(fld.zrange()[0]), 2), "z_max": round(float(fld.zrange()[1]), 2), **mt,
                                    **({"sim_true_whole_sheet": true} if true else {})})
            log(f"  measured after {executed_all} pass(es): span-amp {mt['amp_ratio']:.3f} corr {mt['pattern_corr']:.4f} (the service reads the "
                f"projection progress at the next request)" + (f"; simulator truth: whole-sheet amp {true['exec']['amp_ratio']:.3f} corr "
                                                              f"{true['exec']['pattern_corr']:.4f}" if true and true.get("exec") else ""))
        else:
            stop_now = stop_now or f"no capture after round {r}"
        last_exec = executed_ops
        json.dump({"round": r, "last_field": last_field, "state": state.to_dict(), "plant": plant.state(), "run_id": run_id,
                   "last_exec": last_exec}, open(st_path, "w"), indent=1, default=float)
        L._save_round(out, rd, {**rec, "stop": stop_now}, "done: executed and measured"); L._record(out, {**rec, "stop": stop_now})
        if stop_now:
            state.stopped, state.reason = True, stop_now
            break
        r += 1
    json.dump({"round": r, "last_field": last_field, "state": state.to_dict(), "plant": plant.state(), "run_id": run_id,
               "last_exec": last_exec}, open(st_path, "w"), indent=1, default=float)
    try:
        L.summary(out, state, target, HeightField.load(last_field))
    except Exception as e:                                         # noqa: BLE001
        log(f"(summary figure not written: {e})")
    log(f"stopped: {state.reason}; {state.exec_passes} passes")
    return state
