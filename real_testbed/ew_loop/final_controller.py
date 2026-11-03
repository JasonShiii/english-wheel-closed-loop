"""The paper's FINAL controller (ICCPS 2027, frozen 2026-11-02) as a service for `rigloop --controller server-final`.

Server side only (ideas2: needs the research repo, the planner server and the C++ simulator). Per request the service takes
the current cleaned field, the target, the executed history and the run id, and returns what closedloop_v7.py decides in the
paper's configuration `rulesched_cfg3_ruleteacher_sharedstop` (MEASUREMENT_CONSISTENCY_FIX_2026-11-01 sections 14-16):

  measurement   amp_projection: progress a = <M - mean, T - mean> / <T - mean, T - mean> (closedloop_v7.ctrl_compare, patched into
                ew_loop.state.compare by closedloop_v7.setup, so the stop rules read it too), shape corr
  growth id     after every executed block: matched filter <dM, dN> / <dN, dN> of the measured fields before / after the block
                against the nominal (growth 1.0) fields of the executed history (warm-start chain, nominal_chain.py), kept only if
                the nominal increment exceeds closedloop_safeprobe.MIN_INC_MM; g_hat = median, clipped to G_CLIP (prior 1.0)
  stop          the shared stop rules = ew_loop.state.LoopState.check_stop (amp >= 1.00 / >= 0.95, budget, no progress) before
                planning and block_size(first entry, 3) == 0 after planning ("next_pass_would_overshoot")
  plan          closedloop_v7.VerifierPlanner: best-of-8 at T 0.7 from the augreal ck1170 planner server, every admissible draw
                scored by outcome/om_D_trainfresh_riglike.pkl on the loop's own prompt (state_from_prompt), pick |a-0.98| +
                3 max(0, a-1.03); the prompt = closedloop_v7.model_inputs_v7 (schema v8, the same renderer and history writer)
  block         the tuned threshold rule (rule_search_rover2.json "best"): closedloop_v7.RuleTaskV7.rule_action, called on this
                run's LoopState and g_hat with the nominal forecast of history + min(4, passes left) passes of the plan's first
                entry, then the action -> passes exactly as closedloop_rlsched.RLTask.forecast_done (full passes at the plan's N,
                the last one at N' = round(s N) clipped to [N_min, N]); cap-hits and shared-stop conversions as in the paper

Per-run state (LoopState, g_hat ratios, last field) is kept under <root>/runs/<run id>/ and survives a restart; a request for a
round already answered first restores the state saved before that round (idempotent retries). The client reports what was
EXECUTED (it may differ from the recommendation); that, not the recommendation, enters the history and the growth id.

  python -m ew_loop.cli serve-fields --controller final --port 8766 --planner-url http://127.0.0.1:18631 --core0 120 --cores 8
"""
import argparse
import base64
import copy
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from . import model_newsheet as MN           # puts newsheet_pipeline and GPT-astra_comparison on sys.path

GA = Path(MN.ROOT) / "experiments" / "GPT-astra_comparison"
FROZEN = {"ckpt": str(Path(MN.NEWSHEET) / "runs" / "t1.0_D1-5_augreal_visfix_e3" / "checkpoint-1170"),
          "name": "ours-D1-5-augreal-visfix-e3-ck1170",
          "pairs": str(Path(MN.NEWSHEET) / "pairs_231x174_t1.0_D1-5_clean"),
          "data": str(Path(MN.NEWSHEET) / "data_231x174_t1.0_D1-5_nominal"),
          "om": str(GA / "outcome" / "om_D_trainfresh_riglike.pkl"),
          "rule_json": str(GA / "results_231x174_t1.0_D1-5_rlsched_riglike_proj_caphits_sharedstop" / "rule_search_rover2.json"),
          "K": 8, "T": 0.7, "max_passes": 12, "n": 3}
for k, v in (("SCHED_MEAS", "proj"), ("SCHED_UNDER_C", "1"), ("SCHED_CAP_HITS", "1"), ("SCHED_STOP", "shared")):
    os.environ.setdefault(k, v)                 # the environment of the paper's runs (sched_env reads it at import)


PREVIEW_DATA = ("../data_D_preview_2026-10-30",)                # dataset-D preview trajectories (Demo B = preview D5/t019), relative to the data


def meta_of_target(tgt_meta, tgt=None, data=None):
    """{'source': 'D1/t065', 'cut': [0, 5]} from a rig target's meta ('D1/t065 cycle 5 of 5'), or None. With the target field and
    the data folder, the trajectory is looked up where its state at that cycle IS the target (shape corr > 0.98): the nominal data
    first, then the preview data (the rig's Demo B target D5_t019_c10 is preview D5/t019; nominal D5/t019 is another trajectory)"""
    m = re.match(r"\s*(\S+/\S+) cycle (\d+)", str((tgt_meta or {}).get("source", "")))
    if not m:
        return None
    src, cut = m.group(1), int(m.group(2))
    meta = {"source": src, "cut": [0, cut], "category": src.split("/")[0]}
    if tgt is None or data is None:
        return meta
    import pairs as NP
    for pre in ("",) + tuple(p + "/" for p in PREVIEW_DATA):
        f = Path(data) / (pre + src) / "cycles.npz"
        if not f.exists():
            continue
        xyz = np.load(f)["xyz"]
        if cut >= len(xyz):
            continue
        fld = NP.field_of(xyz[cut].astype(float))
        mk = np.isfinite(fld.z_mm) & np.isfinite(tgt.z_mm)
        a, b = fld.z_mm[mk] - fld.z_mm[mk].mean(), tgt.z_mm[mk] - tgt.z_mm[mk].mean()
        c = float(np.corrcoef(a, b)[0, 1]) if mk.sum() > 50 else 0.0
        if c > 0.98:
            return {**meta, "source": pre + src, "match_corr": round(c, 4)}
    return {**meta, "source": None, "unmatched": src}


class _OneHot:
    def __init__(self, act, obs_names):
        self.act, self.obs_names = act, obs_names

    def probs(self, x):
        import sched_env as E
        p = np.zeros((len(x), E.N_ACT)); p[:, self.act] = 1.0
        return p


class _RuleView:
    """the attributes closedloop_v7.RuleTaskV7.rule_action reads (theta, state, g_hat): the method runs unchanged on them"""
    def __init__(self, theta, state, g_hat):
        self.theta, self.state, self.g_hat = theta, state, g_hat


class FinalController:
    def __init__(self, planner_url, root, core0=120, cores=8, ckpt=None, om=None, rule_json=None, pairs=None, data=None):
        sys.path.insert(0, str(GA))
        import closedloop_v7 as V7
        import closedloop_newsheet as CL
        import closedloop_safeprobe as SP
        import closedloop_rlsched as RLS
        import planners as PL
        import amp_projection as AP
        import sched_env as E
        import planner_server as PS
        self.V7, self.CL, self.SP, self.RLS, self.PL, self.AP, self.E = V7, CL, SP, RLS, PL, AP, E
        # the paper's configuration (closedloop_v7 flags of _cfg3_ruleteacher_sharedstop)
        V7.MEASURE, V7.WARM, V7.VERIFIER_K, V7.AMP_EST = "riglike", True, FROZEN["K"], "projection"
        V7.RL_CAP_HITS, V7.RL_STOP_SHARED, V7.RL_GHAT_FIXED = True, True, None
        a = argparse.Namespace(pairs=pairs or FROZEN["pairs"], data=data or FROZEN["data"], tag="rig_final_controller",
                               root=str(Path(root).resolve()), prompt_schema="auto")
        self.prof, self.root = V7.setup(a)                         # ST.compare -> projection, SP.nominal_spans -> projection, caches
        assert V7.ST.compare is V7.ctrl_compare and SP.nominal_spans is V7.nominal_spans_proj, "projection estimator not in place"
        assert V7.SCHEMA == "v8", V7.SCHEMA
        RLS.K_FC = 4
        CL._BLOCK = (int(core0), int(cores))                       # simulator cores (nominal_chain / run_execute)
        self.ckpt, self.om_path = ckpt or FROZEN["ckpt"], om or FROZEN["om"]
        self.rule_json = rule_json or FROZEN["rule_json"]
        self.theta = json.load(open(self.rule_json))["best"]
        base = PS.RemoteOursPlanner(self.ckpt, planner_url, name=FROZEN["name"])
        self.planner = V7.VerifierPlanner(base, self.om_path, FROZEN["K"], FROZEN["T"])
        self.planner_url, self.lock, self._meta_cache = planner_url, threading.Lock(), {}
        self.info = {"controller": "final (ICCPS 2027 frozen 2026-11-02)", "ckpt": self.ckpt, "planner_url": planner_url,
                     "verifier": {"K": FROZEN["K"], "T": FROZEN["T"], "om": self.om_path}, "rule_json": self.rule_json,
                     "theta": self.theta, "amp_estimator": "projection", "growth_id": "matched filter, median, clip",
                     "stop": "shared (state.py check_stop + block_size 0)", "cap_hits": True, "max_passes": FROZEN["max_passes"],
                     "sim_cores": list(CL._BLOCK), "root": str(self.root), "schema": V7.SCHEMA, "profile": "newsheet-v8"}

    # ---- per-run state --------------------------------------------------------------------------------
    def _rdir(self, run_id):
        d = Path(self.root) / "runs" / re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_id))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load(self, run_id, rnd):
        d = self._rdir(run_id)
        snap = d / f"state_before_round_{rnd:02d}.json"
        if snap.exists():                                          # a retry of this round: start from the state before it
            return json.load(open(snap)), True
        p = d / "state.json"
        return (json.load(open(p)) if p.exists() else None), False

    def _save(self, run_id, rnd, s, before=False):
        d = self._rdir(run_id)
        json.dump(s, open(d / (f"state_before_round_{rnd:02d}.json" if before else "state.json"), "w"), indent=1, default=float)

    # ---- the decision ---------------------------------------------------------------------------------
    def step(self, run_id, rnd, cur, tgt, history, executed_block=None, prev=None, max_passes=None):
        """one round of the final controller. cur / tgt / prev: ew_loop.frames.HeightField (the cleaned field, the target the
        loop compares with, the field the previous request saw); history: executed ops (wa dicts with n, merged as
        LoopState.record_executed does); executed_block: the ops executed since the previous request (default: from the history)."""
        V7, SP, AP, E, RLS, PL = self.V7, self.SP, self.AP, self.E, self.RLS, self.PL
        ST = V7.ST                                                 # ew_loop.state with compare = the projection (setup)
        t0, T = time.time(), {}
        key = json.dumps(getattr(tgt, "meta", {}) or {}, sort_keys=True, default=str)
        if key not in self._meta_cache:
            self._meta_cache[key] = meta_of_target(getattr(tgt, "meta", {}), tgt, self.PL.DATA)
        meta = self._meta_cache[key]
        if meta is not None and meta.get("source") is None:
            meta = None                                           # not a dataset state: the verifier's span from the field
        d = self._rdir(run_id); rd = d / f"round_{rnd:02d}"; rd.mkdir(exist_ok=True)
        saved, retry = self._load(run_id, rnd)
        mp = int(max_passes or FROZEN["max_passes"])
        H_now = int(sum(int(o["n"]) for o in history))
        diag = {"run_id": run_id, "round": rnd, "retry_of_round": retry, "target_meta": meta}
        if saved is None:                                          # first request of a run: the flat (or first) measured state
            st = ST.LoopState(mp, True)
            st.record_initial(ST.compare(cur, tgt))
            s = {"state": st.to_dict(), "ratios": [], "g_hat": 1.0, "H_last": H_now, "last_field": None, "g_log": []}
            st.history = [dict(o) for o in history]
            if H_now:
                diag["warning"] = f"first request with {H_now} executed passes: the growth id starts here"
        else:
            s = copy.deepcopy(saved)
            st = ST.LoopState.from_dict(s["state"])
        self._save(run_id, rnd, s if saved is None else saved, before=True) if not retry else None
        H_prev = int(s["H_last"])
        prev_field = prev
        if prev_field is None and s.get("last_field") and os.path.exists(s["last_field"]):
            from ew_loop.frames import HeightField
            prev_field = HeightField.load(s["last_field"])
        m_now = ST.compare(cur, tgt)
        gid = {"g_hat_before": round(float(s["g_hat"]), 4)}
        if saved is not None:
            if H_now > H_prev:                                     # a block was executed since the last request
                blk = executed_block or _diff_ops(st.history, history)
                k = H_now - H_prev
                st.history = [dict(o) for o in history]
                st.record_block(m_now, blk[-1] if blk else history[-1], rnd, f"r{rnd:02d}", H_now, k)
                # growth identification on the nominal run of the executed history (warm-start chain; usually cached)
                t1 = time.time()
                nd = rd / "nominal_history"
                res = self._nominal([dict(o) for o in history], nd)
                T["nominal_history_s"] = round(time.time() - t1, 1)
                if res.get("rc") == 0 and prev_field is not None:
                    nom = SP.nominal_spans(nd, tgt)
                    inc_nom = nom[H_now] - nom[H_prev]
                    gid.update(inc_nominal=round(inc_nom, 3), H_before=H_prev, H_after=H_now)
                    if inc_nom > SP.MIN_INC_MM:
                        try:
                            xyz = np.load(nd / "cycles.npz")["xyz"]
                            NP = V7.NP
                            nb, na = NP.field_of(xyz[H_prev].astype(float)), NP.field_of(xyz[H_now].astype(float))
                            rho, ncells = AP.increment_ratio(cur, prev_field, na, nb, tgt)
                            s["ratios"].append(round(float(rho), 4))
                            gid.update(ratio=round(float(rho), 4), ratio_rule="<dM, dN> / <dN, dN>", ratio_cells=ncells)
                        except Exception as e:                       # noqa: BLE001 (the paper's rule: no ratio from this block)
                            gid["skipped"] = f"matched filter failed: {e!r}"
                    else:
                        gid["skipped"] = f"nominal increment {inc_nom:.2f} <= {SP.MIN_INC_MM}"
                    s["g_hat"] = float(np.clip(np.median(s["ratios"]), *SP.G_CLIP)) if s["ratios"] else 1.0
                else:
                    gid["skipped"] = "no nominal run" if res.get("rc") != 0 else "no previous field"
            elif H_now == H_prev:                                  # re-scan without passes: the latest state is replaced
                st.scans[-1] = {**st.scans[-1], **m_now}
                gid["skipped"] = "re-scan (no passes since the last request)"
            else:
                raise ValueError(f"history has {H_now} passes, fewer than the {H_prev} the service saw for run {run_id}")
        gid.update(g_hat=round(float(s["g_hat"]), 4), support=len(s["ratios"]))
        s["g_log"].append({"round": rnd, **gid})
        fp = rd / "field_cur.npz"; cur.save(str(fp)); s["last_field"], s["H_last"] = str(fp), H_now
        tgt.save(str(rd / "target.npz"))
        out = {"diag": diag, "measured": m_now, "g_hat": gid, "stop": None, "plan": None, "block": None, "samples": None}
        # ---- the shared stop rules before planning
        if st.check_stop():
            out["stop"] = {"reason": st.reason, "rule": "state.check_stop (shared)"}
            return self._finish(run_id, rnd, s, st, out, T, t0)
        # ---- plan: best-of-8 verifier on the paper's prompt
        V7.PROMPT_G, V7.CURRENT_META = 1.0, meta or {"source": None, "cut": [0, 0]}
        t1 = time.time()
        cur_png, tgt_png, prompt = V7.model_inputs_v7(cur, tgt, [dict(o) for o in history], rd, rnd)
        if meta is None:                                         # not a dataset target: the verifier's target span from the field
            self._sg_override(tgt)
        self.planner.last = None
        ans, attempts = PL.plan_with_retry(self.planner, cur_png, tgt_png, prompt, tag=f"rig_{run_id}_r{rnd:02d}")
        T["plan_s"] = round(time.time() - t1, 1)
        was = V7.plan_to_was(ans["plan"]) if ans.get("plan") else None
        out["plan"] = {"raw": ans.get("raw"), "plan": ans.get("plan"), "was": was, "attempts": len(attempts)}
        out["samples"] = self.planner.last
        out["samples_was"] = []                                    # every sample's plan as executable entries (operator's edit menu)
        for c in (self.planner.last or {}).get("candidates", []):
            try:
                out["samples_was"].append(V7.plan_to_was(c["plan"]) if c.get("plan") and not c.get("problem") else None)
            except Exception:                                      # noqa: BLE001
                out["samples_was"].append(None)
        if not was:
            out["stop"] = {"reason": "plan_failure", "rule": "no executable plan"}
            return self._finish(run_id, rnd, s, st, out, T, t0)
        w = was[0]
        k0 = st.block_size(w, FROZEN["n"])
        if k0 == 0:                                                # the shared block_size rule
            out["stop"] = {"reason": "next_pass_would_overshoot", "rule": f"state.block_size = 0: {st.block_reason}"}
            return self._finish(run_id, rnd, s, st, out, T, t0)
        # ---- nominal forecast of history + min(4, passes left) passes of the first entry
        k_fc = min(RLS.K_FC, st.max_passes - st.exec_passes)
        t1 = time.time()
        fd = rd / "forecast"
        res = self._nominal([dict(o) for o in history] + [{**w, "n": k_fc}], fd)
        T["forecast_s"] = round(time.time() - t1, 1)
        if res.get("rc") != 0:
            out["stop"] = {"reason": f"sim_error: forecast {res.get('error', res.get('rc'))}", "rule": "forecast failed"}
            return self._finish(run_id, rnd, s, st, out, T, t0)
        H = st.exec_passes
        nom = SP.nominal_spans(fd, tgt)
        # ---- the tuned rule (closedloop_v7.RuleTaskV7.rule_action, unchanged) + the paper's mask and passes
        act, info = V7.RuleTaskV7.rule_action(_RuleView(self.theta, st, float(s["g_hat"])), nom, H, k_fc, w)
        pol = V7._CapHitsPolicy(_OneHot(act, ("x",)), int(w["n"]), stop_ok=False)
        act2 = int(np.argmax(pol.probs(np.zeros((1, 1)))[0]))
        nb, sc = E.decode(act2)
        left = st.max_passes - st.exec_passes                        # closedloop_rlsched.RLTask.forecast_done
        n_full = nb - 1 if nb <= left else max(left - 1, 0)
        Nmin = RLS.feasible_min(w)
        Np = int(np.clip(round(sc * w["N"]), min(Nmin, w["N"]), w["N"]))
        entries = [{**w, "n": n_full + 1}] if Np == w["N"] else ([{**w, "n": n_full}] if n_full else []) + [{**w, "N": Np, "n": 1}]
        span_t, amp = st.scans[-1]["span_tgt"], st.scans[-1]["amp_ratio"]
        g_used = float(s["g_hat"]) ** self.theta["kappa"]
        inc = [max(nom[H + j + 1] - nom[H + j], 0.0) / span_t for j in range(k_fc)]
        out["block"] = {"entries": entries, "b": nb, "s": sc, "N": Np, "N_plan": int(w["N"]), "n_full": n_full, "passes": sum(e["n"] for e in entries),
                        "action": act2, "rule": info, "shared_block_size": k0, "shared_block_reason": st.block_reason,
                        "forecast": {"k_fc": k_fc, "nominal_spans": [round(x, 3) for x in nom], "increments_amp": [round(x, 4) for x in inc],
                                     "pred_amp_full_passes": [round(amp + g_used * sum(inc[:j]), 4) for j in range(k_fc + 1)], "g_used": round(g_used, 4)}}
        return self._finish(run_id, rnd, s, st, out, T, t0)

    def _finish(self, run_id, rnd, s, st, out, T, t0):
        s["state"] = st.to_dict()
        self._save(run_id, rnd, s)
        T["total_s"] = round(time.time() - t0, 1)
        out["timing"] = T
        out["state"] = {"exec_passes": st.exec_passes, "max_passes": st.max_passes, "scans": len(st.scans), "replans": len(st.replans)}
        json.dump(out, open(self._rdir(run_id) / f"round_{rnd:02d}" / "decision.json", "w"), indent=1, default=float)
        return out

    def _nominal(self, ops, d):
        """the nominal (growth 1.0) run of ops: closedloop_v7.plant_run_v7's warm-start call (nominal_chain.run, the same chain
        root and core block) without its dataset scoring (rig targets need not be dataset states)"""
        import nominal_chain as NC
        os.makedirs(d, exist_ok=True)
        rc, wall, segs = NC.run(ops, str(d), *self.CL._BLOCK, growth=1.0, growth_keys=("gtop",), root=self.V7.CHAIN, timeout=self.CL.SIM_TIMEOUT)
        return {"rc": rc, "wall": wall}

    def _sg_override(self, tgt):
        """VerifierPlanner reads the target's whole-sheet span from the dataset; a non-dataset target uses its field's ptp"""
        z = tgt.z_mm[np.isfinite(tgt.z_mm)]
        sg = float(np.round(np.ptp(z), 2))
        OM = self.planner.OM
        if not getattr(OM, "_rig_patched", False):
            orig = OM.target_span
            OM.target_span = lambda data, source, cut1: (OM._rig_sg if source is None else orig(data, source, cut1))
            OM._rig_patched = True
        OM._rig_sg = sg


def _diff_ops(old, new):
    """the ops of `new` beyond `old` (both merged as LoopState.record_executed merges)"""
    import copy as _c
    new = _c.deepcopy(new)
    if not old:
        return new
    j = len(old) - 1
    out = []
    if len(new) > j and int(new[j]["n"]) > int(old[j]["n"]):
        out.append({**new[j], "n": int(new[j]["n"]) - int(old[j]["n"])})
    return out + new[j + 1:]


def serve_final(host="0.0.0.0", port=8766, **kw):
    """HTTP front: GET /health; POST /final_step {run_id, round, cur_npz, tgt_npz, prev_npz?, history, executed_block?,
    max_passes?} -> the step() dict. One request at a time (one simulator core block, one planner server)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from .frames import HeightField
    ctl = FinalController(**kw)
    print(f"[serve-final] {json.dumps(ctl.info)}\n[serve-final] http://{host}:{port}", flush=True)

    def field_from_b64(b64):
        if not b64:
            return None
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as t:
            t.write(base64.b64decode(b64)); name = t.name
        f = HeightField.load(name); os.remove(name); return f

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj, default=float).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"ok": True, "profile": "newsheet-v8", "controller": "final", **ctl.info})
            self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/final_step":
                return self._send(404, {"error": "unknown path"})
            try:
                q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                cur, tgt, prev = field_from_b64(q["cur_npz"]), field_from_b64(q["tgt_npz"]), field_from_b64(q.get("prev_npz"))
                with ctl.lock:
                    res = ctl.step(q["run_id"], int(q["round"]), cur, tgt, q.get("history", []), q.get("executed_block"), prev,
                                   q.get("max_passes"))
                self._send(200, res)
            except Exception as e:                                   # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._send(500, {"error": repr(e)})

        def log_message(self, fmt, *args):
            print("[serve-final]", fmt % args, flush=True)

    ThreadingHTTPServer((host, port), H).serve_forever()
