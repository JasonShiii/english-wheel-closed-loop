"""The RSS 2027 main line (frozen 2026-11-13) as a service for `rigloop --controller server-final`: the same request / response
contract as final_controller.py (the robot PC's client and loop are unchanged), the decisions of closedloop_dual.py's DualTaskV7
through sched_dualcontrol/dual_core.py (ONE decision function for the simulator loop and the rig; RIG_RISK_REVIEW section 4).

Per round the service takes the cleaned field, the target, the executed history and the run id, and returns:
  measurement   the cleaned field is DENOISED (scan_denoiser/model.pt, as every simulator scan was) and read as the robust-span
                ratio against the target (DUAL_AMP=span); the raw-cleaned reading is reported next to it
  growth id     after every executed block: matched-filter ratio <dM, dN> / <dN, dN> of the denoised fields before / after the block
                against the nominal (growth 1.0) chain of the executed history, into the Gaussian belief on log g (prior U[0.6, 1.6]
                in log, first-block attenuation correction K1)
  amplitude     Gaussian belief on the noise-free reading: predicted through the EXECUTED block with the pre-block growth belief
                (the operator may have edited the block), updated with the denoised reading (fitted noise curve measfit_span_denoised)
  stop          dual_core.stop_before_plan: budget 16, converged (a_mu >= 0.95, corr >= 0.99), overshoot (a_mu >= 1.0), amp_reached_shape_off,
                no_progress; then after planning the smallest-step risk test (P(over | belief) of one pass at N_min > delta 0.1)
  plan          best-of-8 verifier on the augreal ck1170 planner (as the ICCPS controller), prompt from the denoised field
  block         the RL policy under the chance-constraint shield (N2): among the blocks with P(over | belief) <= delta, bounded by
                ceil(hits / g_hat) <= 8 and the budget; (k full passes at N, last pass at N' = s N) exactly as in the simulator
The environment of the frozen test runs (sim_runs/test_designc2_spanden/.../config.json) is set below before the branch modules load.

  python -m ew_loop.cli serve-fields --controller dual --port 8767 --planner-url http://127.0.0.1:18631 --core0 120 --cores 8
"""
import argparse
import copy
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ---- the frozen main line's environment (config.json of the N2 test runs + the DUAL_* flags of their traces) --------------
FROZEN_ENV = {"SCHED_STOP": "unified", "SCHED_MEAS": "proj", "SCHED_UNDER_C": "1", "SCHED_R_OVER": "-2", "SCHED_SETTING": "scalar",
              "SCHED_G_RANGE": "0.6,1.6", "SCHED_JITTER": "0.03", "SCHED_READDOWN": "1,1", "SCHED_DELTA": "0.1",
              "SCHED_FIDELITY": "1", "SCHED_KMAX": "8", "SCHED_CAP_HITS": "g", "SCHED_BUDGET": "16", "SCHED_AMPBEL": "1",
              "SCHED_SIG_FC": "0.23", "SCHED_OBS": "beliefamp", "SCHED_A_OVER": "1.040", "SCHED_PROBE_FIRST": "0", "SCHED_SHIELD": "1",
              "DUAL_AMP": "span", "DUAL_DENOISE": "1", "DUAL_PROBE_FIRST": "0", "DUAL_SHIELD": "1", "DUAL_FIX_G": "0",
              "DUAL_SHAPE_BOUND": "0", "DUAL_SHAPE_SD": "1", "DUAL_SHAPE_SD_THR": "0.99", "DUAL_SHAPE_OFF_CONTINUE": "0", "DUAL_RAW_OVERSHOOT": "0"}
RT = Path(__file__).resolve().parents[1]                                   # real_testbed
ROOT = RT.parent
SD = ROOT / "experiments" / "rss2027_resubmission" / "sched_dualcontrol"
FROZEN_ENV["SCHED_MEASFIT"] = str(SD / "measfit_span_denoised.json")
for k, v in FROZEN_ENV.items():
    os.environ.setdefault(k, v)

from . import model_newsheet as MN           # noqa: E402  puts newsheet_pipeline and GPT-astra_comparison on sys.path
from . import final_controller as FC         # noqa: E402  (meta_of_target, _diff_ops, FROZEN planner / verifier / data paths)

GA = Path(MN.ROOT) / "experiments" / "GPT-astra_comparison"
for p in (str(GA), str(SD)):                 # SD first: the branch's sched_env / sched_rules / sched_train, not the ICCPS ones
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import dual_core as DC                       # noqa: E402  (before closedloop_v7: closedloop_rlsched must bind the branch's sched_env)
import sched_env as E                        # noqa: E402

FROZEN = {**FC.FROZEN, "policy": str(SD / "results" / "ppo_s0_designc2shield2_spanden_delta0.1" / "policy.pt"), "kind": "rl",
          "delta": float(os.environ["SCHED_DELTA"]), "max_passes": int(E.BUDGET), "name_tag": "rss2027 main line N2 (frozen 2026-11-13)"}
_DENOISER = None


def denoised_field(fld):
    """closedloop_dual.denoised_field: the U-Net scan estimator on one cleaned field (HeightField) -> HeightField on the same grid"""
    global _DENOISER
    import torch
    sys.path.insert(0, str(SD.parent / "scan_denoiser"))
    from train import UNet, denoise
    if _DENOISER is None:
        dev = torch.device("cpu")
        m = UNet(); m.load_state_dict(torch.load(SD.parent / "scan_denoiser" / "model.pt", map_location=dev)); m.eval()
        _DENOISER = (m, dev)
    z = denoise(_DENOISER[0], fld.z_mm[None].astype(np.float32), _DENOISER[1])[0]
    z = np.where(np.isfinite(fld.z_mm), z, np.nan).astype(np.float32)
    return type(fld)(fld.x_mm, fld.y_mm, z, {**dict(fld.meta), "denoised": "scan_denoiser/model.pt"})


class DualController:
    def __init__(self, planner_url, root, core0=120, cores=8, ckpt=None, om=None, pairs=None, data=None, policy=None, kind=None,
                 delta=None, reading_sd_factor=None):
        import closedloop_v7 as V7
        import closedloop_newsheet as CL
        import closedloop_safeprobe as SP
        import closedloop_rlsched as RLS
        import planners as PL
        import amp_projection as AP
        import planner_server as PS
        import sched_train as TR
        self.V7, self.CL, self.SP, self.RLS, self.PL, self.AP = V7, CL, SP, RLS, PL, AP
        assert RLS.E is E, "closedloop_rlsched bound another sched_env (import order)"
        # closedloop_dual.main's configuration of the ICCPS loop modules
        V7.MEASURE, V7.WARM, V7.VERIFIER_K, V7.AMP_EST, V7.PROJ_SCALE = "riglike", True, FC.FROZEN["K"], "projection", 1.0
        V7.RL_CAP_HITS = V7.RL_STOP_SHARED = V7.AREA_GUARD = V7.SP_STOP_SHARED = False
        V7.RL_STOP_MIN = V7.RL_GHAT_FIXED = V7.RL_SIGMA_MM = None
        a = argparse.Namespace(pairs=pairs or FC.FROZEN["pairs"], data=data or FC.FROZEN["data"], tag="rig_dual_controller",
                               root=str(Path(root).resolve()), prompt_schema="auto")
        self.prof, self.root = V7.setup(a)                         # schema v8, caches; ST.compare -> projection (undone below)
        assert V7.SCHEMA == "v8", V7.SCHEMA
        self.ST = V7.ST
        assert os.environ.get("DUAL_AMP") == "span"
        span_compare = AP._ST_COMPARE                              # progress measure and nominal spans = robust span ratio (DUAL_AMP=span)
        V7.ctrl_compare = span_compare; V7.ST.compare = span_compare
        assert SP.nominal_spans is V7.nominal_spans_proj, "nominal spans must go through closedloop_v7.ctrl_compare"
        RLS.K_FC = DC.K_FC
        CL._BLOCK = (int(core0), int(cores))
        self.kind, self.delta = kind or FROZEN["kind"], float(FROZEN["delta"] if delta is None else delta)
        self.reading_sd_factor = float(DC.READING_SD_FACTOR if reading_sd_factor is None else reading_sd_factor)
        self.policy = self.theta = None
        self.policy_path = policy or FROZEN["policy"]
        if self.kind == "rl":
            import torch
            torch.set_num_threads(2)
            self.policy, ck = TR.load_policy(self.policy_path)
            assert ck.get("stop_mode") == "unified" and ck.get("meas") == "proj"
            for key, val in (("kmax", DC.KMAX), ("cap_hits", DC.CAP_HITS), ("cap_mode", E.CAP_MODE), ("ampbel", DC.AMPBEL), ("budget", E.BUDGET),
                             ("probe_first", DC.PROBE_FIRST), ("shield", DC.SHIELD)):
                assert ck.get(key, val) == val, f"policy trained with {key}={ck.get(key)}, service has {val}"
            self.policy_info = {"path": self.policy_path, "iter": ck.get("iter"), "delta": ck.get("delta"), "obs": list(ck["obs_names"])}
        else:
            self.policy_info = None
        self.ckpt, self.om_path = ckpt or FC.FROZEN["ckpt"], om or FC.FROZEN["om"]
        base = PS.RemoteOursPlanner(self.ckpt, planner_url, name=FC.FROZEN["name"])
        self.planner = V7.VerifierPlanner(base, self.om_path, FC.FROZEN["K"], FC.FROZEN["T"])
        self.planner_url, self.lock, self._meta_cache = planner_url, threading.Lock(), {}
        self.info = {"controller": f"final dual: {FROZEN['name_tag']}", "ckpt": self.ckpt, "planner_url": planner_url,
                     "verifier": {"K": FC.FROZEN["K"], "T": FC.FROZEN["T"], "om": self.om_path}, "solver": self.kind, "policy": self.policy_info,
                     "amp_estimator": "robust span of the DENOISED field (DUAL_AMP=span, DUAL_DENOISE=1)", "growth_id": "matched filter -> Gaussian belief on log g",
                     "stop": "dual_core.stop_before_plan (unified, amplitude belief) + smallest-step risk test", "delta": self.delta,
                     "max_passes": FROZEN["max_passes"], "reading_sd_factor": self.reading_sd_factor, "dual_core": DC.CONFIG,
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
    def step(self, run_id, rnd, cur, tgt, history, executed_block=None, prev=None, max_passes=None, denoise=True):
        """one round. cur / tgt: ew_loop.frames.HeightField (the CLEANED field as the rig loop makes it, the target the loop compares
        with); history: executed ops; executed_block: the ops executed since the previous request (default: from the history).
        The service denoises cur itself; `prev` (the client's previous cleaned field) is ignored: the ratio uses the service's own
        stored denoised previous field."""
        V7, SP, AP, ST = self.V7, self.SP, self.AP, self.ST
        t0, T = time.time(), {}
        key = json.dumps(getattr(tgt, "meta", {}) or {}, sort_keys=True, default=str)
        if key not in self._meta_cache:
            self._meta_cache[key] = FC.meta_of_target(getattr(tgt, "meta", {}), tgt, self.PL.DATA)
        meta = self._meta_cache[key]
        if meta is not None and meta.get("source") is None:
            meta = None
        d = self._rdir(run_id); rd = d / f"round_{rnd:02d}"; rd.mkdir(exist_ok=True)
        saved, retry = self._load(run_id, rnd)
        mp = int(FROZEN["max_passes"])
        H_now = int(sum(int(o["n"]) for o in history))
        diag = {"run_id": run_id, "round": rnd, "retry_of_round": retry, "target_meta": meta}
        if max_passes is not None and int(max_passes) != mp:
            diag["warning_max_passes"] = f"client asked max_passes {max_passes}; the service uses the policy's budget {mp} (run rigloop with --max-passes {mp})"
        t1 = time.time()
        cur_d = denoised_field(cur) if denoise else cur             # every scan (the flat one too) goes through the estimator, as in the simulator
        T["denoise_s"] = round(time.time() - t1, 2)                 # (denoise=False: the consistency test feeds the simulator's already denoised fields)
        diag["denoised_by"] = "service" if denoise else "client"
        m_raw = ST.compare(cur, tgt)
        m_now = ST.compare(cur_d, tgt)
        if saved is None:                                          # first request of a run: the flat (or first) measured state
            st = self.CL.SheetGainState(mp)
            st.record_initial(m_now)
            bel = DC.Belief()
            s = {"state": st.to_dict(), "belief": bel.to_dict(), "H_last": H_now, "last_field": None, "g_log": [], "pending": None, "n_blocks": 0}
            st.history = [dict(o) for o in history]
            if H_now:
                diag["warning"] = f"first request with {H_now} executed passes: the growth and amplitude beliefs start here (no block ratio for them)"
        else:
            s = copy.deepcopy(saved)
            st = self.CL.SheetGainState.from_dict(s["state"]) if hasattr(self.CL.SheetGainState, "from_dict") else ST.LoopState.from_dict(s["state"])
            bel = DC.Belief.from_dict(s["belief"])
        self._save(run_id, rnd, s if saved is None else saved, before=True) if not retry else None
        H_prev = int(s["H_last"])
        prev_d = None
        if s.get("last_field") and os.path.exists(s["last_field"]):
            from ew_loop.frames import HeightField
            prev_d = HeightField.load(s["last_field"])
        gid = {"g_hat_before": round(float(bel.g_hat), 4)}
        amp_upd = None
        if saved is not None:
            if H_now > H_prev:                                     # a block was executed since the last request
                blk = executed_block or FC._diff_ops(st.history, history)
                k = H_now - H_prev
                st.history = [dict(o) for o in history]
                st.record_block(m_now, blk[-1] if blk else history[-1], rnd, f"r{rnd:02d}", H_now, k)
                t1 = time.time()
                nd = rd / "nominal_history"
                res = self._nominal([dict(o) for o in history], nd)
                T["nominal_history_s"] = round(time.time() - t1, 1)
                if res.get("rc") == 0 and prev_d is not None:
                    nom = SP.nominal_spans(nd, tgt)                    # span scale (ctrl_compare = span)
                    inc_nom = nom[H_now] - nom[H_prev]
                    gid.update(inc_nominal=round(inc_nom, 3), inc_measured=round(st.scans[-1]["span_cur"] - st.scans[-2]["span_cur"], 3),
                               H_before=H_prev, H_after=H_now)
                    if inc_nom > E.MIN_INC_MM:
                        try:
                            xyz = np.load(nd / "cycles.npz")["xyz"]
                            NP = V7.NP
                            nb, na = NP.field_of(xyz[H_prev].astype(float)), NP.field_of(xyz[H_now].astype(float))
                            rho, ncells = AP.increment_ratio(cur_d, prev_d, na, nb, tgt)
                            gid.update(ratio=round(float(rho), 4), ratio_rule="<dM, dN> / <dN, dN> on the denoised fields", ratio_cells=ncells)
                        except Exception as e:                       # noqa: BLE001 (no ratio from this block)
                            gid["skipped"] = f"matched filter failed: {e!r}"
                    else:
                        gid["skipped"] = f"nominal increment {inc_nom:.2f} <= {E.MIN_INC_MM}"
                else:
                    gid["skipped"] = "no nominal run" if res.get("rc") != 0 else "no previous field"
                pend = s.get("pending")
                as_rec = pend is not None and [(int(e["N"]), int(e["n"])) for e in blk] == [(int(e["N"]), int(e["n"])) for e in pend["recommended"]] \
                    and all(abs(float(e[q]) - float(pend["w"][q])) < 1e-6 for e in blk for q in ("cx", "cy", "wx", "wy", "rot"))
                if as_rec:                                         # executed as recommended: the simulator loop's prediction (forecast increments)
                    DC.predict_amp(bel, [dict(e) for e in blk], pend["inc"], int(pend["N"]), float(pend["mu"]), float(pend["var"]))
                    gid["amp_pred_executed"] = {"source": "forecast increments (as in the simulator)", "D": round(bel.a_pred[2], 4),
                                                "a_prior": round(bel.a_pred[0], 4), "sd_prior": round(float(np.sqrt(bel.a_pred[1])), 4)}
                elif gid.get("inc_nominal") is not None:           # operator edited the block (or ran one against a stop): the nominal increment
                    D = float(gid["inc_nominal"]) / float(st.scans[-2]["span_tgt"])   # of what was EXECUTED, from the nominal history chain
                    pm, pv = E.amp_predict(bel.a_mu, bel.a_var, D, max(k, 1), bel.bel_mu, bel.bel_var, DC.JITTER)
                    bel.a_pred = (float(pm), float(pv), float(D))
                    gid["amp_pred_executed"] = {"source": "nominal increment of the executed block (operator edit; rig only)", "D": round(D, 4),
                                                "a_prior": round(float(pm), 4), "sd_prior": round(float(np.sqrt(pv)), 4)}
                else:
                    gid["amp_belief_skipped"] = "no prediction for the executed block (no nominal run): amplitude belief NOT updated, check before accepting"
                    diag["warning_belief"] = gid["amp_belief_skipped"]
                upd = DC.update_after_scan(st, bel, gid, k, first_block=(H_prev == 0), block=int(s["n_blocks"]) + 1, reading_sd_factor=self.reading_sd_factor)
                amp_upd = upd["amp_belief"]
                s["n_blocks"] = int(s["n_blocks"]) + 1
                s["pending"] = None
            elif H_now == H_prev:                                  # re-scan without passes: the latest state is replaced, one more reading
                st.scans[-1] = {**st.scans[-1], **m_now}
                amp_upd = DC.update_rescan(st, bel, reading_sd_factor=self.reading_sd_factor)
                gid["skipped"] = "re-scan (no passes since the last request)"
            else:
                raise ValueError(f"history has {H_now} passes, fewer than the {H_prev} the service saw for run {run_id}")
        gid.update(g_hat=round(float(bel.g_hat), 4), support=len(bel.bel_obs), belief_mu=round(bel.bel_mu, 4), belief_sd=round(float(np.sqrt(bel.bel_var)), 4))
        s["g_log"].append({"round": rnd, **gid, "amp_belief": amp_upd})
        fp = rd / "field_cur_denoised.npz"; cur_d.save(str(fp)); cur.save(str(rd / "field_cur_cleaned.npz"))
        s["last_field"], s["H_last"] = str(fp), H_now
        tgt.save(str(rd / "target.npz"))
        out = {"diag": diag, "measured": m_now, "measured_raw": m_raw, "g_hat": gid, "belief": bel.summary(), "amp_belief": amp_upd,
               "stop": None, "plan": None, "block": None, "samples": None}
        # ---- the unified stop rules on the belief, before planning (a re-scan round replaces the round it re-measured in the
        #      no-progress window instead of adding one: the window counts blocks, as in the simulator)
        if saved is not None and H_now == H_prev and bel.amp_hist:
            bel.amp_hist = bel.amp_hist[:-1]
        stop = DC.stop_before_plan(st, bel)
        if stop:
            out["stop"] = {"reason": stop, "rule": "dual_core.stop_before_plan (unified rules on the amplitude belief)"}
            return self._finish(run_id, rnd, s, st, bel, out, T, t0)
        # ---- plan: best-of-8 verifier on the paper's prompt, from the denoised field
        V7.PROMPT_G, V7.CURRENT_META = 1.0, meta or {"source": None, "cut": [0, 0]}
        t1 = time.time()
        cur_png, tgt_png, prompt = V7.model_inputs_v7(cur_d, tgt, [dict(o) for o in history], rd, rnd)
        if meta is None:
            self._sg_override(tgt)
        self.planner.last = None
        ans, attempts = self.PL.plan_with_retry(self.planner, cur_png, tgt_png, prompt, tag=f"rig_{run_id}_r{rnd:02d}")
        T["plan_s"] = round(time.time() - t1, 1)
        was = V7.plan_to_was(ans["plan"]) if ans.get("plan") else None
        out["plan"] = {"raw": ans.get("raw"), "plan": ans.get("plan"), "was": was, "attempts": len(attempts)}
        out["samples"] = self.planner.last
        out["samples_was"] = []
        for c in (self.planner.last or {}).get("candidates", []):
            try:
                out["samples_was"].append(V7.plan_to_was(c["plan"]) if c.get("plan") and not c.get("problem") else None)
            except Exception:                                      # noqa: BLE001
                out["samples_was"].append(None)
        if not was:
            out["stop"] = {"reason": "plan_failure", "rule": "no executable plan"}
            return self._finish(run_id, rnd, s, st, bel, out, T, t0)
        w = was[0]
        # ---- nominal forecast of history + min(4, passes left) passes of the first entry
        k_fc = min(DC.K_FC, st.max_passes - st.exec_passes)
        t1 = time.time()
        fd = rd / "forecast"
        res = self._nominal([dict(o) for o in history] + [{**w, "n": k_fc}], fd)
        T["forecast_s"] = round(time.time() - t1, 1)
        if res.get("rc") != 0:
            out["stop"] = {"reason": f"sim_error: forecast {res.get('error', res.get('rc'))}", "rule": "forecast failed"}
            return self._finish(run_id, rnd, s, st, bel, out, T, t0)
        H = st.exec_passes
        nom = SP.nominal_spans(fd, tgt)
        # ---- risk test + solver (dual_core.decide_block: the simulator loop's function)
        hist_before = [dict(o) for o in st.history]
        dec = DC.decide_block(st, bel, self.kind, nom, H, k_fc, w, self.delta, self.policy, self.theta)
        st.history = hist_before                                   # the rig records what is EXECUTED, at the next request
        out["dual"] = dec["dual"]
        if "stop" in dec:
            out["stop"] = {"reason": dec["stop"], "rule": f"smallest-step risk {dec['dual']['p_over_min']} > delta {self.delta} (P(over | belief) of one pass at N_min)"}
            return self._finish(run_id, rnd, s, st, bel, out, T, t0)
        entries, du = dec["entries"], dec["dual"]
        s["pending"] = {"inc": dec["inc"], "N": dec["N"], "w": {q: float(w[q]) for q in ("cx", "cy", "wx", "wy", "rot")}, "mu": bel.bel_mu, "var": bel.bel_var, "a_mu": bel.a_mu, "a_var": bel.a_var,
                        "recommended": [{"N": int(e["N"]), "n": int(e["n"])} for e in entries], "round": rnd}
        bel.a_pred = None                                          # recomputed from the executed block at the next request
        Eg = float(np.exp(bel.bel_mu + 0.5 * bel.bel_var))
        inc = dec["inc"]
        out["block"] = {"entries": entries, "b": int(dec["k"]), "s": dec["decision"]["s"], "N": int(entries[-1]["N"]), "N_plan": int(w["N"]),
                        "n_full": dec["decision"]["k_full"], "passes": int(dec["k"]), "action": du.get("action"),
                        "rule": {"rule": du["rule"], "plan": f"{dec['decision']['k_full']} full pass(es) at N {int(w['N'])}" + (f" + 1 pass at N {int(entries[-1]['N'])} (s {dec['decision']['s']})" if dec['decision']['s'] > 0 else ""),
                                 "conversions": [], "kmax": dec["decision"]["kmax"], "hits_g": du.get("hits_g"), "shield_admissible": du.get("shield_admissible"),
                                 "p_over_min": du["p_over_min"], "g_hat": du["g_hat"], "a_mu": du["a_mu"], "a_sd": du["a_sd"], "action": du.get("action")},
                        "shared_block_size": dec["decision"]["kmax"],
                        "shared_block_reason": (f"block bound {dec['decision']['kmax']} = min(8, ceil(hits {int(w['n'])} / g_hat {du['g_hat']}), passes left "
                                                f"{st.max_passes - st.exec_passes}); shield admissible {du.get('shield_admissible')}; smallest-step risk {du['p_over_min']}"),
                        "forecast": {"k_fc": k_fc, "nominal_spans": [round(x, 3) for x in nom], "increments_amp": [round(x, 4) for x in inc[:k_fc]],
                                     "pred_amp_full_passes": [round(float(bel.a_mu) + Eg * sum(inc[:j]), 4) for j in range(k_fc + 1)], "g_used": round(Eg, 4),
                                     "amp_pred_block": du.get("amp_pred")}}
        return self._finish(run_id, rnd, s, st, bel, out, T, t0)

    def _finish(self, run_id, rnd, s, st, bel, out, T, t0):
        s["state"], s["belief"] = st.to_dict(), bel.to_dict()
        self._save(run_id, rnd, s)
        T["total_s"] = round(time.time() - t0, 1)
        out["timing"] = T
        out["state"] = {"exec_passes": st.exec_passes, "max_passes": st.max_passes, "scans": len(st.scans), "replans": len(bel.amp_hist)}
        json.dump(out, open(self._rdir(run_id) / f"round_{rnd:02d}" / "decision.json", "w"), indent=1, default=float)
        return out

    def _nominal(self, ops, d):
        import nominal_chain as NC
        os.makedirs(d, exist_ok=True)
        rc, wall, segs = NC.run(ops, str(d), *self.CL._BLOCK, growth=1.0, growth_keys=("gtop",), root=self.V7.CHAIN, timeout=self.CL.SIM_TIMEOUT)
        return {"rc": rc, "wall": wall}

    def _sg_override(self, tgt):
        z = tgt.z_mm[np.isfinite(tgt.z_mm)]
        sg = float(np.round(np.ptp(z), 2))
        OM = self.planner.OM
        if not getattr(OM, "_rig_patched", False):
            orig = OM.target_span
            OM.target_span = lambda data, source, cut1: (OM._rig_sg if source is None else orig(data, source, cut1))
            OM._rig_patched = True
        OM._rig_sg = sg


def serve_dual(host="0.0.0.0", port=8767, **kw):
    """HTTP front, the final controller's contract: GET /health; POST /final_step {run_id, round, cur_npz, tgt_npz, prev_npz?, history,
    executed_block?, max_passes?} -> the step() dict. One request at a time."""
    import base64
    import tempfile
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from .frames import HeightField
    ctl = DualController(**kw)
    print(f"[serve-dual] {json.dumps(ctl.info, default=str)}\n[serve-dual] http://{host}:{port}", flush=True)

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
                return self._send(200, {"ok": True, "profile": "newsheet-v8", **ctl.info})
            self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/final_step":
                return self._send(404, {"error": "unknown path"})
            try:
                q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                cur, tgt, prev = field_from_b64(q["cur_npz"]), field_from_b64(q["tgt_npz"]), field_from_b64(q.get("prev_npz"))
                with ctl.lock:
                    res = ctl.step(q["run_id"], int(q["round"]), cur, tgt, q.get("history", []), q.get("executed_block"), prev, q.get("max_passes"))
                self._send(200, res)
            except Exception as e:                                   # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._send(500, {"error": repr(e)})

        def log_message(self, fmt, *args):
            print("[serve-dual]", fmt % args, flush=True)

    ThreadingHTTPServer((host, port), H).serve_forever()
