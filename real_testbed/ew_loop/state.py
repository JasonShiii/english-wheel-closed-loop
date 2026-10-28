"""Loop bookkeeping: shape metrics, executed history, measured per-pass increments, stop
rules, and how many passes to run before the next re-plan.

Rules and thresholds are those adopted in the offline MPC evaluation
(experiments/mpc_end2end/mpc_rollout.py, "E + amp guard", 2026-09-29), restated here for a
loop that measures ONCE PER BLOCK (MPC-like cadence decided 2026-10-28: the robot runs the whole
block, the sheet goes to the scan pose once, measure, re-plan; a block of one pass = a measurement
per pass, as in the simulation evaluation):
  amplitude   amp = span(current) / span(target) on the cells both fields have
  stop        amp >= 1.00 (cannot be undone: over-forming); amp >= 0.95 (close enough from
              below); pass budget used up; no progress (shape correlation and amp both
              stagnant over the last 3 re-plans); the model gives no usable plan
  passes      far from the target more passes per block (amp < 0.70: n, < 0.90: 2, else 1);
              a block never spans two working areas; a working area never measured before
              gets one probe pass; otherwise the last measured mm/pass on that area, decayed
              by the measured ratio of its last two passes (prior 0.8), decides how many passes
              still land below the target (or land closer to it than stopping short); a block's
              measured gain is split into per-pass increments decaying by 0.8 inside the block
On real scans the span is robust (0.5-99.5 percentile) instead of max-min.
"""
import json

import numpy as np

AMP_OK, AMP_HI = 0.95, 1.00
RHO_STOP = 0.99
K_WATCH, EPS_CORR, EPS_AMP = 3, 0.002, 0.02
DECAY_PRIOR = 0.8


def span(z, robust=True):
    return float(np.percentile(z, 99.5) - np.percentile(z, 0.5)) if robust else float(np.ptp(z))


def compare(cur, tgt, robust=True):
    """shape metrics of a current field against the target on the cells both have data."""
    if cur.z_mm.shape != tgt.z_mm.shape:
        raise ValueError(f"current {cur.z_mm.shape} and target {tgt.z_mm.shape} grids differ")
    m = cur.valid & tgt.valid
    if m.sum() < 50:
        raise ValueError(f"only {int(m.sum())} cells in common between current and target")
    zc, zt = cur.z_mm[m].astype(float), tgt.z_mm[m].astype(float)
    a, b = zc - zc.mean(), zt - zt.mean()
    sc, st = span(zc, robust), span(zt, robust)
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0
    return {"pattern_corr": round(corr, 4), "span_cur": round(sc, 2), "span_tgt": round(st, 2),
            "amp_ratio": round(sc / max(st, 1e-9), 3), "resid_mean": round(float(np.abs(a - b).mean()), 2),
            "n_cells": int(m.sum())}


def wa_key(wa):
    return json.dumps([round(float(wa[k]), 1) for k in ("cx", "cy", "wx", "wy")] + [int(wa["rot"]), int(wa["N"])])


class LoopState:
    def __init__(self, max_passes, robust=True):
        self.max_passes = int(max_passes)
        self.robust = robust
        self.history = []            # executed ops: wa dicts with n = passes, merged when repeated
        self.scans = []              # one entry per measured state: {round, toolpath, pass, exec_passes, metrics}
        self.replans = []            # metrics at every re-plan (no-progress rule)
        self.increments = {}         # wa_key -> per-pass span increments (mm), in order
        self.exec_passes = 0
        self.stopped, self.reason = False, None

    # ---- recording -----------------------------------------------------------------------
    def record_initial(self, metrics):
        self.scans.append({"round": 0, "toolpath": None, "pass": 0, "exec_passes": 0, **metrics})

    def record_pass(self, metrics, wa, round_idx, toolpath_id, pass_idx):
        """a measurement after every pass (simulator plants)"""
        self.record_block(metrics, wa, round_idx, toolpath_id, pass_idx, passes=1)

    def record_block(self, metrics, wa, round_idx, toolpath_id, capture_idx, passes):
        """one measurement after a block of `passes` passes of `wa` (rig cadence): the block's span gain
        is split into per-pass increments g, g r, g r^2 ... (r = DECAY_PRIOR) that sum to the block gain,
        so block_size's per-pass model is the same whatever the cadence; passes = 1 is record_pass."""
        prev = self.scans[-1]["span_cur"] if self.scans else 0.0
        p, r = max(int(passes), 1), DECAY_PRIOR
        g1 = (metrics["span_cur"] - prev) * (1 - r) / (1 - r ** p)
        self.increments.setdefault(wa_key(wa), []).extend(round(g1 * r ** j, 3) for j in range(p))
        self.exec_passes += p
        self.scans.append({"round": round_idx, "toolpath": toolpath_id, "pass": capture_idx, "passes": p,
                           "exec_passes": self.exec_passes, **metrics})

    def record_executed(self, wa, n):
        if n <= 0:
            return
        if self.history and wa_key(self.history[-1]) == wa_key(wa):
            self.history[-1]["n"] += n
        else:
            self.history.append({**wa, "n": int(n)})

    # ---- decisions ------------------------------------------------------------------------
    def check_stop(self):
        """called before every re-plan, on the latest measured state."""
        m = self.scans[-1]
        self.replans.append(m)
        if m["amp_ratio"] >= AMP_HI:
            self.stopped, self.reason = True, "converged" if m["pattern_corr"] >= RHO_STOP else "overshoot"
        elif m["amp_ratio"] >= AMP_OK:
            self.stopped, self.reason = True, "converged" if m["pattern_corr"] >= RHO_STOP else "amp_reached_shape_off"
        elif self.exec_passes >= self.max_passes:
            self.stopped, self.reason = True, "budget"
        elif len(self.replans) > K_WATCH:
            old = self.replans[-1 - K_WATCH]
            if m["pattern_corr"] - old["pattern_corr"] < EPS_CORR and m["amp_ratio"] - old["amp_ratio"] < EPS_AMP:
                self.stopped, self.reason = True, "no_progress"
        return self.stopped

    def block_size(self, first_entry, n):
        """passes of the plan's first entry to run before the next re-plan (0 = stop)."""
        m = self.scans[-1]
        if m["amp_ratio"] >= AMP_HI:
            return 0
        k = n if m["amp_ratio"] < 0.70 else (2 if m["amp_ratio"] < 0.90 else 1)
        k = min(k, n, int(first_entry["n"]), self.max_passes - self.exec_passes)
        inc = [d for d in self.increments.get(wa_key(first_entry), [])]
        if not inc or inc[-1] <= 0.5:                 # never measured here (or no growth): probe one pass
            return min(k, 1)
        r = DECAY_PRIOR
        if len(inc) >= 2 and inc[-2] > 0.5:
            r = float(np.clip(inc[-1] / inc[-2], 0.3, 1.0))
        room = (AMP_HI - m["amp_ratio"]) * m["span_tgt"]
        j, pred = 0, 0.0
        while j < k:
            step = inc[-1] * r ** (j + 1)
            if pred + step <= room:
                pred += step; j += 1
            elif (pred + step - room) < (room - pred):
                j += 1; break
            else:
                break
        return j

    # ---- persistence ----------------------------------------------------------------------
    def to_dict(self):
        return {k: getattr(self, k) for k in ("max_passes", "robust", "history", "scans", "replans",
                                              "increments", "exec_passes", "stopped", "reason")}

    @classmethod
    def from_dict(cls, d):
        s = cls(d["max_passes"], d.get("robust", True))
        for k in ("history", "scans", "replans", "increments", "exec_passes", "stopped", "reason"):
            setattr(s, k, d[k])
        return s
