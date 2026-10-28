#!/usr/bin/env python3
"""Stage-2 eval: execute predicted plans in the C++ simulator and compare the
achieved final shape against the recorded ground-truth target state.

Strain hardening handled EXACTLY: the simulator only starts from flat, so every
rollout replays the sample's history ops from flat (which reproduces the start
state incl. per-face hardening counters), then appends the predicted plan.
Controls: replaying the GT plan itself calibrates the run-noise floor.

Usage:
  python eval_execute.py --pred runs/eval_sym_ck1984.jsonl --out runs/eval_exec_ck1984 \
      --per-cat 12 --controls 2 --workers 16 --cores 8
"""
import argparse, glob, json, os, shutil, subprocess, sys, time
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "category_tests"))
import category_tests as ct


def plan_to_was(plan):
    from common import read_entry
    was = []
    for e in plan:
        wa = read_entry(e)
        a, b = ct.strip_stack_dims(wa["wx"], wa["wy"], wa["rot"])
        if wa["N"] < 3 or a - ct.W <= 0 or b - ct.W <= 0 or wa["n"] < 1:
            return None
        was.append(wa)
    return was


def history_was(rec, start):
    bounds = np.cumsum([0] + [w["n"] for w in rec["was"]])
    out = []
    for k, w in enumerate(rec["was"]):
        done = int(np.clip(start - bounds[k], 0, w["n"]))
        if done > 0:
            out.append({**w, "n": done})
    return out


def run_case(args):
    tag, spec, d, cores_start, cores_n = args
    os.makedirs(d, exist_ok=True)
    json.dump(spec, open(os.path.join(d, "sequence.json"), "w"), indent=1)
    cmd = ["taskset", "-c", f"{cores_start}-{cores_start+cores_n-1}", ct.BIN,
           "-sim", "bilayer_growth", "-case", "custom", "-geometry", "rectangle",
           "-lx", "0.15", "-ly", "0.15", "-res", "0.02", "-h_total", "0.0005",
           "-growth_type", "zigzag_sequence", "-cycle_file", "sequence.json",
           "-enable_passE", "false", "-nsteps", "1", "-export_stl", "false", "-tol", "3e-12"]
    t0 = time.time()
    with open(os.path.join(d, "run.log"), "w") as f:
        p = subprocess.run(cmd, cwd=d, stdout=f, stderr=subprocess.STDOUT)
    return tag, p.returncode, time.time() - t0


def field_metrics(z_pred, z_gt):
    zp, zg = z_pred - z_pred.mean(), z_gt - z_gt.mean()
    corr = float(np.corrcoef(zp, zg)[0, 1])
    return dict(pattern_corr=round(corr, 4),
                span_pred=round(float(np.ptp(z_pred)), 2), span_gt=round(float(np.ptp(z_gt)), 2),
                amp_ratio=round(float(np.ptp(z_pred) / max(np.ptp(z_gt), 1e-9)), 3),
                resid_mean=round(float(np.abs(zp - zg).mean()), 2),
                resid_max=round(float(np.abs(zp - zg).max()), 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cat", type=int, default=12)
    ap.add_argument("--controls", type=int, default=2, help="GT-replay controls per category")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cores", type=int, default=8)
    a = ap.parse_args()
    preds = [json.loads(l) for l in open(a.pred)]
    outdir = os.path.join(HERE, a.out); os.makedirs(outdir, exist_ok=True)

    jobs, index = [], {}
    for cat in ["C1", "C2", "C3", "C4"]:
        rows = [p for p in preds if p["meta"]["category"] == cat and p.get("parsed")]
        # half start=0, half mid-cut where possible
        z0 = [p for p in rows if p["meta"]["cut"][0] == 0]
        mid = [p for p in rows if p["meta"]["cut"][0] > 0]
        pick = z0[: a.per_cat // 2] + mid[: a.per_cat - a.per_cat // 2]
        pick = pick[: a.per_cat]
        for j, p in enumerate(pick):
            rec = json.load(open(os.path.join(HERE, "dataset_pilot", p["meta"]["source"], "record.json")))
            hist = history_was(rec, p["meta"]["cut"][0])
            pw = plan_to_was(p["pred_plan"])
            tag = f"{cat}_{j:02d}_{p['meta']['source'].replace('/','-')}_c{p['meta']['cut'][0]}-{p['meta']['cut'][1]}"
            index[tag] = dict(meta=p["meta"], kind="pred", geom_valid=pw is not None,
                              sym=p.get("metrics"))
            if pw is not None:
                jobs.append((tag, ct.spec_for(hist + pw), os.path.join(outdir, tag)))
            if j < a.controls:                       # GT-replay control of the same sample
                gt = plan_to_was(json.loads(p["label"])["plan"])
                tagc = tag + "_CTRL"
                index[tagc] = dict(meta=p["meta"], kind="control", geom_valid=True)
                jobs.append((tagc, ct.spec_for(hist + gt), os.path.join(outdir, tagc)))
    print(f"{len(jobs)} rollouts ({sum(1 for v in index.values() if v['kind']=='pred')} pred, "
          f"{sum(1 for v in index.values() if v['kind']=='control')} control)", flush=True)
    args_list = [(t, s, d, (i % a.workers) * a.cores, a.cores) for i, (t, s, d) in enumerate(jobs)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for tag, rc, wall in pool.imap_unordered(run_case, args_list):
            index[tag]["rc"] = rc
            print(f"{tag} rc={rc} {wall:.0f}s [{time.time()-t0:.0f}s]", flush=True)

    # compare achieved final vs recorded target (vertex-wise; same mesh topology)
    for tag, v in index.items():
        v.setdefault("rc", None)
        if v.get("rc") == 0:
            finals = sorted(glob.glob(os.path.join(outdir, tag, "*_final.vtp")))
            xyz = np.load(os.path.join(HERE, "dataset_pilot", v["meta"]["source"], "cycles.npz"))["xyz"]
            z_gt = xyz[v["meta"]["cut"][1]][:, 2] * 1e3
            P, _ = ct.read_vtp(finals[-1])
            v["exec"] = field_metrics(P[:, 2] * 1e3, z_gt)
            for f in glob.glob(os.path.join(outdir, tag, "*.vtp")) + glob.glob(os.path.join(outdir, tag, "*.dat")):
                os.remove(f)                          # keep disk small; log+json stay
    json.dump(index, open(os.path.join(outdir, "exec_results.json"), "w"), indent=1)
    ok = [v for v in index.values() if v["kind"] == "pred" and "exec" in v]
    print(f"\npred rollouts with metrics: {len(ok)}")
    if ok:
        print("pattern_corr med", np.median([v["exec"]["pattern_corr"] for v in ok]))
    print("saved", os.path.join(outdir, "exec_results.json"))


if __name__ == "__main__":
    main()
