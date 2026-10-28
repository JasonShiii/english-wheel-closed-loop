"""Drives `rigloop --plant file --captures <staged>` as the operator would with the 2026-10-28 cadence: after each
round's toolpath.json appears, the recorded capture after the block's passes (cumulative) is linked in as
pointcloud_pass<round> (ONE capture per block), then Enter is sent."""
import json, os, subprocess, sys, time
S = os.path.dirname(os.path.abspath(__file__))
REC = "/home/xiangyu/English_wheel/real_testbed/share/Danish_Run_2/2026-10-21_04-29-23"
PKG = os.path.join(S, "robot_pc_package", "real_testbed"); STAGED = os.path.join(S, "captures_staged"); OUT = os.path.join(S, "loop_file_fixed")
PY = os.path.join(os.path.dirname(S), "venv_test", "robotpc_venv", "bin", "python")
cmd = [PY, "-m", "ew_loop.cli", "rigloop", "--plant", "file", "--captures", STAGED, "--fixed-plan", "handover/fixed_recipe_S1_t042_c4.json",
       "--target", "figures/rig/targets/S1_t042_c4.npz", "--out", OUT, "--calib", "pc_post_processing/deliverable_092826/calibration results",
       "--clamp", "12", "--n", "3", "--max-passes", "6", "--yes"]
log = open(os.path.join(S, "drive.log"), "w")
p = subprocess.Popen(cmd, cwd=PKG, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
cum, r, t0 = 0, 1, time.time()
while p.poll() is None and time.time() - t0 < 900:
    tp = os.path.join(OUT, f"round_{r:02d}", "toolpath.json")
    if os.path.exists(tp) and not os.path.exists(os.path.join(STAGED, f"pointcloud_pass{r}")):
        time.sleep(1.0)                                   # let the loop reach its prompt
        n = sum(int(t["passes"]) for t in json.load(open(tp))["toolpaths"])
        cum += n
        src = os.path.join(REC, f"pointcloud_pass{min(cum, 6)}")
        os.symlink(src, os.path.join(STAGED, f"pointcloud_pass{r}"))
        print(f"driver: round {r} asked {n} pass(es); cumulative {cum}; linked recorded pass{min(cum, 6)} as pointcloud_pass{r}", flush=True)
        p.stdin.write("\n"); p.stdin.flush()
        r += 1
    time.sleep(0.5)
if p.poll() is None:
    p.kill(); print("driver: timeout, killed", flush=True)
print("driver: rigloop exit", p.returncode, flush=True)
