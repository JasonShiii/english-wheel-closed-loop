# Rig session 2026-10-28: the automatic loop (controller + model) on the robot PC

Supersedes the by-hand runbook `09_rig_session_2026-10-28.md` for the loop itself; the by-hand script
`plan_from_capture.py` stays as the fallback (section 6). Written 2026-10-28 after the port (section 7); revised the
same evening for the MPC-like cadence (one scan per block) and for running the loop on the robot PC.

## 1. What runs where

Plan A, the loop AND the model on Danish's PC (git clone of `github.com/JasonShiii/english-wheel-closed-loop`, built
by `make_robot_pc_package.py`; its `README.md` = `ROBOT_PC.md` is the install and run guide; the checkpoint zip
`model_free_aug_e8_ckpt1832_inference.zip` by scp / USB; the model needs an NVIDIA GPU on his PC, else the service
on ideas2 as below):

```
Danish's robot PC (python + requirements.txt, no GPU)                    ideas2 (one GPU)
---------------------------------------------------------------          -----------------------------------
python -m ew_loop.cli rigloop --plant file --captures <his run folder>   python -m ew_loop.cli serve-fields
   measurement -> controller (stop? passes?) ---- HTTP: two height fields + history --->  model -> plan
   -> loops/sheetNN/round_NN/toolpath.json                                <---------------------------------
   -> waits for the next capture folder in <his run folder>
robot executes the block (his script with the patch, or his own zigzag from the `robot` block),
sheet to the scan pose, ONE capture per block -> <his run folder>/pointcloud_pass<k>
```

Plan B, no route from his PC to ideas2 (`curl http://129.105.61.93:8765/health` fails and `ssh -L` too): the same
loop runs on ideas2 with `--plant file --exchange <X>`, and the two files per block (`toolpath.json` out, the capture
folder in) are copied by hand (USB / SharePoint / scp). `<X>` is one folder per sheet, e.g.
`share/rig_2026-10-28/sheet01/`: `captures/pointcloud_pass<k>` (k = k-th capture of the session, 0 = the flat scan)
and `round_NN/toolpath.json` + `preview.png` copied there by the loop.

Cadence (decided 2026-10-28): the controller chooses a block of n passes, the robot runs the whole block, the sheet
goes to the scan pose ONCE, one capture, measure, re-plan. The capture folder index therefore counts blocks, not
passes. If the robot side captures after every pass anyway, the loop measures every new folder for the record and
takes the last one as the block's measurement.

## 1b. In the lab, 2026-10-28 evening: what is set up on the robot PC (ssh from ideas2 works)

Robot PC `ME-B111-01`, Windows 11, user `Ping Guo Lab`, campus IP 10.105.33.14, sshd installed, ideas2's key
accepted: `ssh "Ping Guo Lab@10.105.33.14"` from ideas2 (scp TO it works; FROM it stream with `ssh ... tar`).
RTX A2000 6 GB (desktop uses 2.3 GB) -> the model stays on ideas2 (GPU 3, `serve-fields`), reached through a reverse
tunnel opened from ideas2: `ssh -N -R 8765:localhost:8765 "Ping Guo Lab@10.105.33.14"` (the robot PC cannot reach
ideas2:8765 directly; port 22 only). Our repo is cloned at `C:\Users\Ping Guo Lab\english-wheel-closed-loop` and
runs in his conda env `EnglishWheel` (python 3.10, numpy/scipy/matplotlib/cv2 present; nothing installed).
Verified there: controller/spec/frames tests, the geometry against his DoBot `zigzag_planner.py` (54 toolpaths,
max 0.0006 mm), the replay rehearsal on his own 10-21 run through the tunnel (bit-identical to ideas2), and
`run_toolpath_dobot.py check` (his safety check accepts our S1_t042 toolpath on the 231 x 174 sheet with margins
43.4 side / 12 top; his preview drawn; his folder untouched).

The robot is a Dobot CR5A now; his live code is `Documents\GitHub\English-Wheel---DoBot-Files\Scripts`
(`run_zigzag_oldclamps.py` -> `zigzag_forming_oldclamps.py`): scan pose `pointcloud_position`, `pointcloud_pass0`,
then NUM_PASSES passes alternating L->R / R->L with unload + scan + Rx/Rz prompt after every pass; one
`outputs/<timestamp>/` per driver run. His constants were the previous sheet (232 x 124, margin 60.4): the adapter
sets ours from the toolpath file at run time. Plan A on site = `rigloop --plant dobot` (`ROBOT_PC.md` 3b), no file
copying at all; plan A' = `--plant file --captures <his run folder>` if his driver must be run by hand.

Dry run 2026-10-29 13:34 (no sheet in the gripper, over ssh): `rigloop --plant dobot --auto-radii --server
http://localhost:8765 --yes` -> service OK -> his driver (TCP mode had to be set in DobotStudio first) moved to the
scan pose, captured `pointcloud_pass0`, parked at pos_9 (20 s) -> the loop's reference step stopped with "no sheet
found" (coverage 0.12), 26 s in all. Everything but the sheet is verified; the same command with the sheet gripped
is the live run. The tunnel from ideas2 runs under a reconnect loop (`tunnel_keepalive.sh`, scratchpad of the session).

## 2. Before leaving (ideas2)

```
cd /home/xiangyu/English_wheel/real_testbed && source ~/miniconda3/etc/profile.d/conda.sh && conda activate EW
python -m ew_loop.tests.test_rig_port           # measurement == 10-21 chain, prompt == training prompt, render == training renderer
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -n -r | head -1   # pick the freest GPU
```

Targets: `figures/rig/targets/S1_t042_c4.npz` first (21 mm, 4 simulated passes), then `S1_t032_c4.npz` (26 mm).
Checkpoint: `experiments/newsheet_pipeline/runs/free_aug_e8/checkpoint-1832` (the default of `rigloop` and `serve-fields`).

Take along: `robot_pc_package_2026-10-28.zip` (USB and e-mail), this file, the recorded run
`share/Danish_Run_2/2026-10-21_04-29-23` (for a replay rehearsal on his PC if wanted; 6 captures, large).
Before leaving, start the model service on ideas2 and leave it running (plan A needs it; a tmux session survives
the ssh logout):

```
tmux new -s serve
CUDA_VISIBLE_DEVICES=<gpu> HF_HUB_OFFLINE=1 UNSLOTH_DISABLE_STATISTICS=1 python -m ew_loop.cli serve-fields --port 8765
```

## 3. On site, in order

**A. Settle with Danish (5 min, write into `<X>/run_info.txt`).** The wheel-line distance for this sheet
(`SHEET_BOTTOM_TO_WHEEL_LINE_MM`; the loop assumes 12, pass `--clamp`), `MARGIN_SIDE_MM` (12?), the current
calibration folder (copy it to `<X>/calibration results/` and pass `--calib`), force 300 N, the scan pose = the
10-21/22 one, sheet orientation in the gripper (231 mm side across the feed), which gripper is compliant.
How he will execute a block: `run_toolpath_sequence.py run toolpath.json` with the patch, or his own `plan_zigzag()`
fed with the file's `robot` block (`zigzag_lv_mm, zigzag_alpha_deg, zigzag_N, rotation_deg, center_x_mm, start_mm`),
all passes of the block, then the scan pose and ONE capture. Either is fine; the capture folders must be
`pointcloud_pass<k>` with k counting up over the session, in one run folder per sheet.

**A2. Install on his PC (plan A, 10 min + model).** `git clone git@github.com:JasonShiii/english-wheel-closed-loop.git`
(or the zip); in its `real_testbed/`: `pip install -r ../requirements.txt`, then `python -m ew_loop.tests.test_state`,
`test_spec`, `test_frames` (no data needed). Model: if his PC has an NVIDIA GPU (6 GB+), `ROBOT_PC.md` 2a
(`requirements-model.txt`, unzip the checkpoint zip at the repo root, base model from Hugging Face or the copied
cache folder) and a replay rehearsal on the recorded 10-21 run (116 MB, bring it) before touching the robot; else
`curl http://129.105.61.93:8765/health` from his PC, if it fails `ssh -N -L 8765:localhost:8765 xiangyu@129.105.61.93`
and `--server http://localhost:8765`; if that fails too, plan B (the loop on ideas2 with `--exchange`, section 1).
Point `--calib` at the current calibration folder.

**B. Flat scan.** Sheet gripped, flat, scan pose, one capture = `pointcloud_pass0` in his run folder. Start the loop
on his PC (plan A):

```
python -m ew_loop.cli rigloop --plant file --captures <his run folder> --server http://129.105.61.93:8765 \
    --target figures/rig/targets/S1_t042_c4.npz --out loops/sheet01 \
    --clamp 12 --calib "<calibration folder>" --n 3 --max-passes 12
```

Plan B on ideas2: `--plant file --exchange share/rig_2026-10-28/sheet01 --out share/rig_2026-10-28/sheet01/loop`
with `--ckpt` (default) instead of `--server`, the flat capture copied to `<X>/captures/pointcloud_pass0/`.

Check `round_00/reference/reference_preview.png` and the log line `reference:`: `inside_fraction` > 0.7, the gripper
bar on ONE side (`beyond_edge_points`), camera offsets within +-8 mm, flat z range a few mm (+-3 is the 10-21 value;
20 mm means the frame or the gripped edge is wrong). `round_00/field_reference.png` must show the sheet above the
crop line with the hidden band white at the bottom.

**C. First block.** The loop prints `round 1: ... -> k pass(es) of WA ...` and asks `execute round 1?`. Look at
`round_01/preview.png` (our frame) and give Danish `round_01/toolpath.json`. Expected on the flat sheet with this
target: one area near (3, 15), about 100 x 106 mm, rot 0, N around 15, ONE probe pass (the controller's rule for an
area never measured). He runs it, scan pose, one capture (`pointcloud_pass1`); press Enter at the prompt (if he ran
fewer passes than the file says, type that number before Enter).

Check the measured field (`round_01/field_TP0_capture1.png` and the log): a cylinder across x where the area was,
amp rising from the flat value (0.27 on the 10-21 flat scan, which is noise) toward 1. If the dome is a dip, the
height sign is inverted: stop, tell me (one sign flip in `measure.py`). If the dome is mirrored in x: X sign
(`frames.PLANNER_X_SIGN`). Photo of the sheet.

**D. Following blocks.** The loop sizes the next block from the measured gain (expect 2, then 1 passes) and stops
at amp >= 0.95, or when the next pass would overshoot, or after 12 passes, or after 3 rounds without progress. Each
round: `round_NN/toolpath.json` to the robot, all passes, scan pose, one capture, Enter. `--max-passes` and `--n`
are the only knobs. At the stop: `summary.png` (target, final shape, amp and corr per block against passes executed)
is the paper figure; photo; if time, a hand-held scan of the released sheet.

**E. Second sheet.** New run folder and `--out loops/sheet02` (plan B: new `<X>`), new flat scan,
`--target figures/rig/targets/S1_t032_c4.npz`.

Resume after an interruption: the same command with `--resume` (state.json, reference saved under round_00).

## 4. What to record

His run folders (every capture), the loop's `loops/` folder (rounds, `state.json`, `summary.png`), `run_info.txt`,
his force CSVs, photos after each block, the measured wheel-line distance, his answers to the questions, the
released-sheet scan. Then `scp -r` everything to `xiangyu@129.105.61.93:/home/xiangyu/English_wheel/real_testbed/share/rig_2026-10-28/`
(or USB).

## 5. Checks the loop does on its own

- Sheet finder (RANSAC plane that best holds a 231 x 174 rectangle), gripped edge from the gripper bar, per-camera
  plane correction from the flat scan: all from `pointcloud_pass0`, reused for every capture (fixed scan pose).
- Every field: band around the flat plane, hardware mask, despeckle, crop at 20 mm from the gripped edge, best-fit
  plane removed: the same functions the training data went through.
- Images and prompt: `newsheet_pipeline/render.py` and the v5facts prompt, byte-identical to training
  (`tests/test_rig_port.py`).
- Plan -> toolpath: our zigzag geometry in the robot planner frame, safe-zone check; rot 90 refused.
- Controller: `state.py` as in `CLOSED_LOOP.md` step 2, one update per block (the block's gain is split into
  per-pass increments decaying by 0.8); the probe pass runs at the model's N (not controlled).

## 6. Fallbacks

- Loop cannot find the sheet (`inside_fraction` low, preview on another plane): `bridge_checks/danish_run2_measure.GATE`
  and the rectangle score are the knobs; send me the capture.
- Model plan refused (`not_executable: rot=90`): the loop already retried with sampling once; rerun with
  `--ckpt experiments/newsheet_pipeline/runs/free_clean_e8/checkpoint-1832`.
- Model plan unreasonable (huge N on the flat sheet): run the first block from the target's recipe instead
  (`--fixed-plan handover/fixed_recipe_S1_t042_c4.json`, the recipe that made the target),
  then switch back to the model for the re-plans (`--resume` without `--fixed-plan`).
- Loop software problem: one round by hand with `plan_from_capture.py` (`09_rig_session_2026-10-28.md`), same
  files, same controller rules applied by a person.
- Model service down or unreachable mid-run: `--resume` later with `--server`, or plan B on ideas2 with `--ckpt`
  (`--resume` reads the same `--out` folder; copy it over).

## 7. What was verified on 2026-10-28 before the session

- `ew_loop/measure.py` (new): the 10-21 measurement chain on the loop's grid. On the recorded 10-21 run, passes 3 and
  6 agree with `danish_run2_measure.py` within 0.35 mm of span and 0.997 correlation; the reference saves and loads.
- `ew_loop/model_newsheet.py` (new): renderer and prompt identical to the training code; history schema of the
  new prompt.
- `ew_loop/predict.py`: `ModelPlanner`, `HttpFieldsPlanner`, `FixedFieldsPlanner`, `serve_fields` (the robot PC
  could send the height fields and get the plan back; not needed with the file exchange).
- `ew_loop/loop.py`: `FilePlant` (file exchange), `ReplayPlant` (recorded run), the rig measurement path.
- `cli rigloop`, `cli serve-fields`. Old tests still pass.
- Replay rehearsal on the 10-21 recording with the real checkpoint: see the end of this file.
- Evening: the cadence of one capture per block (`state.record_block`, `FilePlant`, `ReplayPlant`, the rig branch
  of `run_loop`), `rigloop --captures`, a prompt-free `FixedFieldsPlanner`, `model_newsheet` importing the prompt
  code only when needed, and the robot-PC package (section 9). Unit tests and `test_rig_port` pass.

## 8. Replay rehearsal (2026-10-28, GPU 7, `free_aug_e8/checkpoint-1832`, target `S1_t042_c4`)

`rigloop --plant replay --replay-run share/Danish_Run_2/2026-10-21_04-29-23 --n 3 --max-passes 6`. The replayed
captures are the recorded 10-21 passes (14-leg zigzag, pivot 5), not what the plan asked for, so the shapes do not
follow the plan; the mechanics do. Outputs in `figures/rig/rehearsal_replay_S1_t042/`, log `logs/rigloop_replay_S1_t042.log`.

| step | result |
|---|---|
| reference (flat scan) | inside fraction 0.853, gripper bar on y+, camera offsets +2.6 / +7.9 / -1.2 mm, flat z -3.3..3.1 mm, 77 % of the observed region covered; grid == target grid (117 x 88) |
| round 1 | flat scan amp 0.27 (noise), corr -0.29; model (30 s with load) proposed ONE area at (3.0, 15.1), 99.5 x 105.6, rot 0, N 15, 3 hits = the target's own recipe; controller: probe pass -> 1 pass; `toolpath.json` 16 pivots, lv 191.2, alpha 4.12, N 15, start 42.3; `cli check` OK; Danish's script in plan mode: inside the safe zone, 15 legs, 14 pivots |
| pass 1 (replayed) | amp 0.69, corr 0.85, z -14.5..29.1 mm, coverage 0.76; measured gain 8.8 mm/pass |
| round 2 | model (10 s): same area, 1 hit; controller: 1 pass (gain 8.8 x 0.8 = 7 vs 6.5 mm of room, closer than stopping) |
| pass 2 (replayed) | amp 1.14, corr 0.92 -> stop: overshoot |

Two things to keep in mind on the rig: the flat scan already reads amp 0.27 because its +-3 mm noise is a quarter of
this target's span, so the controller's `amp < 0.70` block rule starts from there; and the real first pass of a
15-leg zigzag will be smaller than the replayed 14-leg/pivot-5 pass (8.8 mm), so expect 3-5 passes to the stop.

## 9. The robot-PC package, verified 2026-10-28 evening

`make_robot_pc_package.py` -> `robot_pc_package/` = the git clone pushed to
`github.com/JasonShiii/english-wheel-closed-loop` (private; invite Danish: `gh repo add-collaborator` or the web UI,
his login is `danishgalebotswe`), plus `robot_pc_package_2026-10-28.zip` (0.6 MB; code, targets, calibration, fixed
recipe, robot-side reference, documents, the prompt code for a local model; `requirements.txt` = numpy, scipy,
matplotlib, opencv-python-headless; `requirements-model.txt` = the training environment's torch/unsloth versions)
and, with `--model`, `model_free_aug_e8_ckpt1832_inference.zip` (132 MB, scp / USB; not in git). After every code
change: rebuild, then `git -C robot_pc_package add -A && git commit && git push`. Both checks ran from a copy OUTSIDE the repository, with a fresh `python -m venv` holding
only `requirements.txt` (python 3.10), i.e. what Danish's PC will have:

| check | result |
|---|---|
| `rigloop --plant replay --server http://127.0.0.1:8765` (model service on ideas2 GPU 3, 23 s load), target S1_t042 | plan, prompt, model answer, rendered image and `toolpath.json` of round 1 byte-identical to the morning rehearsal; cadence one capture per block: capture 1 amp 0.69 corr 0.85, capture 2 amp 1.14 -> stop overshoot (replayed captures are the stronger recorded zigzag). Outputs `figures/rig/rehearsal_robotpc_http_S1_t042/`, log `logs/rigloop_robotpc_http_S1_t042.log` |
| `rigloop --plant replay` with the MODEL LOCAL (no `--server`; checkpoint from `model_free_aug_e8_ckpt1832_inference.zip` unpacked at the package root, training environment, GPU 3) | import chain of the prompt code inside the package OK (planners, eval_execute, category_tests, facts.json); model loaded in 23 s; plan, prompt and toolpath of round 1 identical to the in-repo rehearsal. Outputs `figures/rig/rehearsal_robotpc_localmodel_S1_t042/`, log `logs/rigloop_robotpc_localmodel_S1_t042.log` |
| `rigloop --plant file --captures <staged run folder> --fixed-plan handover/fixed_recipe_S1_t042_c4.json`, driven by `ew_loop/tests/drive_file_plant_rehearsal.py` (links one recorded capture per block after each `toolpath.json`, then Enter) | the loop waited at the prompt, found `pointcloud_pass1`, measured, re-planned, waited for `pointcloud_pass2`, stopped on overshoot; no model code needed. Outputs `figures/rig/rehearsal_robotpc_fileplant_fixed_S1_t042/`, log `logs/rigloop_robotpc_fileplant_fixed_S1_t042.log` |

Not verified until on the machine: that Danish's PC reaches port 8765 on ideas2 (else the ssh tunnel, else plan B),
Python 3.10+ and pip on his PC, the current calibration folder, the clamp 12 mm, the height and X signs, the scan
pose, and whether his script takes `toolpath.json` (the patch) or he feeds the `robot` block to his own planner.
