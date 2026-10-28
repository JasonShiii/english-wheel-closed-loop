# The loop on the robot PC

This folder is the closed-loop software of the English-wheel forming experiment, packaged to run on the robot PC
(written 2026-10-28; built by `real_testbed/make_robot_pc_package.py` from the research repository).

```
captures (3 cameras)  -->  measurement  -->  controller (stop? how many passes?)  -->  planner (model)  -->  toolpath.json
      ^                                                                                                          |
      |                         robot runs the block, sheet to the scan pose, ONE capture                        |
      +----------------------------------------------------------------------------------------------------------+
```

Get it: `git clone git@github.com:JasonShiii/english-wheel-closed-loop.git` (or `https://github.com/JasonShiii/english-wheel-closed-loop`),
or unzip `robot_pc_package_<date>.zip`. Updates: `git pull`. The checkpoint comes separately (section 2a).

Inputs: the capture folders the robot software writes (`pointcloud_pass<k>/device_1..3_pointcloud_001.ply`), the
rig calibration folder, a target height field. Output per block: `round_NN/toolpath.json` (the zigzag as corner
points in the robot planner frame, plus the `robot` block `zigzag_lv_mm, zigzag_alpha_deg, zigzag_N, rotation_deg,
center_x_mm, start_mm` for `plan_zigzag()`), `preview.png`, the measured field and the model's answer. The robot
software converts the file into its own calls (reference: `real_testbed/robot_pr/run_toolpath_sequence.py` with
`zigzag_forming_toolpath_api.patch`); the loop never drives the robot itself.

## 1. Install (once, 5 min)

Python 3.10 or newer. Then:

```
pip install -r requirements.txt          # numpy, scipy, matplotlib, opencv-python-headless
cd real_testbed
python -m ew_loop.tests.test_state && python -m ew_loop.tests.test_spec && python -m ew_loop.tests.test_frames
```

That is all the loop needs without the model. The model (a LoRA fine-tune of Qwen3-VL-4B, 4-bit) runs either on the
robot PC's own NVIDIA GPU (2a, preferred: no network dependence) or as a service on ideas2 (2b).

## 2a. The model on the robot PC (NVIDIA GPU with 6 GB or more)

Linux, or Windows through WSL2 (Ubuntu): unsloth and bitsandbytes are Linux-first; native Windows is possible but we
have not tried it. Then:

```
pip install -r requirements-model.txt        # torch 2.10 (pick the wheel for the machine's CUDA from pytorch.org), transformers, unsloth, bitsandbytes, peft
unzip model_free_aug_e8_ckpt1832_inference.zip   # at the repository root -> experiments/newsheet_pipeline/runs/free_aug_e8/checkpoint-1832/
```

The checkpoint zip (132 MB, by scp / USB, not in git) holds the adapter and tokenizer. The 4.1 GB base model
`unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` is downloaded from Hugging Face on the first run (internet needed
once), or copied from ideas2: the folder `~/.cache/huggingface/hub/models--unsloth--Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`
into the same place on the robot PC, then run with `HF_HUB_OFFLINE=1`. With the checkpoint in place `rigloop`
needs no `--server` and no `--ckpt` (the default is that path). Loading takes about 25 s, a plan 7 to 30 s.
Check before the session: `python -m ew_loop.cli rigloop --plant replay --replay-run <a recorded run> ...` (section 3).
Verified 2026-10-28 from a clone outside the research repository with the checkpoint zip unpacked at its root: the
model loads in 23 s and gives the same plan, prompt and `toolpath.json` as the research repository on the recorded
10-21 run.

## 2b. The model as a service (GPU machine)

On ideas2, in the training environment:

```
cd /home/xiangyu/English_wheel/real_testbed && conda activate EW
CUDA_VISIBLE_DEVICES=<free gpu> HF_HUB_OFFLINE=1 UNSLOTH_DISABLE_STATISTICS=1 python -m ew_loop.cli serve-fields --port 8765
```

From the robot PC: `curl http://129.105.61.93:8765/health` must answer `{"ok": true, ...}`. If the port is not
reachable but ssh is: `ssh -N -L 8765:localhost:8765 xiangyu@129.105.61.93` in a second terminal and use
`--server http://localhost:8765`. If there is no network at all, run the loop on ideas2 with the file exchange
instead (`10_automatic_loop_session_2026-10-28.md`, section 1): same software, the two files per block are copied
by hand.

A retrained model is deployed by restarting the service with `--ckpt <checkpoint>`; the robot PC copy does not change.
A model with another prompt or renderer gets a new profile (`ew_loop/model_newsheet.py` is the current one; the
profile has four methods: `render`, `build_prompt`, `history`, `executable`).

## 3. Run one sheet

```
cd real_testbed
python -m ew_loop.cli rigloop --plant file --captures <the robot software's run folder for this sheet> \
    --target figures/rig/targets/S1_t042_c4.npz --out loops/sheet01 \
    --calib "pc_post_processing/deliverable_092826/calibration results" --clamp 12 --n 3 --max-passes 12
```

(model on the local GPU, 2a; with the service instead add `--server http://129.105.61.93:8765`.)

- `--captures`: the folder where the capture folders appear, named `pointcloud_pass<k>`, `k` counting the captures
  of the session (`pointcloud_pass0` = the flat gripped sheet at the scan pose). The loop reads them in place.
- `--calib`: the current calibration folder (`extrinsics.json`, `intrinsics.json`, optional `icp/`).
- `--clamp`: the robot's `SHEET_BOTTOM_TO_WHEEL_LINE_MM` for this sheet. `--n`: most passes per block.

What happens:

1. Flat scan: grip the flat sheet, scan pose, capture `pointcloud_pass0`, press Enter. The loop finds the sheet,
   writes `loops/sheet01/round_00/reference/reference_preview.png` (check: the sheet, the gripped edge at the bottom).
2. Every block: the loop prints `round N: amp ..., corr ... -> k pass(es) of WA ...`, writes
   `loops/sheet01/round_0N/toolpath.json` and `preview.png`, asks `execute round N? [y/N]`. Answer `y`, run the file on
   the robot (all k passes), move the sheet to the scan pose, capture once (`pointcloud_pass<k>`), press Enter at
   `press Enter when the capture folder is there`. If fewer passes were run than planned, type that number before Enter.
3. The loop measures the capture, decides, re-plans. It stops on its own (amplitude reached, next pass would
   overshoot, budget, no progress) and writes `loops/sheet01/summary.png` and `state.json`.

Interrupted? The same command with `--resume`. Second sheet: new run folder, new `--out`.

Rehearsal without the robot: `--plant replay --replay-run <a recorded run folder with pointcloud_pass0..k>` replays
the recorded captures as if the robot had produced them.

## 4. The pieces, and what is replaceable

| piece | file | interface |
|---|---|---|
| measurement | `ew_loop/measure.py` (chain: `bridge_checks/danish_run2_measure.py`) | capture folder -> `HeightField` (2 mm grid, mm, NaN = no data) |
| controller | `ew_loop/state.py` | metrics of the measured field -> stop / passes of the next block |
| planner | `ew_loop/predict.py`: `ModelPlanner` (local GPU), `HttpFieldsPlanner` (service), `FixedFieldsPlanner` (a given plan) | `plan_fields(cur, tgt, history, outdir)` -> plan |
| model profile | `ew_loop/model_newsheet.py` (renderer `experiments/newsheet_pipeline/render.py`) | render, prompt, history schema |
| plan -> robot | `ew_loop/spec.py` | plan entry -> `toolpath.json` (schema 1) |
| loop | `ew_loop/loop.py` `run_loop`, plants `FilePlant`, `ReplayPlant` | |

Targets: `figures/rig/targets/*.npz` (clamped shapes from the simulator, 117 x 88 nodes on the 231 x 174 mm sheet).
`CLOSED_LOOP.md` explains the loop step by step; `ew_loop/README.md` the code; the session runbook is
`10_automatic_loop_session_2026-10-28.md`.
