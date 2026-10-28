# Closed-loop forming on the robotic English wheel

This folder connects three parts that already exist on their own: the planning model trained on simulated forming (`experiments/mpc_end2end`), the UR5e English-wheel cell (repository `Northwestern-AMPL/Engish_Wheel_T109`, branch `oct_2_demo`) and that cell's camera rig. The loop forms a sheet toward a target shape by repeating four things: measure the sheet, let the model plan, run a few passes, measure again.

The loop software (`ew_loop/`) runs on the robot PC. The model runs there as well if that PC has an NVIDIA GPU; otherwise it runs on a GPU machine that the loop reaches over HTTP.

```
                          target shape (height field)
                                      |
                                      v
   +------>  [1 Measurement]  ---->  [2 Executor evaluation]  --done-->  stop
   |         cameras -> height field  stop rules, passes per block
   |                                          | proceed
   |                                          v
   +----  [5 Robot actuation]  <--  [4 toolpath.json]  <--  [3 VLM: plan / re-plan]
          run_toolpath_sequence.py                           model service
```

The numbers are the steps described below. Step 1 and step 5 run on the robot cell; steps 2 to 4 are the loop software.

## Conventions used by every step

| Item | Definition |
|---|---|
| Sheet frame | Origin at the sheet centre. x runs across the sheet. y runs along the feed direction, from the gripped edge (y = −L/2) to the free edge (y = +L/2). z is height. All lengths are in mm. The model's images use the same axes, with x to the right and y up. |
| Wheel line | Where the wheel touches the sheet when the robot is at its forming start pose (`TARGET_POS_9`). It lies at y = −L/2 + c, where c is the robot constant `SHEET_BOTTOM_TO_WHEEL_LINE_MM` (100 mm on `oct_2_demo`), passed to the loop as `--clamp`. The band between the gripped edge and the wheel line is never wheeled. |
| Robot frame | The frame of the robot's own toolpath planner (`zigzag_planner.plan_zigzag`). Its origin is the middle of the wheel line, X runs across the sheet and Y along the feed, so X = x and Y = y − (wheel-line y). The sign of X still has to be confirmed on the machine (item 3 at the end). |
| Pass | One run of the wheel through one zigzag. It equals one simulator cycle and one "hit" in the model's plan. |
| Height field | An `.npz` file holding `x_mm` and `y_mm` (the nodes of a 2 mm grid over the sheet, edges included), `z_mm` (NaN where there is no data) and `meta` (JSON). A field made from a simulation also keeps the mesh vertices (`points_mm`). The target, the current shape and every measurement use this format. |
| Band behind the wheel line | Set to no-data in every field the model sees, the target included. Its shape is a passive side effect of the clamping, and neither the cameras (the gripper and wheel block the view) nor the simulator (idealised clamp) capture it reliably. |

All frame constants live in `ew_loop/frames.py` and nowhere else.

## Before forming: the flat reference capture

`run_toolpath_sequence.py baseline` moves the robot, with the sheet gripped and the wheel open, to its forming start pose and saves one capture. This flat scan does four jobs.

1. **It fixes the reference plane.** A plane is fitted to the flat sheet, and every later height is the distance from this plane. The plane is never refitted, because a refit on a formed sheet would absorb part of the shape.
2. **It locates the sheet in the camera view.** The known rectangle (width × length) is fitted to the visible outline: the free edge and the two side edges. The gripped edge is hidden by the gripper. Pixels at depth edges blend sheet and background, which makes the sheet look 5–15 mm wider. All edges are therefore pulled in by half of the measured excess width. A rough free-edge direction in the camera frame (`--feed-dir`, set once per rig setup) tells the fit which side of the rectangle is which. `reference_preview.png` shows the result for checking.
3. **It records what sits above the sheet.** Points well above the flat sheet around the wheel contact belong to the upper wheel, which does not move relative to the cameras; those spots of the camera view are dropped from every later capture. Points above the sheet along the gripped edge belong to the gripper, which moves with the sheet; those spots of the sheet are dropped from every later field.
4. **It provides the baseline field.** The flat sheet's own height field (gravity sag, waviness, plane-fit residual) is subtracted from every later field. Where the flat scan saw no sheet, a smooth quadratic fit of it stands in.

No calibration between the cameras and the robot is needed. The cameras only have to stay where they were during the reference capture.

## Step 1. Measure: point cloud to height field

Code: `ew_loop/scan_to_field.py` (first setup) and, since 2026-10-28, `ew_loop/measure.py` for the rig as it stands (231 x 174 x 0.6 mm sheet, captures at a withdrawn scan pose). The rig version replaces the reference step below by the sheet finder of `bridge_checks/danish_run2_measure.py` (the RANSAC plane whose inliers best fill the known rectangle, gripped edge from the gripper bar, per-camera plane correction) and needs no robot pose, because every capture is taken at the same scan pose; crop and plane removal are the training code's own. `cli rigloop` runs this loop; `10_automatic_loop_session_2026-10-28.md` is its runbook.

The robot saves a cloud from every camera after every pass. The loop uses the three rig cameras merged into one cloud with the rig calibration (`extrinsics.json`, refined by `icp/refined_transforms.json` when present), the same transform the metrology team's merge viewer applies. Merging adds the parts of the sheet that only cameras 1 and 2 see: 11–18 % more measured cells on the August captures. The merge is only as good as the calibration: on those captures the cameras disagree on the sheet by 2–6 mm, and the metrology team's own ICP report of 2026-09-28 gives about 6 mm median cross-camera distance, while one camera alone measures a flat surface with 0.8 mm point noise. The reference step therefore measures the agreement of the cameras on the flat sheet and warns above 2 mm; a single camera (`--camera 3`) is the fallback. One capture becomes a height field in six steps.

1. **Depth window.** Points 200–700 mm from each camera are kept, before the merge. The sheet is 250–600 mm away, and the floor and walls are farther.
2. **The sheet's position comes from the robot motion.** The gripper holds the sheet rigidly, and the robot moves it only within the forming plane. Every zigzag starts and ends with a leg straight along the feed, so at every capture the sheet has the orientation it had at the reference capture. Its position then follows from which sheet point is under the wheel. At the reference capture that point is the planner origin. The robot runs the passes of a zigzag back and forth and captures where each pass ends: after an odd pass the zigzag's last corner is under the wheel, after an even pass its first corner. This is exact up to robot accuracy. Unlike fitting the outline, it is not biased by how far the sheet has curled.
3. **Pose check.** The capture's own outline is still fitted and compared with the predicted position. A difference above 2° or 8 mm across the sheet stops the loop. That catches the gross errors, which are off by tens of mm: the capture was taken somewhere else, an axis sign is wrong, or the sheet slipped badly in the gripper. A tighter limit would raise false alarms, because a strongly curled sheet moves its own outline by a few mm.
4. **Gridding.** Points inside the sheet rectangle and within 100 mm of the reference plane are assigned to their nearest grid node. Each node gets the median height of its points, with at least 3 points required (a node collects about 10 points at 450 mm).
5. **Cleanup.** Nodes covered by the upper wheel or the gripper are dropped. So are nodes that differ from the median of their 7 × 7 neighbourhood by more than 3 mm. One node is eroded off every boundary.
6. **Baseline and mask.** The baseline field is subtracted, and the band behind the wheel line is set to no-data.

Each field carries `meta.pose_check`, `meta.coverage_usable` (the fraction of the area beyond the wheel line that has data) and the number of points used. Note that the cameras record the whole scene: the sheet is not cut out beforehand, the steps above find it.

## Step 2. How many passes now, and when to stop

Code: `ew_loop/state.py`.

After every measurement the loop first decides whether to stop, and otherwise how many passes to run before the next measurement. It asks the model (step 3) only when a new block is needed, and only the plan's first working area is executed. The rules are the ones that worked best in the offline evaluation on the simulator (`experiments/mpc_end2end/MPC_EVAL_REPORT.md`).

- **Progress** is measured by two numbers on the cells that both fields have. The amplitude ratio is the height span of the current field divided by that of the target, using the 0.5 and 99.5 percentiles so a few bad cells cannot move it. The shape correlation compares the two fields after removing their means.
- **Stop rules.** The loop stops when any of these holds:
  - the amplitude ratio reaches 1.0, because over-forming cannot be undone;
  - the amplitude ratio reaches 0.95, which is close enough from below;
  - the pass budget is used up;
  - neither the correlation nor the amplitude improved over the last 3 re-plans;
  - the model gives no usable plan.
- **Block size** (applied to the model's plan from step 3). Only the first working area of the plan runs before the next re-plan; a block never mixes areas. (The robot side can execute any number of zigzags from one file; the loop sends one at a time because it re-plans after every few passes. `cli plan --n 0` writes a whole plan into one file for open-loop execution.)
  - The block is `--n` passes (default 3) while the amplitude ratio is below 0.70, 2 passes below 0.90, then 1.
  - A working area that was never measured gets a single probe pass.
  - After that, the measured height gain per pass on that area limits the block. The gain shrinks from pass to pass by the measured ratio of its last two passes (work hardening; 0.8 until measured). The block keeps only the passes that stay below the target, or that land closer to it than stopping short would.
  - Cadence on the rig (2026-10-28): the robot runs the whole block, the sheet goes to the scan pose once, one capture, then the controller updates (the block's measured gain is split into per-pass gains with the 0.8 decay). A scan after every pass is not needed; if it happens, the extra captures are recorded and the last one counts.

## Step 3. Plan: the model

Code: `ew_loop/predict.py`.

The model gets two images and some text:

- the current and target height fields, drawn in the frame the model was trained on (300 × 300 mm for checkpoint 1984, with the sheet in the middle). Colours are normalised per image, and no-data areas are white. The drawing code is the training code itself (`model_io.heatmap_png`). A simulated field is drawn from its mesh vertices, exactly like a training image, and a measured field from its grid. This matters. On one test case, two drawings of the same target that differ in 22 pixels, by at most 2 of 255 colour levels, gave first zigzags of opposite orientation. Across 28 small variations of four targets the orientation held, but the strip count, the working area and the number of passes changed (`figures/q1_*.png`).
- the height range of each field as numbers, because the images carry only the shape.
- the history: every executed zigzag with its working area, orientation, strip count and passes done.

The model answers with the remaining plan in JSON. The plan is an ordered list of working areas, each given by its centre and size, a zigzag orientation (0: strips along the feed; 90: strips across it), a strip count N and a number of passes. The prompt text and the history format are copied from the training code into `ew_loop/model_io.py`, and `tests/test_model_io.py` keeps them identical. One answer takes 7–20 s on an RTX A5000, and loading the model takes 10–15 s.

There are three ways to run the model. `LocalPlanner` uses the robot PC's own NVIDIA GPU. `HttpPlanner` calls a GPU machine running `python -m ew_loop.cli serve`. `FixedPlanner` replays a fixed plan file, which tests the robot and the cameras without a model.

## Step 4. From the plan to robot motion

Code: `ew_loop/spec.py`.

A working area becomes a zigzag by the same rule that generated the training data:

- strip length lv = (side along the strips) − 10 mm;
- tan α = ((side across the strips) − 10 mm) / ((N − 2) · lv).

The corner points are computed in the sheet frame by the training-data code, moved to the robot frame, and reversed, because the robot runs the same pattern from the other end. On 54 test cases these points match the robot's own planner to within 1e-3 mm. When comparing, note that the robot's lv is one full up-and-down V, which is twice ours.

An entry is refused, with its reason, in three cases:

- its orientation is 90, because the sheet cannot yet be turned under the wheel;
- a corner point leaves the robot's safe zone (`--margin-side`, 12 mm by default as on the robot, from the side and free edges; nothing behind the wheel line);
- N is below 3.

Odd N is accepted. The model may plan things the robot cannot do yet; the refusal is where the training distribution meets the machine.

`toolpath.json` is the only thing the robot side reads. Executing it is the robot team's part; `robot_pr/` is our reference implementation of it (section 5), to be used, adapted or replaced.

```json
{"schema_version": 1, "units": {"length": "mm", "angle": "deg"},
 "sheet": {"width_mm": 195.0, "length_mm": 190.0, "thickness_mm": 1.0, "clamp_mm": 100.0,
           "margin_side_mm": 12.0, "margin_bottom_mm": 0.0, "...": "..."},
 "toolpaths": [{"id": "TP0", "passes": 3,
                "pivots_xy_mm": [[50.0, 0.5], [50.0, 65.5], [30.0, 0.5], "..."],
                "robot": {"zigzag_lv_mm": 130.0, "zigzag_alpha_deg": 17.1027, "zigzag_N": 8,
                          "rotation_deg": 0, "center_x_mm": -10.0, "start_mm": 0.5},
                "ours":  {"cx": -10.0, "cy": 38.0, "wx": 130.0, "wy": 75.0, "rot": 0, "N": 8,
                          "lv_mm": 65.0, "alpha_deg": 17.1027, "strip_width_mm": 10.0}}]}
```

`pivots_xy_mm` is executed as given, in the robot frame. `robot` and `ours` are kept only as a record. Because the robot never re-derives the corner points, the whole system has a single definition of a zigzag.

## Step 5. Execution on the robot

Code: `robot_pr/run_toolpath_sequence.py`.

This script goes into the robot's `code/` folder together with a short patch, `robot_pr/zigzag_forming_toolpath_api.patch`. The patch lets `run_zigzag_forming()` take the corner points and the pass count as arguments; its default behaviour is unchanged.

`run` first sets the sheet constants from the file, then checks and previews every zigzag before any motion. For each zigzag it approaches, loads the wheel, runs the passes back and forth without lifting the wheel, captures after every pass, and unloads the wheel. It asks before each zigzag unless `--yes` is given. The output is `<out>/tp<i>_<id>/`, containing `pointcloud_pass<k>/`, the commanded path of every pass, the preview, `run_info.txt` and `done.flag`.

The loop runs this script as a subprocess in the robot's Python environment and then reads the capture folders. If a pass finished without a capture, or fewer passes ran than planned, the loop stops and waits for the operator.

## Logs and resuming

| Location | Contents |
|---|---|
| `OUT/round_00/` | The reference and its preview. |
| `OUT/round_NN/` | What the model saw (`cur.png`, `tgt.png`, `prompt.txt`), what it answered (`response.json`), what was executed (`toolpath.json`, `preview.png`, the robot's output folder), and the measured field after every pass (`field_<id>_pass<k>.npz` / `.png`). |
| `OUT/state.json` | Rewritten every round. `--resume` continues from it after an interruption. |
| `OUT/summary.png` | The target, the final shape, and progress after each pass. |

## Running it

Run the loop from the folder that contains `ew_loop/`. The loop needs numpy, scipy and matplotlib. A local model also needs unsloth and CUDA.

```
# Model service on a GPU machine. Skip this if the robot PC has an NVIDIA GPU: give the loop --ckpt instead.
HF_HUB_OFFLINE=1 UNSLOTH_DISABLE_STATISTICS=1 CUDA_VISIBLE_DEVICES=0 \
  python -m ew_loop.cli serve --ckpt <checkpoint folder> --port 8765

# Target height field, here taken from a simulated trajectory.
python -m ew_loop.cli target --from-sim cycles.npz --cycle -1 --sheet 195x190 --clamp 100 --out target.npz

# Once per rig setup: check that camera 3 finds the sheet.
python run_toolpath_sequence.py baseline --out C:\runs\setup              # robot env, in code/
python -m ew_loop.cli reference --capture C:\runs\setup\baseline --sheet 195x190 --clamp 100 \
       --camera merged --calib "<robot repo>\code\mvs_recorder\calibration results" \
       --feed-dir 1,-0.35,0 --out C:\runs\setup\ref
#   1,-0.35,0 is the value for camera 3 in the August-September captures; check it on the current rig.
#   reference_preview.png: +y must point from the gripped edge to the free edge

# The loop.
python -m ew_loop.cli loop --sheet 195x190 --clamp 100 --thickness 1.0 --target target.npz \
       --out C:\runs\sheet01 --server http://<gpu machine>:8765 \
       --plant robot --robot-code <robot repo>\code --robot-python C:\anaconda3\envs\EnglishWheel\python.exe \
       --camera merged --calib "<robot repo>\code\mvs_recorder\calibration results" \
       --feed-dir 1,-0.35,0 --n 3 --max-passes 20
```

Offline, without the machine:

```
python -m ew_loop.cli loop --plant sim [--fake-camera] --sheet 300x300 --clamp 0 --thickness 0.5 \
       --target tgt.npz --out sim_run --server <url> --allow-rot90 --yes   # simulator in place of the robot
python robot_pr/dry_run_toolpath_sequence.py                               # robot script + loop, robot faked
python -m ew_loop.tests.test_scan_to_field                                 # likewise test_model_io, test_frames,
                                                                           # test_spec, test_state, test_predict_local
python bridge_checks/real_scan_to_field_check.py OUT                       # measurement on recorded real captures
```

## What has been verified, and how

| What | How | Result |
|---|---|---|
| Model inputs identical to training | `tests/test_model_io.py` | Prompts, plan schema, zigzag geometry and metrics are identical. |
| Zigzag corner points equal the robot's planner | `tests/test_spec.py`, 54 cases | Within 1e-3 mm. |
| Measurement on a simulated camera (upper wheel hiding part of the sheet, gripper bar, blended edge pixels, 0.8 mm depth noise, another surface in range) | `tests/test_scan_to_field.py` | Sheet located within 0.1° and 0.5 mm. Heights within 0.34 mm RMS (99th percentile 0.9 mm), also after the sheet moved 100 mm under the wheel. A wrong axis sign shows as an 80 mm mismatch and stops the loop. |
| Camera noise and the sheet in real captures | `bridge_checks/inplace_cloud_noise_probe.py`; Run 2 with the sheet located by hand in the infrared image (`figures/q2_*.png`, `figures/q3_*.png`) | One camera measures a flat surface with 0.8 mm point noise. On the sheet the cameras disagree by 2–6 mm with every calibration we have. Camera 3 measured about two thirds of the 300 × 195 mm sheet after pass 5; merging the three cameras adds 11–18 % more cells. The automatic search in the reference step picked a perforated table of similar width instead of the sheet, so the sheet must be confirmed by eye once per rig (see the end of this table). |
| Robot script and loop with the robot faked | `robot_pr/dry_run_toolpath_sequence.py` | Capture positions are identical to the robot's own kinematics (0 mm). The loop reached its target in 6–7 passes (the fake camera noise differs per run), and every pose check passed. |
| Three-camera merge | `tests/test_merge.py` | Per-camera depth clipping, the calibration transform, and the rule for using the ICP refinement only for its own calibration run. |
| Model and loop on the simulator | `cli loop --plant sim`, the 7 tasks of the offline MPC evaluation that start from a flat sheet | Same outcome as the offline evaluation on 5 of 7 tasks; one stopped at amplitude 0.946 instead of converging; one, already off-shape offline, ended further off (table below). Through the fake cameras (56–69 % of the sheet visible), C1/t018 reached the target in 3 passes: true amplitude 0.997, correlation 0.985. |

The reference step chooses the flat surface whose outline has the sheet's width. In the August captures that rule picked a perforated table of similar width in camera 3's view, and an earlier version of the real-data check (`bridge_checks/real_scan_to_field_check.py`) reported numbers for that table. Until the reference step shows its choice in the camera's own view, check `reference_preview.png` against the camera image before forming.

Simulator loop against the offline evaluation (`experiments/mpc_end2end/runs/mpc_n3_perwa_decay08_ampguard`), same model and simulator. Each cell: how it stopped · passes · shape correlation · amplitude ratio.

| Task | Offline evaluation | This loop |
|---|---|---|
| C1/t018, flat to cycle 3 | converged · 4 · 0.997 · 0.98 | converged · 4 · 0.996 · 0.98 |
| C1/t044, flat to cycle 5 | converged · 8 · 0.994 · 0.98 | converged · 12 · 0.994 · 0.97 |
| C2/t030, flat to cycle 8 | shape off · 8 · 0.924 · 0.95 | no progress · 10 · 0.826 · 0.91 |
| C3/t010, flat to cycle 8 | converged · 5 · 1.000 · 1.01 | converged · 5 · 0.999 · 1.00 |
| C3/t012, flat to cycle 10 | converged · 13 · 1.000 · 0.99 | next pass would overshoot · 8 · 1.000 · 0.95 |
| C4/t018, flat to cycle 4 | converged · 2 · 0.999 · 1.11 | converged · 2 · 0.999 · 1.10 |
| C4/t020, flat to cycle 6 | shape off · 6 · 0.902 · 0.97 | shape off · 6 · 0.902 · 0.98 |

The fake-camera run also shows the cost of partial coverage: the measured amplitude ratio read 0.45, 0.80 and 1.08 after the three passes while the true values were 0.41, 0.74 and 1.00; the measured correlation matched the true one within 0.001.

## To settle on the machine, in order

1. **Camera setup, once per rig.** Set `--feed-dir` and check `reference_preview.png`.
2. **Height sign.** Does the sheet side that faces the cameras correspond to the simulator's +z? Wheel one known zigzag and compare the direction of the measured dome with the simulation of the same zigzag. If they are opposite, add `--up-away`.
3. **X sign of the robot frame** (`frames.PLANNER_X_SIGN`). The pose check reports it on the first pass that moves the sheet sideways: a wrong sign shows as a sideways mismatch of twice the shift.
4. **Wheel-line distance for the 190 mm sheet** (`--clamp`). With the robot at its start pose, measure from the gripped edge to the wheel contact. The 100 mm robot constant was set on a 300 × 200 sheet.
5. **Coverage.** Coverage is the share of the compared area that has a measured height. In the August captures (Run 2, 300 × 195 mm sheet) camera 3 measured about 68 % of the sheet after pass 5. The gaps come from the upper wheel blocking the line of sight, the gripper, and shiny spots where the polished sheet reflects the camera's own projector. The new sheet is smaller and its formed area sits closer to the wheel, so check `coverage_usable` on the first real capture. Partial coverage also biases the progress figure: in the simulator run through the fake cameras, with 56–69 % of the sheet visible, the measured amplitude ratio read about 10 % above the true one. If coverage is too low, capture at a scan position (sheet moved clear of the upper wheel); the measurement code needs no change, only the sheet point the wheel faces at that position.
6. **Clamped versus released shape.** On Run 2, the sheet in the gripper after pass 5 spanned about 35 mm out of plane; the hand-held scan of the released sheet spanned about 70 mm. The overall form agrees (correlation 0.65 after matching orientation), but the released sheet curves much more across its width. The loop measures the clamped state, while the part's final shape is the released state. Decide which of the two the target describes. The simulator with the two-pad clamp gives the clamped state after every pass and the released state after the last one.
7. **GPU on the robot PC.** If it has one, use `--ckpt` locally; otherwise run `serve` on a GPU machine and give the loop `--server`.
8. **Camera agreement.** The reference step prints how far the three cameras disagree on the flat sheet. Above 2 mm, recalibrate or fall back to one camera.
9. **Retraining.** The current checkpoint knows only 300 × 300 × 0.5 mm sheets. Most of its plans for the 195 × 190 × 1.0 sheet are refused (orientation 90, working areas too large). Retrain after the simulator has been calibrated to the real sheet. Measured images will never match training images pixel for pixel (holes where the wheel hides the sheet, noise, a different white margin), and the current model reacts to small margin changes. The retraining data should therefore randomise exactly these: margins, holes and height noise.

## Design choices in short

- **Measurements decide.** Stop rules and block sizes use only measured fields. When a measurement and the simulation disagree, the simulator is recalibrated; the measurement is not corrected.
- **One definition of everything.** `frames.py` holds every frame constant. The robot executes the corner points from `toolpath.json`. The prompt and plan schema are a checked copy of the training code.
- **Standalone.** `ew_loop/` runs on the robot PC by copying the folder, with no other code from the research repository.

## Files

| Path | Contents |
|---|---|
| `ew_loop/frames.py` | Sheet geometry, frames and the height-field format. |
| `ew_loop/scan_to_field.py` | Point clouds (merged or one camera) to height field, and the reference capture. |
| `ew_loop/predict.py` | The model: local, HTTP or fixed plan, plus the HTTP service. |
| `ew_loop/state.py` | Metrics, stop rules and block size. |
| `ew_loop/spec.py` | Plan to `toolpath.json`, plus the preview. |
| `ew_loop/loop.py` | The loop, with the robot and simulator plants. |
| `ew_loop/synthetic.py` | Fake camera rig for tests. |
| `ew_loop/model_io.py` | Prompt, schema and zigzag geometry shared with training. |
| `ew_loop/cli.py` | Command-line entry points. |
| `ew_loop/tests/` | Tests. |
| `robot_pr/` | Reference robot side (script, patch, dry run); the robot team may use, adapt or replace it. |
| `implementation_requirements.md` | The short handover for the robot team: contract, steps, checks. |
| `bridge_checks/` | One-off checks against the robot code and the recorded real data. |
