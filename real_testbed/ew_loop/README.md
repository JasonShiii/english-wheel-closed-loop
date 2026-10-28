# ew_loop

The loop software for closed-loop forming on the robotic English wheel: camera captures in, `toolpath.json` out, repeated until the measured sheet matches the target. Plain Python (numpy, scipy, matplotlib). Copy this folder to the robot PC and run everything from the folder that contains it:

```
python -m ew_loop.cli --help
python -m ew_loop.tests.test_spec        # likewise test_frames, test_state, test_scan_to_field, test_merge, test_model_io
```

`CLOSED_LOOP.md` (one folder up) explains the loop step by step; `implementation_requirements.md` is the handover for the robot side. This file explains the code.

## One round of the loop

```
captures (robot) --scan_to_field--> HeightField --predict--> plan --state--> how many passes
                                                                  |
robot executes toolpath.json <--spec-- first zigzag of the plan <--+
```

`loop.py` runs this round until a stop rule holds. Every round is written to `OUT/round_NN/`.

## Modules

### frames.py: the conventions

Holds everything that must be agreed between the model, the simulator, the cameras and the robot: the sheet frame (origin at the sheet centre, x across, y along the feed toward the free edge), the robot planner frame (origin on the wheel line) and the conversion between them, the robot's safe-zone formula, and the `HeightField` format (nodes of a 2 mm grid over the sheet, NaN where there is no data, saved as `.npz`). `Sheet` is a frozen dataclass with the sheet size, thickness, the wheel-line distance (`clamp_mm`), the safe-zone margins and the gripper pad positions. `field_png` draws a field exactly as the training images were drawn (through `model_io.heatmap_png`). `mask_clamp_band` sets the band behind the wheel line to no-data; it is applied to every field the model sees, the target included.

### scan_to_field.py: captures to height fields

Input: a capture folder as the robot writes it (`device_<k>_pointcloud_001.ply`, binary, mm, each camera's own frame). `merge_capture` clips each camera to 200–700 mm depth and transforms it into the device_1 frame with the rig calibration (`extrinsics.json`, replaced by `icp/refined_transforms.json` when that file is accepted for the same calibration run); `load_capture(..., camera=3)` reads one camera instead.

`make_reference` is run once per sheet on the flat capture. It fits the sheet plane (RANSAC on the dominant plane, then the plane whose outline has the sheet's width), fits the known rectangle to the visible outline (free edge and side edges; the gripped edge is hidden), records what sits above the sheet near the wheel (the upper wheel, fixed in the camera view) and along the gripped edge (the gripper, fixed on the sheet), and grids the flat sheet's own height as the baseline. `camera_agreement` reports how far the cameras disagree on the flat sheet. `reference_preview` draws the result for the operator.

`scan_to_field` turns every later capture into a `HeightField`: heights are distances from the reference plane; the sheet's position comes from the robot motion (which sheet point is under the wheel at capture time, `contact_xy`), while the capture's own outline is fitted as a check (`meta.pose_check`); points are gridded by nearest node with a median per node, the wheel and gripper nodes are dropped, outliers against the 7 × 7 neighbourhood median are dropped, one node is eroded off every boundary, the baseline is subtracted, and the clamp band is masked. `meta.coverage_usable` says what fraction of the area beyond the wheel line has data.

### predict.py: the planning model

`LocalPlanner` loads the checkpoint (unsloth, 4-bit) on a local GPU; `HttpPlanner` sends the same request to `python -m ew_loop.cli serve` on another machine; `FixedPlanner` answers with a fixed plan file (to test everything but the model). All three take the two rendered images (bytes), the two height ranges and the history, and return the model's text and the parsed plan. `predict()` renders the current and target fields, builds the prompt, calls the planner, parses the answer, and writes `cur.png`, `tgt.png`, `prompt.txt` and `response.json` into the round folder. `serve()` is a stdlib HTTP server with `/health` and `/predict`.

### model_io.py: what must match training

A copy of the prompt text, the history and plan schema, the image renderer, the working-area-to-zigzag rule and the zigzag corner-point geometry from the training code. `tests/test_model_io.py` checks the copy against the originals in the research repository. If the model is retrained with another prompt or schema, this file changes with the checkpoint.

### state.py: progress, stopping, block size

`compare` gives the amplitude ratio (robust span of the current field over the target's) and the shape correlation on the nodes both fields have. `LoopState` keeps the executed history in the model's schema, the metrics after every measurement (one per block on the rig: `record_block` splits the block's gain into per-pass increments decaying by 0.8; one per pass in the simulator: `record_pass`), the measured height gain per pass of each working area, and decides: `check_stop` (amplitude 1.0 or 0.95 reached, budget, no progress over three re-plans) and `block_size` (`n` passes far from the target, 2 then 1 near it; one probe pass on an area never measured; then as many passes as the decayed measured gain keeps below the target). The rules are the ones adopted in the offline MPC evaluation. `to_dict`/`from_dict` make `--resume` possible.

### spec.py: plan to toolpath.json

`wa_to_toolpath` turns one plan entry (working area, orientation, strip count, passes) into a zigzag: strip length and angle by the training-data rule (`inscribe`), corner points by the training-data geometry (`zigzag_vertices`), moved into the robot planner frame and reversed to the robot's traversal order. It refuses, with a reason, orientation 90 (not executable yet), corner points outside the safe zone, and N < 3. `toolpath_doc` and `write_toolpath_json` produce the file; `preview_png` draws every zigzag on the sheet with the wheel line, the clamped band, the safe zone and the pads.

### loop.py: the loop and the two plants

`run_loop` is the round described above plus the bookkeeping and the files. A plant executes a `toolpath.json` and returns captures (points, a field, or a capture folder). On the rig (2026-10-28) the cadence is one capture per block: `FilePlant` writes the round's `toolpath.json` (and a copy into `--exchange`), the robot side runs the whole block, scans once into `--captures` (its own run folder, or `<exchange>/captures/`) as `pointcloud_pass<k>`, and the operator presses Enter; `ReplayPlant` returns the recorded capture of a run folder after the block's passes, for rehearsals. The first setup's plants measure after every pass: `RobotPlant` runs `run_toolpath_sequence.py baseline|run` in the robot's Python environment as a subprocess and reads the capture folders it wrote; it checks that every planned pass produced a capture. `SimPlant` runs the C++ simulator with the training physics (300 × 300 × 0.5 mm sheet only), warm-started from pass to pass; with `--fake-camera` every simulated pass is rendered through the fake rig so the measurement code runs as on the machine. `capture_contact` encodes where the sheet is at each capture: after an odd pass the zigzag's last corner point is under the wheel, after an even pass the first one.

### synthetic.py: the fake camera rig

`FakeRig.capture(surface, contact_xy)` returns what one rig camera would record for a given sheet surface and robot position: a tilted view at 450 mm, 0.8 mm depth noise, the upper wheel standing on the sheet and hiding part of it, the gripper bar, blended pixels behind the sheet edges, another flat object in range and the floor. Used by the tests and by `SimPlant --fake-camera`.

### measure.py: the rig measurement (2026-10-28)

`RigMeasurer.reference(run)` on the flat scan: three cameras merged with the rig calibration, gate box, sheet finder
(the RANSAC plane whose inliers best fill the 231 x 174 rectangle), sheet frame with the gripped edge found from the
gripper bar, per-camera plane correction, hardware mask. `field(pass_dir)` for every later capture at the same scan
pose: band around the flat plane, median per grid node (14 mm fallback for sparse spots), despeckle, crop below the
crop line, best-fit plane removed: the same two last steps the training fields went through (`pairs.py`). The chain
is `bridge_checks/danish_run2_measure.py`, which it imports. No robot pose is needed (fixed scan pose).

### model_newsheet.py: the model profile (2026-10-28)

What a planner needs to know about the trained model: `render` (newsheet_pipeline/render.py), `build_prompt`
(schema v5facts through GPT-astra_comparison/planners.py), `history` (prompt schema of executed work),
`executable`. Another model = another profile with the same four methods; `predict.ModelPlanner`,
`HttpFieldsPlanner` and `FixedFieldsPlanner` take a profile and expose `plan_fields(cur, tgt, history, outdir)`.

### cli.py: commands

| Command | Does |
|---|---|
| `serve` | model service on a GPU machine |
| `target` | height-field target from a simulated trajectory |
| `reference` | the one-time flat capture step; writes `reference_preview.png` |
| `measure` | one capture to a height field |
| `plan` | one re-plan only (model to `toolpath.json` + preview) |
| `check` | re-check a `toolpath.json` against its sheet and draw it |
| `loop` | the loop of the first setup, with `--plant robot` or `--plant sim` |
| `rigloop` | the loop on the 2026-10-28 rig setup: `--plant file --captures DIR` (the robot software's run folder; `--exchange DIR` to copy files by hand) or `--plant replay --replay-run DIR` (recorded run); model by `--server URL`, `--ckpt` or `--fixed-plan` |
| `serve-fields` | model service that takes height fields and returns the plan (for `rigloop --server`) |

## Interfaces with the robot side

| Direction | What | Where defined |
|---|---|---|
| loop to robot | `toolpath.json`: sheet block (size, thickness, wheel-line distance, margins) and an ordered list of zigzags, each with `pivots_xy_mm` in the robot planner frame and `passes` | `spec.py` |
| robot to loop | capture folders `pointcloud_pass<k>/device_<1..3>_pointcloud_001.ply`, the flat capture `baseline/`, the calibration folder `calibration results/` | `scan_to_field.py`, `loop.RobotPlant` |
| both | the sheet point under the wheel at each capture (end of each pass) | `loop.capture_contact` |

The robot side executes the corner points as given and never re-derives them; the loop never reads anything from the robot but the capture folders and the exit code. The reference robot-side implementation is `../robot_pr/run_toolpath_sequence.py` with `../robot_pr/zigzag_forming_toolpath_api.patch`.

## The copy for the robot PC

`../make_robot_pc_package.py` builds `../robot_pc_package/` (and a dated zip): this folder plus the modules it imports (`bridge_checks/{danish_run2_measure,plate_refill,danish_run2_metrics}.py`, `experiments/newsheet_pipeline/{cfg,pairs,render}.py`, `experiments/mpc_end2end/common.py`), the targets, the rig calibration, the fixed recipe, the robot-side reference and the documents, in the repository layout. Its `requirements.txt` is numpy, scipy, matplotlib, opencv-python-headless. The copy runs `rigloop` with `--server` (model on a GPU machine through `serve-fields`) or `--fixed-plan`; `model_newsheet.py` imports the prompt code (`planners.py`, research repository) only when a prompt is built locally (`--ckpt`). `../ROBOT_PC.md` is the guide.

## Tests

| Test | Checks |
|---|---|
| `test_model_io` | prompt, schema, renderer, zigzag geometry and metrics identical to the training code |
| `test_frames` | frame conversions, safe zone equal to the robot's formula, field format and rendering |
| `test_spec` | corner points equal to the robot's own planner (54 cases), refusals, file round trip |
| `test_state` | metrics, stop rules, block-size decisions, save/load |
| `test_merge` | per-camera depth clipping, calibration transform, ICP file rule |
| `test_scan_to_field` | reference and measurement on the fake rig: pose within 0.1° and 0.5 mm, heights within 0.34 mm RMS, wrong axis sign caught |
| `test_predict_local` | the model answers on a GPU; local and HTTP give the same plan (needs the checkpoint) |
| `test_rig_port` | the rig measurement reproduces the 10-21 chain, the profile's prompt and renderer equal the training code, a recipe converts to a toolpath |
| `../robot_pr/dry_run_toolpath_sequence.py` | the robot script and the whole loop with the robot faked, capture positions against the robot's own kinematics |
