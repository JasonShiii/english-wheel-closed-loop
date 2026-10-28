"""ew_loop: closed-loop forming on the robotic English wheel (model -> robot -> cameras -> model).

Read CLOSED_LOOP.md (one folder up) first. Modules, in loop order:
  frames.py        sheet geometry, coordinate frames, the height-field format
  predict.py       the planning model (local GPU or HTTP)
  spec.py          model plan -> toolpath.json for the robot
  scan_to_field.py camera point cloud -> height field in the sheet frame
  state.py         metrics, stop rules, how many passes to run before re-planning
  loop.py          the loop itself, with a simulator plant and a robot plant
  synthetic.py     fake camera rig for offline tests
  model_io.py      prompt / schema / zigzag geometry shared with the trained model

The loop itself needs only numpy, scipy and matplotlib (plus unsloth for a local model).
The simulator plant and the tests also use modules of the research repository; they call
add_repo_paths() to make those importable.
"""
import os
import sys

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def add_repo_paths():
    for p in (PROJECT,
              os.path.join(PROJECT, "experiments", "mpc_end2end"),
              os.path.join(PROJECT, "experiments", "category_tests")):
        if p not in sys.path:
            sys.path.insert(0, p)
