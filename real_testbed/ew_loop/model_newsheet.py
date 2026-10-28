"""Model profile of the new-sheet checkpoints (newsheet_pipeline, prompt schema v5facts, 2026-10-24 data):
how a HeightField is drawn for the model, how the prompt is built, how executed work is written into the
history, how the answer is parsed. Everything is taken from the training code itself (newsheet_pipeline/render.py,
pairs.py, facts.json) through GPT-astra_comparison/planners.py, so training and deployment cannot drift.

A planner (predict.ModelPlanner, predict.HttpPlanner, predict.FixedPlanner) takes a profile; to deploy a model
trained with another renderer or prompt, write another profile with the same four methods.
"""
import os
import sys

RT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # real_testbed
ROOT = os.path.dirname(RT)
for _p in (os.path.join(ROOT, "experiments", "newsheet_pipeline"), os.path.join(ROOT, "experiments", "GPT-astra_comparison"),
           os.path.join(ROOT, "experiments", "mpc_end2end"), os.path.join(ROOT, "experiments", "category_tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import render as R                        # noqa: E402  THE renderer of the new model
from .model_io import entry_desc, wa_desc, zz_desc   # noqa: E402

_PL = None


def planners():
    """GPT-astra_comparison/planners.py (build_prompt v5facts, executable), imported on first use only: a
    robot-PC copy that talks to `serve-fields` over HTTP renders and keeps history but never builds a prompt,
    so it needs neither planners.py nor the training data behind it."""
    global _PL
    if _PL is None:
        import planners as PL             # noqa: E402  needs the research repo (eval_execute, category_tests)
        _PL = PL
    return _PL

NEWSHEET = os.path.join(ROOT, "experiments", "newsheet_pipeline")
DEFAULT_PAIRS = os.path.join(NEWSHEET, "pairs_231x174_free_clean")
DEFAULT_DATA = os.path.join(NEWSHEET, "data_231x174_free_g9e-4")
DEFAULT_CKPT = os.path.join(NEWSHEET, "runs", "free_aug_e8", "checkpoint-1832")
SCHEMA = "v5facts"


class NewsheetModel:
    name = "newsheet-v5facts"

    def __init__(self, pairs=DEFAULT_PAIRS, data=DEFAULT_DATA):
        self.pairs, self.data = pairs, data
        self._configured = False

    def _pl(self):
        PL = planners()
        if not self._configured:
            PL.configure(pairs=self.pairs, data=self.data, tag="rig"); self._configured = True
        return PL

    def render(self, field, path):
        return R.field_png(field, path)

    def build_prompt(self, cz, tz, history):
        return self._pl().build_prompt(cz, tz, history, schema=SCHEMA)

    @staticmethod
    def history(ops):
        """executed ops (wa dicts with n) -> the prompt's history list; consecutive identical
        area+zigzag entries are merged, hits_done accumulated."""
        out = []
        for wa in ops:
            if out and out[-1]["wa"] == wa_desc(wa) and out[-1]["zigzag"] == zz_desc(wa):
                out[-1]["hits_done"] += int(wa["n"])
            else:
                out.append(entry_desc(wa, int(wa["n"]), "hits_done", "v2"))
        return out

    def executable(self, plan):
        return self._pl().executable(plan)
