"""Planner backends for the closed-model comparison. Every system answers through one call, the
signature of real_testbed/ew_loop/predict.LocalPlanner.plan:

    plan(cur_png, tgt_png, cz, tz, history, sample=False, tag=None)
        -> {"raw", "plan", "secs", "usage", "finish", "inputs_sha", "cached", "key", ...}

  ClosedPlanner(model, level)   closed VLM through OpenRouter, level 0shot / 5shot / 20shot = number of
                                solved training samples sent as earlier turns; nothing else is added.
                                The process facts are a section of the prompt itself (schema v5facts:
                                <data>/facts.json written by newsheet_pipeline/facts.py and inserted by
                                pairs.py), the same for our model and the closed models
  OursPlanner(ckpt)             our checkpoint, generation code of mpc_rollout.generate
  LabelPlanner()                free oracle that answers with the sample's own label (plumbing test of
                                the record / cache / execution path; its executed shapes must sit on the
                                ground-truth replays' noise floor)

Data location (2026-10-20, new-sheet mode): --pairs <dir> (train.jsonl / val.jsonl / images, made by
newsheet_pipeline/pairs.py) and --data <dir> (the trajectories: manifest.json, <id>/record.json,
<id>/cycles.npz in mm, made by gen.py) default to newsheet_pipeline/pairs_231x174_g9e-4 and
data_231x174_g9e-4; --tag names the result layout (results_<tag>/, icl_examples_<tag>.json,
RESULTS_<tag>.md; "" = the pre-retrain layout results/, icl_examples.json, RESULTS.md of
mpc_end2end/pairs_pilot). Categories are read from the data (Z1..Z4 now, C1..C4 before), never
hardcoded. The runners call configure(args) once after parsing.

Identical by construction for every system: the prompt text (the open-loop runner passes the training
prompt of the jsonl verbatim through plan_prompt; plan() rebuilds it with newsheet_pipeline/pairs.build_prompt
for schema v5 / v5facts or mpc_end2end/common.build_prompt for the old schemas), the image bytes (the PNG
files of the pairs dir go out as they are, never re-rendered), the parser (ew_loop.predict.parse_plan) and
the retry rule (plan_with_retry). Every answer is cached in cache/<system>/<level>/<sha256 of the request>.json
and never requested twice.
"""
import base64
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MPC = ROOT / "experiments" / "mpc_end2end"
NEWSHEET = ROOT / "experiments" / "newsheet_pipeline"
for _p in (ROOT / "real_testbed", ROOT / "experiments" / "category_tests", MPC, NEWSHEET):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import cfg                                          # noqa: E402  newsheet_pipeline/cfg.py: sheet, dose, BC, sim command
from eval_execute import plan_to_was                # noqa: E402  the executor's validity check
from ew_loop.predict import parse_plan              # noqa: E402  the loop's parser

CACHE = HERE / "cache"
KEY_FILE = ROOT / "openrouter_api.txt"
LEVELS = ("0shot", "5shot", "20shot")
OURS_LEVEL = "trained"          # our checkpoint: no examples, the training prompt
LABEL_LEVEL = "oracle"          # the label planner
MAX_TOKENS = 512                # our model's max_new_tokens
# Closed models: hidden reasoning tokens count against max_tokens. Probe 2026-10-07 (check_inputs.py
# --probe astra): at 512 GPT-6 Astra spent all 512 tokens reasoning and returned an empty answer; at
# 16000 it reasoned 516 tokens and answered in 40. Answer lengths are reported next to ours.
CLOSED_MAX_TOKENS = 64000       # 2026-10-20: raised for the maximum reasoning setting (Anthropic's budget = 0.95 x max_tokens)
CLOSED_REASONING = "max"        # 2026-10-20 decision: each model at ITS maximum effort (MAX_EFFORT below); "default" sends nothing
# OpenRouter's unified reasoning.effort accepts max/xhigh/high/medium/low/minimal/none (docs read 2026-10-20). The
# provider maxima differ: OpenAI GPT-5.1+/6 take "xhigh"; Anthropic converts the effort to a token budget
# (max/xhigh = 0.95 x max_tokens, capped at 128k); Gemini 3/4 map xhigh down to "high" (their highest thinkingLevel).
MAX_EFFORT = {"openai/": "xhigh", "anthropic/": "max", "google/": "high"}
SAMPLE_TEMPERATURE = 0.7        # the one sampled retry (mpc_rollout.generate)
SCHEMAS = ("v5facts", "v5", "v6lean", "v1", "v1facts", "v2")
SCHEMA_DEFAULT = "v5facts"      # the new pipeline's prompt (pairs.PROMPT_V5 + the facts of <data>/facts.json)

# ---- data location (set once per run by configure) --------------------------------------------
DEFAULT_PAIRS = NEWSHEET / "pairs_231x174_g9e-4"
DEFAULT_DATA = NEWSHEET / "data_231x174_g9e-4"
DEFAULT_TAG = "231x174"
PAIRS = DATA = TAG = RESULTS = EXAMPLES = SUBSET = REPORT = None
_CATS = None

# OpenRouter ids and parameter support checked against https://openrouter.ai/api/v1/models on
# 2026-10-07 (prices: USD per million input / output tokens). temperature: whether the endpoint
# accepts one. By default no temperature is sent to any closed model (--temperature default): Google
# advises against lowering it for Gemini 3, Anthropic requires 1 when thinking is on, and GPT-6 Astra,
# GPT-5.2 and Claude Fable 5.1 do not take one. --temperature zero applies PROPOSAL §2's 0 / 0.7
# where the endpoint accepts it.
MODELS = {
    "astra":           {"id": "openai/gpt-6-astra",            "temperature": False},  # 10 / 50
    "gpt5.2":          {"id": "openai/gpt-5.2",                "temperature": False},  # 1.75 / 14
    "gemini3.1-pro":   {"id": "google/gemini-3.1-pro-preview", "temperature": True},   # 2 / 12
    "gemini4-argon":   {"id": "google/gemini-4-argon",         "temperature": True},   # 2 / 10
    "claude-fable5.1": {"id": "anthropic/claude-fable-5.1",    "temperature": False},  # 10 / 50
    "claude-opus5.5":  {"id": "anthropic/claude-opus-5.5",     "temperature": True},   # 4 / 20
}


def configure(pairs=None, data=None, tag=None):
    """Set the data location and the result layout. Accepts an argparse namespace (add_data_args) or
    the three values; None keeps the defaults (the new pipeline)."""
    global PAIRS, DATA, TAG, RESULTS, EXAMPLES, SUBSET, REPORT, _CATS
    if hasattr(pairs, "pairs"):                     # argparse namespace
        pairs, data, tag = pairs.pairs, pairs.data, pairs.tag
    PAIRS = Path(pairs or DEFAULT_PAIRS).resolve()
    DATA = Path(data or DEFAULT_DATA).resolve()
    TAG = DEFAULT_TAG if tag is None else tag
    sfx = f"_{TAG}" if TAG else ""
    RESULTS, EXAMPLES = HERE / f"results{sfx}", HERE / f"icl_examples{sfx}.json"
    REPORT = HERE / f"RESULTS{sfx}.md"
    # the pre-retrain layout kept its subset in mpc_end2end/runs (eval_symbolic.py); the new one is written by make_subset.py
    SUBSET = RESULTS / "eval_subset.json" if TAG else MPC / "runs" / "eval_subset.json"
    _CATS = None


configure()


def add_data_args(ap):
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS), help="pairs dir (train.jsonl, val.jsonl, images/) of newsheet_pipeline/pairs.py")
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="trajectory dir (manifest.json, <id>/record.json, <id>/cycles.npz) of gen.py")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="result layout: results_<tag>/, icl_examples_<tag>.json, RESULTS_<tag>.md; '' = the pre-retrain layout "
                         "(results/, icl_examples.json, RESULTS.md, subset mpc_end2end/runs/eval_subset.json)")


def categories(rows=None):
    """Categories of the data, in manifest order (Z1..Z4 on the new sheet; C1..C4 on the pilot); from the
    rows if the data dir has no manifest."""
    global _CATS
    if _CATS is None:
        man = DATA / "manifest.json"
        if man.exists():
            _CATS = list(dict.fromkeys(c["category"] for c in json.load(open(man))["cases"]))
        elif rows is not None:
            return sorted({r["meta"]["category"] for r in rows})
        else:
            raise FileNotFoundError(f"{man} (categories are read from the data)")
    return _CATS


def facts_text(schema=SCHEMA_DEFAULT):
    """The process facts inside the prompt: the list of <data>/facts.json (schema v5facts; None when the
    file does not exist = schema v5), or common.PROCESS_FACTS for the pilot's v1facts."""
    if schema == "v1facts":
        from common import PROCESS_FACTS
        return [PROCESS_FACTS]
    fp = DATA / "facts.json"
    return json.load(open(fp))["facts"] if schema == "v5facts" and fp.exists() else None


def build_prompt(cz, tz, history, schema=SCHEMA_DEFAULT):
    """plan()'s prompt: the training prompt of the schema. v5 / v5facts = newsheet_pipeline/pairs.build_prompt
    (sheet, reachable zone and facts from cfg and <data>/facts.json); v1 / v1facts / v2 = mpc_end2end/common."""
    if schema in ("v5", "v5facts"):
        import pairs as NP                          # newsheet_pipeline/pairs.py (its FACTS_TEXT is the module state pairs.py sets)
        NP.FACTS_TEXT = facts_text(schema)
        return NP.build_prompt(cz, tz, history)
    if schema == "v6lean":                          # pairs_to_lean.py's request-only prompt (runs/free_clean_leanpv_e8)
        import pairs as NP
        return NP.build_prompt_lean(cz, tz, history)
    from common import build_prompt as _bp
    return _bp(cz, tz, history, schema=schema)


# ---- shared helpers -----------------------------------------------------------------------
def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def inputs_sha(cur_png, tgt_png, prompt):
    """Fingerprint of what a planner was asked; check_inputs.py compares it across systems."""
    return {"cur": sha(cur_png)[:16], "tgt": sha(tgt_png)[:16], "prompt": sha(prompt)[:16]}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_subset(path=None):
    """Indices into val.jsonl: make_subset.py's {"rule", ..., "indices"} or the pilot's plain list."""
    s = json.load(open(path or SUBSET))
    return s["indices"] if isinstance(s, dict) else s


def sample_inputs(row, base=None):
    """(cur PNG bytes, tgt PNG bytes, prompt text) of a pairs row, exactly as training saw them."""
    content = row["messages"][0]["content"]
    cur, tgt = [(Path(base or PAIRS) / c["image"]).read_bytes() for c in content if c["type"] == "image"]
    return cur, tgt, next(c["text"] for c in content if c["type"] == "text")


def label(row):
    return row["messages"][1]["content"][0]["text"]


def executable(plan):
    return plan is not None and plan_to_was(plan) is not None


def sample_tag(meta):
    """sample identity (source + cut), never an index: results of a smaller selection are reused by a larger one"""
    return f"{meta['source'].replace('/', '-')}_c{meta['cut'][0]}-{meta['cut'][1]}"


def pick_rows(rows, per_cat=None, cats=None):
    """eval_execute.py's selection applied to the whole subset: per category (the data's categories, in
    manifest order), in subset order, the first per_cat/2 flat starts then the first per_cat/2 intermediate
    starts; per_cat None = every row (flat starts first). Returns [(tag, row)] with tag = sample identity."""
    out = []
    for cat in cats or categories(rows):
        r = [x for x in rows if x["meta"]["category"] == cat]
        z0 = [x for x in r if x["meta"]["cut"][0] == 0]
        mid = [x for x in r if x["meta"]["cut"][0] > 0]
        sel = z0 + mid if per_cat is None else (z0[: per_cat // 2] + mid[: per_cat - per_cat // 2])[:per_cat]
        out += [(sample_tag(x["meta"]), x) for x in sel]
    return out


def plan_with_retry(planner, cur_png, tgt_png, prompt, tag=None):
    """mpc_rollout.generate's rule for every system: one deterministic attempt, then one sampled
    attempt if the plan cannot be executed (unparsable, empty or invalid geometry).
    Returns (final answer, [all attempts])."""
    first = planner.plan_prompt(cur_png, tgt_png, prompt, sample=False, tag=tag)
    if executable(first["plan"]):
        return first, [first]
    second = planner.plan_prompt(cur_png, tgt_png, prompt, sample=True, tag=tag)
    return second, [first, second]


def call_summary(ans):
    return {k: ans.get(k) for k in ("attempt", "secs", "usage", "finish", "provider", "cached", "key")}


def _cached(system, level, key, call, meta):
    """The stored record for `key`, or make the call once and store it. API errors raise and are not
    stored (a rerun retries them); every answer the model gave is stored, empty or not."""
    f = CACHE / system / level / f"{key}.json"
    if f.exists():
        return {**json.loads(f.read_text()), "cached": True}
    rec = {**meta, **call(), "key": key, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(f)
    return {**rec, "cached": False}


def load_examples(k):
    """The fixed k-shot list of the examples file (icl_examples_<tag>.json): [(cur_png, tgt_png, prompt, label)].
    The file must hold at least k examples (fix_conditions.py)."""
    spec = json.loads(EXAMPLES.read_text())
    rows = read_jsonl(PAIRS / "train.jsonl")
    assert len(spec["examples"]) >= k, f"{EXAMPLES.name} has {len(spec['examples'])} examples, {k} needed"
    out = []
    for e in spec["examples"][:k]:
        row = rows[e["train_index"]]
        assert row["meta"] == e["meta"], f"train.jsonl changed under {EXAMPLES.name} at {e}"
        out.append((*sample_inputs(row), label(row)))
    return out


# ---- closed models (OpenRouter) -------------------------------------------------------------
def _image(png):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode(),
                                               "detail": "high"}}


def _user(cur_png, tgt_png, prompt):
    """the order our model sees: image 1 (current), image 2 (target), then the text"""
    return [_image(cur_png), _image(tgt_png), {"type": "text", "text": prompt}]


def _usage(u):
    d = u.model_dump() if u is not None else {}
    pt, ct = d.get("prompt_tokens_details") or {}, d.get("completion_tokens_details") or {}
    return {"prompt_tokens": d.get("prompt_tokens"), "completion_tokens": d.get("completion_tokens"),
            "reasoning_tokens": ct.get("reasoning_tokens"), "cached_tokens": pt.get("cached_tokens"),
            "image_tokens": pt.get("image_tokens"), "cost": d.get("cost"), "raw": d}


def _retryable(e):
    s = f"{type(e).__name__} {e}".lower()
    return any(k in s for k in ("429", "rate", "timeout", "timed out", "connection", "500", "502", "503",
                                "504", "overloaded", "no choices"))


class ClosedPlanner:
    parallel = True

    def __init__(self, model, level="0shot", schema=SCHEMA_DEFAULT, reasoning=CLOSED_REASONING, max_tokens=CLOSED_MAX_TOKENS,
                 temperature="default", name=None):
        assert level in LEVELS, level
        self.model, self.level, self.schema = model, level, schema
        self.spec = MODELS[model]
        self.reasoning, self.max_tokens, self.temperature = reasoning, max_tokens, temperature
        self.effort = (next(v for k, v in MAX_EFFORT.items() if self.spec["id"].startswith(k)) if reasoning == "max"
                       else reasoning)                       # "max" = this provider's highest level
        self.name = name or (model + (f"-reasoning-{reasoning}" if reasoning != CLOSED_REASONING else "")
                             + (f"-max{max_tokens}" if max_tokens != CLOSED_MAX_TOKENS else "")
                             + ("-t0" if temperature == "zero" else ""))
        k = int(level[:-4])
        self.examples = load_examples(k) if k else []
        facts = facts_text(schema)
        self.conditions = {"facts_sha": sha("\n".join(facts))[:12] if facts else None,
                           "examples_sha": sha(json.dumps([e[3] for e in self.examples]) +
                                               "".join(sha(e[0]) + sha(e[1]) for e in self.examples))[:12]
                           if self.examples else None}
        self._client = None

    def messages(self, cur_png, tgt_png, prompt):
        """2026-10-20: the invariant part of the training prompt (task, setting, image convention, process facts,
        output schema) goes ONCE as the system message; every example turn and the query turn carry only the two
        images and the per-request text (z ranges, history). Our model sees the same text concatenated in one
        user message (pairs.build_prompt), so the content is identical and nothing is repeated k times."""
        from pairs import split_prompt                       # newsheet_pipeline/pairs.py (sys.path set above)
        system, user = split_prompt(prompt)
        msgs = [{"role": "system", "content": system}]
        for cur, tgt, text, lab in self.examples:            # k-shot: training samples as earlier turns
            ex_system, ex_user = split_prompt(text)
            assert ex_system == system, "a k-shot example was built with another system text than the query"
            msgs += [{"role": "user", "content": _user(cur, tgt, ex_user)}, {"role": "assistant", "content": lab}]
        return msgs + [{"role": "user", "content": _user(cur_png, tgt_png, user)}]

    def request(self, cur_png, tgt_png, prompt, sample=False):
        req = {"model": self.spec["id"], "messages": self.messages(cur_png, tgt_png, prompt),
               "max_tokens": self.max_tokens, "seed": int(sample)}
        if self.temperature == "zero" and self.spec["temperature"]:
            req["temperature"] = SAMPLE_TEMPERATURE if sample else 0.0
        extra = {"usage": {"include": True}}                  # OpenRouter reports the billed cost
        if self.reasoning != "default":
            extra["reasoning"] = {"effort": self.effort}
        return req, extra

    def plan(self, cur_png, tgt_png, cz, tz, history, sample=False, tag=None):
        return self.plan_prompt(cur_png, tgt_png, build_prompt(cz, tz, history, schema=self.schema), sample, tag)

    def plan_prompt(self, cur_png, tgt_png, prompt, sample=False, tag=None):
        req, extra = self.request(cur_png, tgt_png, prompt, sample)
        key = sha(json.dumps([req, extra, int(sample)], sort_keys=True))
        meta = {"system": self.name, "model": self.spec["id"], "level": self.level, "tag": tag,
                "attempt": int(sample), "inputs_sha": inputs_sha(cur_png, tgt_png, prompt),
                "n_images": 2 * (len(self.examples) + 1), "conditions": self.conditions,
                "params": {k: v for k, v in req.items() if k != "messages"} | {"extra": extra}}
        rec = _cached(self.name, self.level, key, lambda: self._call(req, extra), meta)
        return {**rec, "plan": parse_plan(rec["raw"])[1]}

    def _call(self, req, extra, retries=4, base_delay=20.0):
        from openai import OpenAI
        if self._client is None:
            api_key = KEY_FILE.read_text().strip() if KEY_FILE.exists() else os.environ["OPENROUTER_API_KEY"]
            self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, timeout=900)
        for attempt in range(retries + 1):
            try:
                t0 = time.time()
                r = self._client.chat.completions.create(**req, extra_body=extra)
                if not r.choices:
                    raise RuntimeError(f"no choices in response: {getattr(r, 'error', None)}")
                msg = r.choices[0].message
                return {"raw": msg.content or "", "reasoning": getattr(msg, "reasoning", None) or "",
                        "finish": r.choices[0].finish_reason, "provider": getattr(r, "provider", None),
                        "secs": round(time.time() - t0, 2), "usage": _usage(r.usage)}
            except Exception as e:
                if attempt == retries or not _retryable(e):
                    raise
                wait = base_delay * 2 ** attempt
                print(f"    retry {attempt + 1}/{retries} in {wait:.0f}s: {str(e)[:120]}", flush=True)
                time.sleep(wait)


# ---- our checkpoint ---------------------------------------------------------------------------
class OursPlanner:
    """unsloth 4-bit Qwen3-VL + LoRA; the generation code of mpc_rollout.generate / eval_symbolic.py
    (two image slots then the text, greedy or the sampled retry at 0.7, 512 new tokens). The model
    loads on the first call that is not cached."""
    parallel = False
    level = OURS_LEVEL
    BASE = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"

    def __init__(self, ckpt, schema=SCHEMA_DEFAULT, name=None):
        self.ckpt, self.schema = str(Path(ckpt).resolve()), schema
        self.name = name or "ours-" + Path(ckpt).name.replace("checkpoint-", "ck")
        w = Path(ckpt) / "adapter_model.safetensors"
        self.weights_sha = sha(w.read_bytes())[:16] if w.exists() else self.ckpt
        self._model = self._proc = None

    def plan(self, cur_png, tgt_png, cz, tz, history, sample=False, tag=None):
        return self.plan_prompt(cur_png, tgt_png, build_prompt(cz, tz, history, schema=self.schema), sample, tag)

    def plan_prompt(self, cur_png, tgt_png, prompt, sample=False, tag=None):
        key = sha(json.dumps([self.weights_sha, sha(cur_png), sha(tgt_png), sha(prompt), int(sample), MAX_TOKENS]))
        meta = {"system": self.name, "model": self.ckpt, "level": self.level, "tag": tag, "attempt": int(sample),
                "inputs_sha": inputs_sha(cur_png, tgt_png, prompt), "weights_sha": self.weights_sha}
        rec = _cached(self.name, self.level, key, lambda: self._generate(cur_png, tgt_png, prompt, sample), meta)
        return {**rec, "plan": parse_plan(rec["raw"])[1]}

    def _generate(self, cur_png, tgt_png, prompt, sample):
        import io
        from PIL import Image
        if self._model is None:
            from unsloth import FastVisionModel
            from transformers import AutoProcessor
            self._model, _ = FastVisionModel.from_pretrained(self.ckpt, load_in_4bit=True)
            FastVisionModel.for_inference(self._model)
            self._proc = AutoProcessor.from_pretrained(self.BASE)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in (cur_png, tgt_png)]
        batch = self._proc(text=[chat], images=[imgs], return_tensors="pt").to(self._model.device)
        kw = dict(do_sample=True, temperature=SAMPLE_TEMPERATURE) if sample else dict(do_sample=False)
        t0 = time.time()
        o = self._model.generate(**batch, max_new_tokens=MAX_TOKENS, pad_token_id=self._proc.tokenizer.eos_token_id, **kw)
        n_in = int(batch["input_ids"].shape[1])
        n_out = int(o.shape[1]) - n_in
        return {"raw": self._proc.tokenizer.decode(o[0][n_in:], skip_special_tokens=True),
                "finish": "length" if n_out >= MAX_TOKENS else "stop", "provider": "local",
                "secs": round(time.time() - t0, 2), "usage": {"prompt_tokens": n_in, "completion_tokens": n_out}}


# ---- the label oracle (free) ------------------------------------------------------------------
class LabelPlanner:
    """Answers every sample with its own label from val.jsonl (or train.jsonl), looked up by the input
    fingerprint (both images and the prompt). Same record and cache path as the other planners
    (cache/label/oracle/), so run_openloop -> run_execute -> make_report can be exercised without a model;
    the executed shapes must match the ground-truth replays (--controls) up to simulator noise."""
    parallel = True
    level = LABEL_LEVEL
    schema = SCHEMA_DEFAULT

    def __init__(self, name="label"):
        self.name = name
        self._labels, self._lock = None, threading.Lock()

    def _lookup(self, fp):
        with self._lock:                                     # built once, by the first of the worker threads
            if self._labels is None:
                table = {}
                for split in ("val", "train"):
                    for row in read_jsonl(PAIRS / f"{split}.jsonl"):
                        table[json.dumps(inputs_sha(*sample_inputs(row)), sort_keys=True)] = label(row)
                self._labels = table
        try:
            return self._labels[json.dumps(fp, sort_keys=True)]
        except KeyError:
            raise KeyError(f"no sample of {PAIRS} has these inputs; the label planner only answers its own data") from None

    def plan(self, cur_png, tgt_png, cz, tz, history, sample=False, tag=None):
        return self.plan_prompt(cur_png, tgt_png, build_prompt(cz, tz, history, schema=self.schema), sample, tag)

    def plan_prompt(self, cur_png, tgt_png, prompt, sample=False, tag=None):
        fp = inputs_sha(cur_png, tgt_png, prompt)
        lab = self._lookup(fp)
        key = sha(json.dumps(["label", fp, lab, int(sample)], sort_keys=True))
        meta = {"system": self.name, "model": "label", "level": self.level, "tag": tag, "attempt": int(sample),
                "inputs_sha": fp}
        rec = _cached(self.name, self.level, key,
                      lambda: {"raw": lab, "finish": "stop", "provider": "label", "secs": 0.0, "usage": {}}, meta)
        return {**rec, "plan": parse_plan(rec["raw"])[1]}


# ---- command line ---------------------------------------------------------------------------
def add_planner_args(ap):
    ap.add_argument("--model", required=True, help="closed model (%s), 'ours' or 'label' (free oracle)" % ", ".join(MODELS))
    ap.add_argument("--level", default="0shot", choices=LEVELS,
                    help="closed models: number of solved training samples in the request (ignored for ours / label)")
    ap.add_argument("--ckpt", help="checkpoint directory, with --model ours")
    ap.add_argument("--schema", default=SCHEMA_DEFAULT, choices=SCHEMAS,
                    help="prompt schema of plan() (closed loop, robustness); must be the one the data and the checkpoint "
                         "use (new pipeline: v5facts = pairs.PROMPT_V5 + <data>/facts.json; v5 without facts; "
                         "pairs_pilot / ckpt-1984: v1)")
    ap.add_argument("--reasoning", default=CLOSED_REASONING,
                    help="reasoning effort sent to every closed model: max = each provider's highest level "
                         "(xhigh / max / high), or one of none/minimal/low/medium/high/xhigh for all, "
                         "or 'default' to send nothing and take the provider default")
    ap.add_argument("--max-tokens", type=int, default=CLOSED_MAX_TOKENS,
                    help="output budget of closed models, hidden reasoning included (ours: always 512)")
    ap.add_argument("--temperature", default="default", choices=["default", "zero"],
                    help="closed models: provider default for both attempts, or 0 then 0.7 where accepted")
    ap.add_argument("--name", help="system name in result and cache paths (default: derived)")
    add_data_args(ap)


def make_planner(a):
    configure(a)
    if a.model == "ours":
        assert a.ckpt, "--model ours needs --ckpt"
        return OursPlanner(a.ckpt, a.schema, a.name)
    if a.model == "label":
        return LabelPlanner(a.name or "label")
    return ClosedPlanner(a.model, a.level, a.schema, a.reasoning, a.max_tokens, a.temperature, a.name)
