"""predict(): current + target height fields + history -> model plan.

Backends
  LocalPlanner(ckpt)   the MPC checkpoint (unsloth 4-bit Qwen3-VL LoRA) on this machine's GPU
  HttpPlanner(url)     the same model behind `python -m ew_loop.cli serve` on another machine
Both take the two rendered PNGs (bytes), the two z ranges and the history list, and return
{"raw": generated text, "plan": parsed plan list or None, "secs": generation time}.

Prompt, image rendering and history schema are exactly those of experiments/mpc_end2end
(copied in model_io.py, checked by tests/test_model_io.py), so a plan produced here is what
the offline MPC evaluation produced. ckpt-1984 was trained with
prompt schema v1 (the mpc_rollout default) — keep DEFAULT_SCHEMA in sync with the checkpoint.
"""
import base64
import io
import json
import os
import re
import sys
import time
import urllib.request

from . import frames
from .model_io import build_prompt, entry_desc, read_entry, wa_desc, zz_desc

BASE_MODEL = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
DEFAULT_SCHEMA = "v1"
MAX_NEW_TOKENS = 512


# ---- history / plan helpers (mirror mpc_rollout.py; copied so the loop does not import the
# ---- offline rollout script and its simulator dependencies) ----------------------------
def ops_to_history_json(ops, schema=DEFAULT_SCHEMA):
    """executed ops (wa dicts with n) -> history list in the prompt schema (consecutive
    identical WA+zigzag entries are merged, hits_done accumulated)."""
    out = []
    for wa in ops:
        if out and out[-1].get("wa") == wa_desc(wa) and out[-1].get("zigzag", wa_desc(wa)) == zz_desc(wa):
            out[-1]["hits_done"] += wa["n"]
        elif schema == "v1" and out and out[-1]["wa"] == {**wa_desc(wa), **zz_desc(wa)}:
            out[-1]["hits_done"] += wa["n"]
        else:
            out.append(entry_desc(wa, wa["n"], "hits_done", schema))
    return out


def plan_take_cycles(plan_was, n):
    """first n cycles of a plan (may cut an entry)."""
    taken, left = [], n
    for wa in plan_was:
        if left <= 0:
            break
        k = min(wa["n"], left)
        taken.append({**wa, "n": k}); left -= k
    return taken


def parse_plan(text):
    """generated text -> (plan list of wa dicts, raw plan json) or (None, None)."""
    try:
        plan = json.loads(re.search(r"\{.*\}", text, re.S).group(0))["plan"]
        was = [read_entry(e) for e in plan]
        if not was:
            return None, None
        return was, plan
    except Exception:
        return None, None


# ---- backends ---------------------------------------------------------------------------
class LocalPlanner:
    def __init__(self, ckpt, schema=DEFAULT_SCHEMA):
        from unsloth import FastVisionModel
        from transformers import AutoProcessor
        t0 = time.time()
        self.ckpt, self.schema = ckpt, schema
        self.model, _ = FastVisionModel.from_pretrained(ckpt, load_in_4bit=True)
        FastVisionModel.for_inference(self.model)
        self.proc = AutoProcessor.from_pretrained(BASE_MODEL)
        self.load_secs = time.time() - t0

    def plan(self, cur_png, tgt_png, cz, tz, history, sample=False):
        from PIL import Image
        prompt = build_prompt(cz, tz, history, schema=self.schema)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                             {"type": "text", "text": prompt}]}]
        chat = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs = [Image.open(io.BytesIO(cur_png)).convert("RGB"), Image.open(io.BytesIO(tgt_png)).convert("RGB")]
        batch = self.proc(text=[chat], images=[imgs], return_tensors="pt").to(self.model.device)
        kw = dict(do_sample=True, temperature=0.7) if sample else dict(do_sample=False)
        t0 = time.time()
        o = self.model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS,
                                pad_token_id=self.proc.tokenizer.eos_token_id, **kw)
        raw = self.proc.tokenizer.decode(o[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
        _, plan = parse_plan(raw)
        return {"raw": raw, "plan": plan, "secs": round(time.time() - t0, 2), "prompt": prompt}


class HttpPlanner:
    def __init__(self, url, timeout=600):
        self.url, self.timeout = url.rstrip("/"), timeout

    def health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
            return json.loads(r.read())

    def plan(self, cur_png, tgt_png, cz, tz, history, sample=False):
        body = json.dumps({"cur_png": base64.b64encode(cur_png).decode(), "tgt_png": base64.b64encode(tgt_png).decode(),
                           "cz": [float(cz[0]), float(cz[1])], "tz": [float(tz[0]), float(tz[1])],
                           "history": history, "sample": bool(sample)}).encode()
        req = urllib.request.Request(self.url + "/predict", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())


class FixedPlanner:
    """Answers every request with the same plan: runs the loop without a model, e.g. to test the
    robot and the cameras with a known zigzag. plan_json: {"plan": [...]} in the model schema."""
    def __init__(self, plan_json):
        self.text = json.dumps(json.load(open(plan_json)) if isinstance(plan_json, str) else plan_json)

    def plan(self, cur_png, tgt_png, cz, tz, history, sample=False):
        return {"raw": self.text, "plan": json.loads(self.text)["plan"], "secs": 0.0}


def serve(ckpt, host="0.0.0.0", port=8765, schema=DEFAULT_SCHEMA, planner=None):
    """Minimal HTTP front for LocalPlanner (stdlib only). One request at a time (GPU)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    planner = planner or LocalPlanner(ckpt, schema)
    lock = threading.Lock()
    print(f"[serve] {ckpt} loaded in {planner.load_secs:.0f}s; schema {schema}; http://{host}:{port}", flush=True)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "ckpt": ckpt, "schema": schema})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, {"error": "unknown path"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                q = json.loads(self.rfile.read(n))
                with lock:
                    out = planner.plan(base64.b64decode(q["cur_png"]), base64.b64decode(q["tgt_png"]),
                                       q["cz"], q["tz"], q.get("history", []), q.get("sample", False))
                self._send(200, out)
            except Exception as e:
                self._send(500, {"error": repr(e)})

        def log_message(self, fmt, *args):
            print("[serve]", fmt % args, flush=True)

    ThreadingHTTPServer((host, port), H).serve_forever()


# ---- profile-based planners (2026-10-28): the model profile owns rendering and prompt ----------
class ModelPlanner:
    """a checkpoint on a local GPU, with a model profile (model_newsheet.NewsheetModel) that renders
    the fields and builds the prompt. plan_fields() is the loop-facing call."""

    def __init__(self, ckpt, profile):
        from unsloth import FastVisionModel
        from transformers import AutoProcessor
        t0 = time.time()
        self.ckpt, self.profile = ckpt, profile
        self.model, _ = FastVisionModel.from_pretrained(ckpt, load_in_4bit=True)
        FastVisionModel.for_inference(self.model)
        self.proc = AutoProcessor.from_pretrained(BASE_MODEL)
        self.load_secs = time.time() - t0

    def generate(self, cur_png, tgt_png, prompt, sample=False):
        from PIL import Image
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
        chat = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs = [Image.open(io.BytesIO(cur_png)).convert("RGB"), Image.open(io.BytesIO(tgt_png)).convert("RGB")]
        batch = self.proc(text=[chat], images=[imgs], return_tensors="pt").to(self.model.device)
        kw = dict(do_sample=True, temperature=0.7) if sample else dict(do_sample=False)
        t0 = time.time()
        o = self.model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=self.proc.tokenizer.eos_token_id, **kw)
        raw = self.proc.tokenizer.decode(o[0][batch["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"raw": raw, "secs": round(time.time() - t0, 2)}

    def plan_fields(self, cur_field, tgt_field, history, outdir):
        return _plan_fields_with(self.generate, self.profile, cur_field, tgt_field, history, outdir)


def _plan_fields_with(generate, profile, cur_field, tgt_field, history, outdir):
    """render -> prompt -> generate (greedy, then one sampled retry if not executable) -> parse."""
    os.makedirs(outdir, exist_ok=True)
    cur_png, tgt_png = os.path.join(outdir, "cur.png"), os.path.join(outdir, "tgt.png")
    profile.render(cur_field, cur_png); profile.render(tgt_field, tgt_png)
    cz, tz = cur_field.zrange(), tgt_field.zrange()
    prompt = profile.build_prompt(cz, tz, history)
    open(os.path.join(outdir, "prompt.txt"), "w").write(prompt)
    cb, tb = open(cur_png, "rb").read(), open(tgt_png, "rb").read()
    attempts = []
    for sample in (False, True):
        out = generate(cb, tb, prompt, sample=sample)
        was, plan = parse_plan(out.get("raw", ""))
        attempts.append({**out, "sampled": sample, "parsed": was is not None})
        if was is not None and profile.executable(plan):
            break
    res = {"was": was, "plan": plan, "raw": out.get("raw", ""), "secs": out.get("secs"), "sampled": sample,
           "attempts": attempts, "cz": cz, "tz": tz, "history": history, "profile": profile.name}
    json.dump(res, open(os.path.join(outdir, "response.json"), "w"), indent=1)
    return res


class HttpFieldsPlanner:
    """the same over HTTP: the robot PC sends the two height fields (npz) and the history; the server
    (serve_fields) renders, prompts and runs the model, and returns the answer plus the images."""

    def __init__(self, url, profile_name="newsheet-v5facts", timeout=900):
        self.url, self.profile_name, self.timeout = url.rstrip("/"), profile_name, timeout

    def health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
            return json.loads(r.read())

    def plan_fields(self, cur_field, tgt_field, history, outdir):
        import tempfile
        os.makedirs(outdir, exist_ok=True)
        def npz_bytes(f):
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as t:
                name = t.name
            f.save(name); b = open(name, "rb").read(); os.remove(name); return b
        body = json.dumps({"cur_npz": base64.b64encode(npz_bytes(cur_field)).decode(),
                           "tgt_npz": base64.b64encode(npz_bytes(tgt_field)).decode(), "history": history}).encode()
        req = urllib.request.Request(self.url + "/predict_fields", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            res = json.loads(r.read())
        for k in ("cur_png", "tgt_png"):
            if k in res:
                open(os.path.join(outdir, k.replace("_png", ".png")), "wb").write(base64.b64decode(res.pop(k)))
        if "prompt" in res:
            open(os.path.join(outdir, "prompt.txt"), "w").write(res["prompt"])
        json.dump(res, open(os.path.join(outdir, "response.json"), "w"), indent=1)
        return res


class FixedFieldsPlanner:
    """a fixed plan instead of a model (tests the measurement, controller and robot side alone). No prompt is
    built, so it also runs on a robot-PC copy that has no model code (planners.py) at all."""
    def __init__(self, plan_json, profile):
        self.text = json.dumps(json.load(open(plan_json)) if isinstance(plan_json, str) else plan_json)
        self.profile = profile

    def plan_fields(self, cur_field, tgt_field, history, outdir):
        os.makedirs(outdir, exist_ok=True)
        self.profile.render(cur_field, os.path.join(outdir, "cur.png")); self.profile.render(tgt_field, os.path.join(outdir, "tgt.png"))
        was, plan = parse_plan(self.text)
        res = {"was": was, "plan": plan, "raw": self.text, "secs": 0.0, "sampled": False, "fixed": True,
               "cz": cur_field.zrange(), "tz": tgt_field.zrange(), "history": history, "profile": self.profile.name}
        json.dump(res, open(os.path.join(outdir, "response.json"), "w"), indent=1)
        return res


def serve_fields(ckpt, profile, host="0.0.0.0", port=8765):
    """HTTP front for ModelPlanner: POST /predict_fields {cur_npz, tgt_npz (base64), history} ->
    {raw, plan, was, secs, cur_png, tgt_png (base64), prompt}. One request at a time (GPU)."""
    import tempfile, threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from .frames import HeightField
    planner = ModelPlanner(ckpt, profile)
    lock = threading.Lock()
    print(f"[serve] {ckpt} loaded in {planner.load_secs:.0f}s; profile {profile.name}; http://{host}:{port}", flush=True)

    def field_from_b64(b64):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as t:
            t.write(base64.b64decode(b64)); name = t.name
        f = HeightField.load(name); os.remove(name); return f

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"ok": True, "ckpt": ckpt, "profile": profile.name}) if self.path == "/health" else self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/predict_fields":
                return self._send(404, {"error": "unknown path"})
            try:
                q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                cur, tgt = field_from_b64(q["cur_npz"]), field_from_b64(q["tgt_npz"])
                with lock, tempfile.TemporaryDirectory() as d:
                    res = planner.plan_fields(cur, tgt, q.get("history", []), d)
                    res["cur_png"] = base64.b64encode(open(os.path.join(d, "cur.png"), "rb").read()).decode()
                    res["tgt_png"] = base64.b64encode(open(os.path.join(d, "tgt.png"), "rb").read()).decode()
                    res["prompt"] = open(os.path.join(d, "prompt.txt")).read()
                self._send(200, res)
            except Exception as e:
                self._send(500, {"error": repr(e)})

        def log_message(self, fmt, *args):
            print("[serve]", fmt % args, flush=True)

    ThreadingHTTPServer((host, port), H).serve_forever()


# ---- the loop-facing call ---------------------------------------------------------------
def predict(cur_field, tgt_field, history_ops, planner, outdir, schema=DEFAULT_SCHEMA, retry_sample=True):
    """Render the two fields in the model frame, ask the planner, parse. Everything is written
    to outdir (cur.png, tgt.png, prompt.txt, response.json). Returns a dict with
    'was' (list of wa dicts with n, or None), 'plan' (raw json), 'raw', 'secs', 'history'."""
    os.makedirs(outdir, exist_ok=True)
    cur_png, tgt_png = os.path.join(outdir, "cur.png"), os.path.join(outdir, "tgt.png")
    frames.field_png(cur_field, cur_png)
    frames.field_png(tgt_field, tgt_png)
    cz, tz = cur_field.zrange(), tgt_field.zrange()
    history = ops_to_history_json(history_ops, schema)
    with open(os.path.join(outdir, "prompt.txt"), "w") as f:
        f.write(build_prompt(cz, tz, history, schema=schema))
    cb, tb = open(cur_png, "rb").read(), open(tgt_png, "rb").read()
    out = planner.plan(cb, tb, cz, tz, history, sample=False)
    was, plan = parse_plan(out.get("raw", ""))
    if was is None and retry_sample:
        out2 = planner.plan(cb, tb, cz, tz, history, sample=True)
        was, plan = parse_plan(out2.get("raw", ""))
        out = {**out2, "greedy_raw": out.get("raw", ""), "sampled": True}
    res = {"was": was, "plan": plan, "raw": out.get("raw", ""), "secs": out.get("secs"),
           "sampled": out.get("sampled", False), "cz": cz, "tz": tz, "history": history}
    with open(os.path.join(outdir, "response.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res
