"""Robot-PC side of the final controller (ew_loop/final_controller.py runs on ideas2): sends the cleaned field, the target, the
executed history and the run id to `serve-fields --controller final`, returns its recommendation. No research code needed here."""
import base64
import json
import os
import tempfile
import urllib.request


def _npz_b64(f):
    if f is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as t:
        name = t.name
    f.save(name)
    b = open(name, "rb").read(); os.remove(name)
    return base64.b64encode(b).decode()


class FinalControllerClient:
    def __init__(self, url, timeout=1800):
        self.url, self.timeout = url.rstrip("/"), timeout

    def health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
            return json.loads(r.read())

    def step(self, run_id, rnd, cur, tgt, history, executed_block=None, prev=None, max_passes=12):
        body = json.dumps({"run_id": run_id, "round": int(rnd), "cur_npz": _npz_b64(cur), "tgt_npz": _npz_b64(tgt),
                           "prev_npz": _npz_b64(prev), "history": history, "executed_block": executed_block,
                           "max_passes": int(max_passes)}, default=float).encode()
        req = urllib.request.Request(self.url + "/final_step", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"final controller service error {e.code}: {e.read()[:500]!r}") from None
