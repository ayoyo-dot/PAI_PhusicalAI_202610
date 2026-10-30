import argparse
import http.server
import io
import json
import sys
import threading

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


class Model:
    def __init__(self, ckpt, threads, n_action_steps=None, temporal_ensemble=None):
        torch.set_num_threads(threads)
        cfg = PreTrainedConfig.from_pretrained(ckpt)
        if n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        if temporal_ensemble is not None:
            cfg.temporal_ensemble_coeff = temporal_ensemble
            cfg.n_action_steps = 1
        self.policy = ACTPolicy.from_pretrained(ckpt, config=cfg)
        print(f"n_action_steps={cfg.n_action_steps} chunk_size={cfg.chunk_size} "
              f"temporal_ensemble_coeff={cfg.temporal_ensemble_coeff}", flush=True)
        self.policy.eval()
        st = json.load(open(f"{ckpt}/stats.json"))
        self.s_mean, self.s_std = np.array(st["state_mean"], np.float32), np.array(st["state_std"], np.float32)
        self.mask_idx = list(st.get("mask_state_idx", []))
        self.a_mean, self.a_std = np.array(st["action_mean"], np.float32), np.array(st["action_std"], np.float32)
        self.lock = threading.Lock()
        self.n_calls = 0

    @staticmethod
    def _img(x):
        x = x.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return torch.from_numpy((x - IMNET_MEAN) / IMNET_STD)

    def reset(self):
        with self.lock:
            self.policy.reset()

    def act(self, agentview, wrist, state):
        s = ((state - self.s_mean) / self.s_std).astype(np.float32)
        if self.mask_idx:
            s[self.mask_idx] = 0.0
        batch = {"observation.images.agentview": self._img(agentview), "observation.images.wrist": self._img(wrist),
                 "observation.state": torch.from_numpy(s[None])}
        with self.lock, torch.no_grad():
            a = self.policy.select_action(batch)[0].numpy()
            self.n_calls += 1
        return (a * self.a_std + self.a_mean).astype(np.float32)


def make_handler(model):
    class H(http.server.BaseHTTPRequestHandler):
        def _send(self, body, code=200):
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            if self.path == "/reset":
                model.reset()
                return self._send(b"ok")
            if self.path == "/act":
                z = np.load(io.BytesIO(body), allow_pickle=False)
                a = model.act(z["agentview"], z["wrist"], z["state"])
                buf = io.BytesIO()
                np.savez(buf, action=a)
                return self._send(buf.getvalue())
            self._send(b"not found", 404)

        def do_GET(self):
            self._send(json.dumps({"status": "ok", "calls": model.n_calls}).encode())

        def log_message(self, *a):
            pass
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--temporal-ensemble", type=float, default=None)
    a = ap.parse_args()
    m = Model(a.ckpt, a.threads, a.n_action_steps, a.temporal_ensemble)
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(m))
    print(f"serving {a.ckpt} on :{a.port}", flush=True)
    srv.serve_forever()
