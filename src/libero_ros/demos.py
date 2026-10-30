import json
import os

import numpy as np

from .control import ee_pose_base

CAMERAS = ("agentview", "robot0_eye_in_hand")


class Recorder:
    def __init__(self, env, size=128):
        self.env, self.size = env, size
        self.active = False
        self.clear()

    def clear(self):
        self.agent, self.wrist, self.state, self.action = [], [], [], []
        self._pack = None
        self.active = False

    def observe(self, obs):
        sim = self.env.sim
        imgs = []
        for cam in CAMERAS:
            rgb = sim.render(width=self.size, height=self.size, camera_name=cam)[::-1]
            imgs.append(np.ascontiguousarray(rgb))
        p, q = ee_pose_base(self.env, obs)
        st = np.concatenate([p, q, np.array(obs["robot0_gripper_qpos"]), np.array(obs["robot0_joint_pos"])]).astype(np.float32)
        return imgs[0], imgs[1], st

    def start(self):
        from .control import current_obs
        self.clear()
        self.active = True
        self._pack = self.observe(current_obs(self.env))

    def on_step(self, env, obs, action):
        if not self.active:
            return
        a, w, s = self._pack
        self.agent.append(a)
        self.wrist.append(w)
        self.state.append(s)
        self.action.append(np.asarray(action, dtype=np.float32))
        self._pack = self.observe(obs)

    def __len__(self):
        return len(self.action)

    def save(self, root, dataset, episode_id, meta):
        label = meta["label"]
        sub = "success" if meta["success"] else os.path.join("failed", label)
        d = os.path.join(root, dataset, sub)
        os.makedirs(d, exist_ok=True)
        meta = dict(meta, episode_id=int(episode_id), n_steps=len(self))
        path = os.path.join(d, f"ep_{episode_id:06d}.npz")
        if len(self) > 0:
            np.savez_compressed(path, agentview=np.stack(self.agent), wrist=np.stack(self.wrist),
                                state=np.stack(self.state), action=np.stack(self.action),
                                meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
        else:
            path = path.replace(".npz", ".json")
            with open(path, "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
        with open(os.path.join(root, dataset, "index.jsonl"), "a") as f:
            f.write(json.dumps({"path": os.path.relpath(path, os.path.join(root, dataset)), **{
                k: meta[k] for k in ("episode_id", "object", "label", "success", "n_steps", "seed", "delta_xy", "layout", "mode")
                if k in meta}}, ensure_ascii=False) + "\n")
        return path
