import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from approach import approach
from expert import grasp_and_lift
from provided import OBJECT_TABLE, PREGRASP_H, make_env, make_recorder, object_position, reset_scene

OBJECTS = list(OBJECT_TABLE)
DELTA_MAX = 0.03
OUT_ROOT = "/workspace/outputs/demos"


def collect_one(env, rec, seed, obj, delta_xy):
 

    p0 = object_position(env, obj)
    pre = [p0[0] + delta_xy[0], p0[1] + delta_xy[1], OBJECT_TABLE[obj]['center_z'] + PREGRASP_H]

    result_approach = approach(env,pre)
  

    meta = dict(success="", label = "",steps = "", object = obj, seed = seed, delta_xy = delta_xy,layout="original")
    if result_approach["success"]  == False:
        meta["success"] = False
        meta["label"] = "pregrasp_failed"
        meta["steps"] = result_approach["steps"]
        rec.clear()
        return meta

    rec.start()
    result_grasp = grasp_and_lift(env, obj, on_step=rec.on_step) 
    meta["success"] = result_grasp["success"]
    meta["label"] = result_grasp["label"]
    meta["steps"] = result_grasp["steps"]
    return(meta)


def main():
    name, n = sys.argv[1], int(sys.argv[2])
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    env, rec = make_env(), None
    rec = make_recorder(env)
    rng = np.random.default_rng(seed0)
    os.makedirs(os.path.join(OUT_ROOT, name), exist_ok=True)
    json.dump({"n": n, "seed0": seed0, "objects": OBJECTS, "delta_max_m": DELTA_MAX}, open(os.path.join(OUT_ROOT, name, "config.json"), "w"))
    n_ok = 0

    for ep in range(n):
        seed = seed0 + ep
        obj = OBJECTS[ep % len(OBJECTS)]
        r_, th = DELTA_MAX * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        delta = (float(r_ * np.cos(th)), float(r_ * np.sin(th)))
        reset_scene(env, seed)
        res = collect_one(env, rec, seed, obj, delta)

        rec.save(OUT_ROOT,name,ep,res)
        n_ok += bool(res["success"])
        print(f"ep{ep:03d} seed{seed} {obj:10s} delta=({delta[0] * 100:+.1f},{delta[1] * 100:+.1f})cm  {res['label']:14s}  成功 {n_ok}/{ep + 1}", flush=True)
    print(f"完了: 成功 {n_ok}/{n}  保存先 {OUT_ROOT}/{name}")


if __name__ == "__main__":
    main()
