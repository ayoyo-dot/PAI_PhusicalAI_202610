import argparse
import io
import json
import math
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from provided import GRIP_YAW, OBJECT_TABLE, PREGRASP_H, ee_position, make_env, make_recorder, object_position, reset_scene

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("name")
ap.add_argument("--n", type=int, default=6, help="物体ごとの回数")
ap.add_argument("--seed", type=int, default=7000)
ap.add_argument("--delta", type=float, default=3.0, help="接近の水平のずれの最大(cm)")
ap.add_argument("--video", type=int, default=1, help="物体ごとに、最初の何回を動画にするか(0 で無し)")
ap.add_argument("--only", default="", help="指定した seed だけを実行する(例 7026,7027)。ずれの乱数は、同じ並びで引くので、全体の評価と同じ条件を再現でき、この seed は、全部、動画にする")
ap.add_argument("--max-steps", type=int, default=150)
ap.add_argument("--threshold", type=float, default=None, help="自分の HandoffDetector の、持ち上げのしきい値(m)を、この評価だけ上書きする(診断用。handoff.py は変えない)")
a = ap.parse_args()
url = a.url


def post(path, payload=b""):
    with urllib.request.urlopen(urllib.request.Request(url + path, data=payload, method="POST"), timeout=120) as r:
        return r.read()


def act(ag, wr, st):
    buf = io.BytesIO()
    np.savez(buf, agentview=ag, wrist=wr, state=st)
    return np.load(io.BytesIO(post("/act", buf.getvalue())), allow_pickle=False)["action"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


from approach import approach as do_approach
from handoff import HandoffDetector, run_policy_until_handoff as _my_run


def do_policy(env, rec):
    det = HandoffDetector()
    if a.threshold is not None:
        det.threshold = a.threshold
    return _my_run(env, rec.observe, act, det, max_steps=a.max_steps)


env = make_env()
rec = make_recorder(env)
rng = np.random.default_rng(a.seed)
os.makedirs("/workspace/outputs/videos/my_eval", exist_ok=True)
os.makedirs("/workspace/outputs/eval/source", exist_ok=True)


class Cap:

    def __init__(self, env_):
        self.env, self.frames, self._orig = env_, [], env_.step

    def __enter__(self):
        def stepped(action):
            out = self._orig(action)
            sim = self.env.sim
            ag = np.ascontiguousarray(sim.render(width=256, height=256, camera_name="agentview")[::-1])
            wr = cv2.resize(np.ascontiguousarray(sim.render(width=128, height=128, camera_name="robot0_eye_in_hand")[::-1]), (256, 256))
            f = cv2.cvtColor(np.concatenate([ag, wr], axis=1), cv2.COLOR_RGB2BGR)
            cv2.putText(f, f"t={len(self.frames)} grip={action[6]:+.0f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            self.frames.append(f)
            return out
        self.env.step = stepped
        return self

    def __exit__(self, *e):
        self.env.step = self._orig


class ZTrack:

    def __init__(self, env_):
        self.env, self._orig, self.z = env_, env_.step, []

    def __enter__(self):
        def stepped(action):
            out = self._orig(action)
            self.z.append(float(ee_position(self.env)[2]))
            return out
        self.env.step = stepped
        return self

    def __exit__(self, *e):
        self.env.step = self._orig


only = set(int(x) for x in a.only.split(",") if x)
rows, t0, k_seed = [], time.time(), 0
for obj in OBJECT_TABLE:
    for k in range(a.n):
        seed = a.seed + k_seed
        k_seed += 1
        reset_scene(env, seed)
        tab = OBJECT_TABLE[obj]
        p0 = object_position(env, obj)
        r_, th = a.delta / 100.0 * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        pre = np.array([p0[0] + r_ * np.cos(th), p0[1] + r_ * np.sin(th), tab["center_z"] + PREGRASP_H])
        row = {"object": obj, "seed": seed, "delta_cm": round(float(r_) * 100, 2)}
        if only and seed not in only:
            continue
        want_video = (k < a.video) if not only else True
        cap = Cap(env) if want_video else None
        if cap:
            cap.__enter__()
        try:
            ar = do_approach(env, pre)
            row["approach_ok"] = bool(ar["success"])
            post("/reset")
            zt = ZTrack(env)
            zt.__enter__()
            res = do_policy(env, rec)
            zt.__exit__()
            zs = zt.z or [0.0]
            fired = res["handoff_step"] is not None
            row.update(handoff_step=res["handoff_step"], steps=res["steps"])
            pos = object_position(env, obj)
            lifted = float(pos[2] - tab["center_z"])
            dist = float(np.linalg.norm(pos - ee_position(env)))
            picked = lifted >= 0.08 and dist < 0.08
            row.update(hand_lift_cm=round((max(zs) - min(zs)) * 100, 1), fired=fired, lifted_cm=round(lifted * 100, 1), dist_cm=round(dist * 100, 1), picked=bool(picked))
        except NotImplementedError as e:
            print("まだ書いていない部分に当たりました:", e)
            sys.exit(1)
        finally:
            if cap:
                cap.__exit__()
        if cap and cap.frames:
            from libero_ros.video import to_h264
            path = f"/workspace/outputs/videos/my_eval/{a.name}_{obj}_{k}_{'ok' if row['picked'] else 'ng'}.mp4"
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (512, 256))
            for f in cap.frames:
                vw.write(f)
            vw.release()
            to_h264(path)
        rows.append(row)
        print(f"{obj:10s} #{k} seed{seed} ずれ{row['delta_cm']:.1f}cm  切り替え={row['handoff_step']}  持ち上がり {row['lifted_cm']:5.1f}cm 手との距離 {row['dist_cm']:5.1f}cm  "
               f"手の上昇 {row['hand_lift_cm']:4.1f}cm  " f"{'成功' if row['picked'] else '失敗'}" , flush=True)

n = len(rows)
ok = sum(r["picked"] for r in rows)
lo, hi = wilson(ok, n)
print("\n===== 結果 =====")
print(f"全体: 持ち上げ成功 {ok}/{n} = {100 * ok / n:.0f}%  (95% 区間 {100 * lo:.0f}〜{100 * hi:.0f}%)  所要 {time.time() - t0:.0f} 秒")
for obj in OBJECT_TABLE:
    sub = [r for r in rows if r["object"] == obj]
    k_ok = sum(r["picked"] for r in sub)
    print(f"  {obj:10s} {k_ok}/{len(sub)}")
tp = sum(r["fired"] and r["picked"] for r in rows)
fp = sum(r["fired"] and not r["picked"] for r in rows)
fn = sum((not r["fired"]) and r["picked"] for r in rows)
tn = sum((not r["fired"]) and not r["picked"] for r in rows)
print(f"切り替え判定: TP {tp}(持ち上がって、True)/ FP {fp}(True なのに、持ち上がっていない)/ FN {fn}(見逃し)/ TN {tn}")
json.dump({"name": a.name, "n": n, "success": ok, "rows": rows}, open(f"/workspace/outputs/eval/source/{a.name}.json", "w"), ensure_ascii=False, indent=1)
print(f"保存: outputs/eval/source/{a.name}.json、動画: outputs/videos/my_eval/{a.name}_*.mp4")
