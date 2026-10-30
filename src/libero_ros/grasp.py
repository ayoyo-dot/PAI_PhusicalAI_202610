import numpy as np
from scipy.spatial.transform import Rotation

from .control import CONTROL_HZ, EEController, base_transform, current_obs, ee_pose_base, move_to

GRIPPER_OPEN, GRIPPER_CLOSE = -1.0, 1.0


def object_pose_base(env, name):
    T = base_transform(env)
    sim = env.sim
    b = env.env.obj_body_id[name]
    p = T[:3, :3].T @ (sim.data.body_xpos[b] - T[:3, 3])
    R = T[:3, :3].T @ sim.data.body_xmat[b].reshape(3, 3)
    yaw = float(Rotation.from_matrix(R).as_euler("xyz")[2])
    return p, yaw


FINGER_OPENING = 0.077
HAND_HALF_LEN = 0.10
HAND_HALF_WID = 0.04
PALM_ABOVE_SITE = 0.036
MIN_EEF_Z = 0.014
GRASP_WIDTH_MARGIN = 0.007
ALIGN_TOL = 0.003


def _descendants(m, body):
    out = [body]
    for c in range(m.nbody):
        if m.body_parentid[c] == body and c != body:
            out += _descendants(m, c)
    return out


def object_points(env, name):
    sim = env.sim
    m, d = sim.model, sim.data
    T = base_transform(env)
    ids = set(_descendants(m, env.env.obj_body_id[name]))
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in ids or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        c, R, gt, sz = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_type[g], m.geom_size[g]
        if gt == 7:
            ctr, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        elif gt == 6:
            ctr, half = np.zeros(3), np.array(sz[:3])
        elif gt in (5, 3):
            ctr, half = np.zeros(3), np.array([sz[0], sz[0], sz[1]])
        else:
            ctr, half = np.zeros(3), np.full(3, sz[0])
        corners = np.array([[a, b, c_] for a in (-1, 1) for b in (-1, 1) for c_ in (-1, 1)]) * half + ctr
        pts.append((R @ corners.T).T + c)
    P = np.vstack(pts)
    return (T[:3, :3].T @ (P - T[:3, 3]).T).T


def _hand_rect(center_xy, yaw):
    ax = np.array([-np.sin(yaw), np.cos(yaw)])
    nx = np.array([np.cos(yaw), np.sin(yaw)])
    return center_xy, ax, nx


def _overlaps(center_xy, yaw, lo_xy, hi_xy, pad=0.01):
    c, ax, nx = _hand_rect(center_xy, yaw)
    bc, bh = (lo_xy + hi_xy) / 2, (hi_xy - lo_xy) / 2 + pad
    for axis in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), ax, nx):
        r_a = HAND_HALF_LEN * abs(ax @ axis) + HAND_HALF_WID * abs(nx @ axis)
        r_b = bh[0] * abs(axis[0]) + bh[1] * abs(axis[1])
        if abs((bc - c) @ axis) > r_a + r_b:
            return False
    return True


def plan_grasp(env, name, obj_pos):
    P = object_points(env, name)
    top = float(P[:, 2].max())
    grasp_z = float(max(obj_pos[2], top - PALM_ABOVE_SITE + 0.002, MIN_EEF_Z))
    others = {n: object_points(env, n) for n in env.env.obj_body_id if n != name}
    boxes = {n: (Q[:, :2].min(0), Q[:, :2].max(0)) for n, Q in others.items()}
    c_xy = obj_pos[:2]
    best = None
    for yaw in (np.pi / 2, 0.0, -np.pi / 2, np.pi):
        axis = np.array([-np.sin(yaw), np.cos(yaw)])
        width = float((P[:, :2] @ axis).max() - (P[:, :2] @ axis).min())
        hits = [n for n, (lo, hi) in boxes.items() if _overlaps(c_xy, yaw, lo, hi)]
        ok_w = width <= FINGER_OPENING - GRASP_WIDTH_MARGIN
        score = (ok_w, -len(hits), -abs(yaw - np.pi / 2))
        if best is None or score > best[0]:
            best = (score, yaw, width, hits)
    _, yaw, width, hits = best
    reason = "ok" if (width <= FINGER_OPENING - GRASP_WIDTH_MARGIN and not hits) else \
        ("width %.1fcm > opening" % (width * 100) if width > FINGER_OPENING - GRASP_WIDTH_MARGIN else "hand overlaps " + ",".join(hits))
    return {"grasp_z": grasp_z, "grip_yaw": float(yaw), "width": width, "top": top, "reason": reason, "hits": hits}


def aligned_grip_yaw(obj_yaw, base=np.pi / 2):
    folded = (obj_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    return base + folded


def hold(env, steps, gripper, on_step=None):
    obs = None
    for _ in range(steps):
        a = np.zeros(7)
        a[6] = gripper
        obs, *_ = env.step(a)
        if on_step is not None:
            on_step(env, obs, a)
    return obs


def scripted_grasp(env, name, delta_xy=(0.0, 0.0), grasp_dz=None, pregrasp_h=0.12, lift_h=0.15, close_steps=15,
                   lift_gain_ok=0.08, recorder=None, grip_yaw="plan"):
    out = {"object": name, "delta_xy": [float(delta_xy[0]), float(delta_xy[1])]}
    p0, yaw0 = object_pose_base(env, name)
    out["obj_pos0"] = p0.tolist()
    out["obj_yaw0"] = yaw0
    plan = plan_grasp(env, name, p0)
    out["plan"] = {k: (v if not isinstance(v, (np.floating, np.ndarray)) else float(v)) for k, v in plan.items()}
    if grip_yaw == "plan":
        grip_yaw = plan["grip_yaw"]
    grasp_z = plan["grasp_z"] if grasp_dz is None else p0[2] + grasp_dz
    out["grip_yaw"], out["grasp_z"] = float(grip_yaw), float(grasp_z)

    pre = p0 + np.array([delta_xy[0], delta_xy[1], pregrasp_h])
    r = move_to(env, pre, gripper=GRIPPER_OPEN, yaw_offset=grip_yaw)
    if not r["success"]:
        out.update(success=False, label="pregrasp_failed", note="pregrasp unreachable/timeout")
        return out
    if recorder is not None:
        recorder.start()
    on_step = recorder.on_step if recorder is not None else None

    grasp = np.array([p0[0], p0[1], grasp_z])
    above = np.array([p0[0], p0[1], pre[2]])
    r = move_to(env, above, gripper=GRIPPER_OPEN, on_step=on_step, tol=ALIGN_TOL, z_safe=False)
    if r["success"]:
        r = move_to(env, grasp, gripper=GRIPPER_OPEN, on_step=on_step, z_safe=False)
    p1, _ = object_pose_base(env, name)
    out["knocked_xy"] = float(np.linalg.norm(p1[:2] - p0[:2]))
    if not r["success"]:
        out.update(success=False, label="descent_failed", note="descent unreachable/timeout (hand blocked above the object?)")
        return out
    if out["knocked_xy"] > 0.03:
        out.update(success=False, label="knocked", note=f"object moved {out['knocked_xy'] * 100:.1f} cm before grasp")
        return out

    hold(env, close_steps, GRIPPER_CLOSE, on_step)
    lift_goal = np.array([p0[0], p0[1], grasp_z + lift_h])
    rl = move_to(env, lift_goal, gripper=GRIPPER_CLOSE, on_step=on_step, z_safe=False)
    hold(env, 8, GRIPPER_CLOSE, on_step)

    p2, _ = object_pose_base(env, name)
    ee_p = rl["final_pos"]
    gain = float(p2[2] - p0[2])
    out.update(lift_gain=gain, obj_ee_dist=float(np.linalg.norm(p2 - ee_p)), lift_ok=bool(rl["success"]))
    if gain >= lift_gain_ok and out["obj_ee_dist"] < 0.08:
        out.update(success=True, label="success")
    elif gain >= 0.02:
        out.update(success=False, label="slipped", note=f"lifted {gain * 100:.1f} cm then lost")
    else:
        out.update(success=False, label="no_grasp", note=f"object not lifted ({gain * 100:.1f} cm)")
    return out


def _smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def scripted_grasp_funnel(env, name, delta_xy=(0.0, 0.0), pregrasp_h=0.12, lift_h=0.15, close_steps=15,
                          lift_gain_ok=0.08, recorder=None, grip_yaw="plan", e_lo=0.006, e_hi=0.020,
                          vz_max=0.12, ki=2.0, max_steps=220, close_margin=0.004, close_xy=0.012):
    out = {"object": name, "delta_xy": [float(delta_xy[0]), float(delta_xy[1])], "mode": "funnel"}
    p0, yaw0 = object_pose_base(env, name)
    out["obj_pos0"], out["obj_yaw0"] = p0.tolist(), yaw0
    plan = plan_grasp(env, name, p0)
    out["plan"] = {k: (v if not isinstance(v, (np.floating, np.ndarray)) else float(v)) for k, v in plan.items()}
    if grip_yaw == "plan":
        grip_yaw = plan["grip_yaw"]
    grasp_z = plan["grasp_z"]
    out["grip_yaw"], out["grasp_z"] = float(grip_yaw), float(grasp_z)
    pre_z = p0[2] + pregrasp_h

    pre = p0 + np.array([delta_xy[0], delta_xy[1], pregrasp_h])
    r = move_to(env, pre, gripper=GRIPPER_OPEN, yaw_offset=grip_yaw)
    if not r["success"]:
        out.update(success=False, label="pregrasp_failed", note="pregrasp unreachable/timeout")
        return out
    if recorder is not None:
        recorder.start()
    on_step = recorder.on_step if recorder is not None else None

    dt = 1.0 / CONTROL_HZ
    ctrl = EEController(ki=ki)
    Rb = base_transform(env)[:3, :3]
    obs = current_obs(env)
    cp, hold_q = ee_pose_base(env, obs)
    z_ref = float(cp[2])
    reached = False
    for step in range(max_steps):
        err_xy = float(np.linalg.norm(p0[:2] - cp[:2]))
        z_des = grasp_z + (pre_z - grasp_z) * _smoothstep((err_xy - e_lo) / (e_hi - e_lo))
        z_ref += float(np.clip(z_des - z_ref, -vz_max * dt, vz_max * dt))
        target = np.array([p0[0], p0[1], z_ref])
        a = ctrl.action(Rb, target, hold_q, cp, ee_pose_base(env, obs)[1], GRIPPER_OPEN)
        obs, *_ = env.step(a)
        if on_step is not None:
            on_step(env, obs, a)
        cp, _ = ee_pose_base(env, obs)
        if cp[2] <= grasp_z + 0.004 and err_xy < close_xy:
            reached = True
            break
    p1, _ = object_pose_base(env, name)
    out["knocked_xy"] = float(np.linalg.norm(p1[:2] - p0[:2]))
    out["descent_steps"] = step + 1
    if not reached:
        out.update(success=False, label="descent_failed", note="funnel descent did not reach grasp height")
        return out
    if out["knocked_xy"] > 0.03:
        out.update(success=False, label="knocked", note=f"object moved {out['knocked_xy'] * 100:.1f} cm before grasp")
        return out

    n_close = 0
    for n_close in range(1, close_steps + 1):
        a = np.zeros(7)
        a[6] = GRIPPER_CLOSE
        obs, *_ = env.step(a)
        if on_step is not None:
            on_step(env, obs, a)
        g = obs["robot0_gripper_qpos"]
        if (g[0] - g[1]) <= plan["width"] + close_margin and n_close >= 2:
            break
    out["close_steps_used"] = n_close
    lift_goal = np.array([p0[0], p0[1], grasp_z + lift_h])
    rl = move_to(env, lift_goal, gripper=GRIPPER_CLOSE, on_step=on_step, z_safe=False)
    hold(env, 8, GRIPPER_CLOSE, on_step)
    p2, _ = object_pose_base(env, name)
    gain = float(p2[2] - p0[2])
    out.update(lift_gain=gain, obj_ee_dist=float(np.linalg.norm(p2 - rl["final_pos"])), lift_ok=bool(rl["success"]))
    if gain >= lift_gain_ok and out["obj_ee_dist"] < 0.08:
        out.update(success=True, label="success")
    elif gain >= 0.02:
        out.update(success=False, label="slipped", note=f"lifted {gain * 100:.1f} cm then lost")
    else:
        out.update(success=False, label="no_grasp", note=f"object not lifted ({gain * 100:.1f} cm)")
    return out


BASKET_INNER_MARGIN = 0.02
RIM_CLEARANCE = 0.04


def basket_geometry(env, basket_name):
    P = object_points(env, basket_name)
    return P[:, :2].min(0), P[:, :2].max(0), float(P[:, 2].max())


def object_in_basket(env, obj_name, basket_name):
    lo, hi, top = basket_geometry(env, basket_name)
    p, _ = object_pose_base(env, obj_name)
    inside = bool(np.all(p[:2] > lo + BASKET_INNER_MARGIN) and np.all(p[:2] < hi - BASKET_INNER_MARGIN))
    return inside and bool(p[2] < top), p


def scripted_place(env, obj_name, basket_xy=None, basket_top=None, release_steps=8, settle_steps=15, recorder=None,
                   basket_name="basket_1", vmax=0.20, direct=True, hang=None):
    on_step = recorder.on_step if recorder is not None else None
    if basket_xy is None or basket_top is None:
        lo, hi, top = basket_geometry(env, basket_name)
        basket_xy = (lo + hi) / 2 if basket_xy is None else np.asarray(basket_xy)
        basket_top = top if basket_top is None else basket_top
    ee_p, _ = ee_pose_base(env, current_obs(env))
    if hang is None:
        bottom = float(object_points(env, obj_name)[:, 2].min())
        hang = float(ee_p[2] - bottom)
    carry_z = float(max(ee_p[2], basket_top + hang + RIM_CLEARANCE))
    goal = np.array([basket_xy[0], basket_xy[1], carry_z])
    r = move_to(env, goal, gripper=GRIPPER_CLOSE, on_step=on_step, vmax=vmax, tol=0.01, z_safe=not direct)
    out = {"carry_z": carry_z, "hang": hang, "move_ok": bool(r["success"]), "steps": int(r["steps"])}
    hold(env, release_steps, GRIPPER_OPEN, on_step)
    hold(env, settle_steps, GRIPPER_OPEN, on_step)
    placed, p = object_in_basket(env, obj_name, basket_name)
    out.update(placed=placed, obj_pos=p.tolist())
    return out

