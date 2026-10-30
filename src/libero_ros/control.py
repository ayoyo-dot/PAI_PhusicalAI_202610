import numpy as np
from scipy.spatial.transform import Rotation

ACTION_POS_SCALE = 0.05
ACTION_ROT_SCALE = 0.5
CONTROL_HZ = 20.0
CAMERA = "agentview"


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def rotvec_error(q_target, q_current):
    qc_inv = np.array([-q_current[0], -q_current[1], -q_current[2], q_current[3]])
    dq = _quat_mul(q_target, qc_inv)
    if dq[3] < 0:
        dq = -dq
    n = np.linalg.norm(dq[:3])
    return np.zeros(3) if n < 1e-9 else dq[:3] / n * 2.0 * np.arctan2(n, dq[3])


def base_transform(env):
    sim = env.sim
    bid = sim.model.body_name2id(env.robots[0].robot_model.root_body)
    T = np.eye(4)
    T[:3, :3] = sim.data.body_xmat[bid].reshape(3, 3)
    T[:3, 3] = sim.data.body_xpos[bid]
    return T


def get_observation(env, size=512, camera=CAMERA):
    from robosuite.utils import camera_utils as cu
    sim = env.sim
    rgb, depth = sim.render(width=size, height=size, camera_name=camera, depth=True)
    depth = cu.get_real_depth_map(sim, depth)
    rgb, depth = np.ascontiguousarray(rgb[::-1]), np.ascontiguousarray(depth[::-1])
    K = cu.get_camera_intrinsic_matrix(sim, camera, size, size)
    T_base_cam = np.linalg.inv(base_transform(env)) @ cu.get_camera_extrinsic_matrix(sim, camera)
    return rgb, depth, K, T_base_cam


def current_obs(env):
    return env.env._get_observations(force_update=True)


def ee_pose_base(env, obs):
    T = base_transform(env)
    Rb, tb = T[:3, :3], T[:3, 3]
    p = Rb.T @ (np.array(obs["robot0_eef_pos"]) - tb)
    R = Rb.T @ Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()
    return p, Rotation.from_matrix(R).as_quat()


class EEController:

    def __init__(self, rate=CONTROL_HZ, pos_gain=1.0, rot_gain=1.0, ki=2.0, i_limit=0.03):
        self.rate, self.pos_gain, self.rot_gain, self.ki, self.i_limit = rate, pos_gain, rot_gain, ki, i_limit
        self.i_term = np.zeros(3)

    def reset(self):
        self.i_term[:] = 0.0

    def action(self, Rb, target_p, target_q, cur_p, cur_q, gripper=-1.0):
        err = target_p - cur_p
        if np.linalg.norm(err) < 0.02:
            self.i_term = np.clip(self.i_term + self.ki * err / self.rate, -self.i_limit, self.i_limit)
        else:
            self.i_term *= 0.9
        a = np.zeros(7)
        a[:3] = np.clip(Rb @ ((err + self.i_term) * self.pos_gain / ACTION_POS_SCALE), -1, 1)
        a[3:6] = np.clip(Rb @ rotvec_error(target_q, cur_q) * self.rot_gain / ACTION_ROT_SCALE, -1, 1)
        a[6] = gripper
        return a


def make_waypoints(start, goal, z_safe=True):
    z = max(start[2], goal[2])
    pts = [np.array([start[0], start[1], z]), np.array([goal[0], goal[1], z]), np.asarray(goal, dtype=float)] \
        if z_safe else [np.asarray(goal, dtype=float)]
    out, prev = [], np.asarray(start, dtype=float)
    for w in pts:
        if np.linalg.norm(w - prev) > 1e-4:
            out.append(w)
            prev = w
    return out or [np.asarray(goal, dtype=float)]


def move_to(env, goal, vmax=0.15, amax=0.5, tol=0.005, settle_speed=0.01, settle_samples=3, max_steps=None,
            gripper=-1.0, ctrl=None, z_safe=True, on_step=None, yaw_offset=0.0):
    goal = np.asarray(goal, dtype=float)
    dt = 1.0 / CONTROL_HZ
    ctrl = ctrl or EEController()
    ctrl.reset()
    Rb = base_transform(env)[:3, :3]
    obs = current_obs(env)
    cp, hold_q = ee_pose_base(env, obs)
    if yaw_offset:
        hold_q = (Rotation.from_euler("z", yaw_offset) * Rotation.from_quat(hold_q)).as_quat()
    waypoints = make_waypoints(cp, goal, z_safe)
    path = sum(np.linalg.norm(w - (waypoints[i - 1] if i else cp)) for i, w in enumerate(waypoints))
    max_steps = max_steps or int((8.0 + 6.0 * path / vmax) * CONTROL_HZ)
    ref, v, wp_i, settled, prev = cp.copy(), 0.0, 0, 0, cp.copy()
    for step in range(1, max_steps + 1):
        wp = waypoints[wp_i]
        rem = float(np.linalg.norm(wp - ref))
        v = min(vmax, np.sqrt(2.0 * amax * rem), v + amax * dt)
        v = max(v, 0.02) if rem > 1e-4 else 0.0
        if rem > 1e-9:
            ref = ref + (wp - ref) / rem * min(v * dt, rem)
        if np.linalg.norm(wp - ref) < 1e-4 and wp_i < len(waypoints) - 1:
            wp_i, v = wp_i + 1, 0.0
        action = ctrl.action(Rb, ref, hold_q, cp, ee_pose_base(env, obs)[1], gripper)
        obs, *_ = env.step(action)
        if on_step is not None:
            on_step(env, obs, action)
        cp, _ = ee_pose_base(env, obs)
        speed = float(np.linalg.norm(cp - prev) / dt)
        prev = cp
        err = float(np.linalg.norm(goal - cp))
        at_goal = wp_i == len(waypoints) - 1 and np.linalg.norm(goal - ref) < 1e-4
        settled = settled + 1 if (at_goal and err < tol and speed < settle_speed) else 0
        if settled >= settle_samples:
            return {"success": True, "steps": step, "error": err, "final_pos": cp}
    return {"success": False, "steps": max_steps, "error": float(np.linalg.norm(goal - cp)), "final_pos": cp}
