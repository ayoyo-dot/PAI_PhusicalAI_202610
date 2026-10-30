import numpy as np

from .control import base_transform


def shuffle_layout(env, seed, yaw=True):
    rng = np.random.default_rng(seed)
    sim = env.sim
    names = [n for n in env.env.obj_body_id if not n.startswith("basket")]
    adr = {n: sim.model.get_joint_qpos_addr(env.env.objects_dict[n].joints[0]) for n in names}
    cur = {n: sim.data.qpos[adr[n][0]:adr[n][0] + 7].copy() for n in names}
    perm = rng.permutation(len(names))
    for n, k in zip(names, perm):
        q = cur[n].copy()
        q[:2] = cur[names[k]][:2] + rng.uniform(-0.015, 0.015, 2)
        yaw_ = rng.uniform(-np.pi / 2, np.pi / 2) if yaw else 0.0
        w1, x1, y1, z1 = np.cos(yaw_ / 2), 0.0, 0.0, np.sin(yaw_ / 2)
        w2, x2, y2, z2 = q[3:7]
        q[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
        sim.data.qpos[adr[n][0]:adr[n][0] + 7] = q
    sim.data.qvel[:] = 0
    sim.forward()


def rotate_object_yaw(env, name, dyaw):
    sim = env.sim
    adr = sim.model.get_joint_qpos_addr(env.env.objects_dict[name].joints[0])
    q = sim.data.qpos[adr[0] + 3:adr[0] + 7].copy()
    w1, x1, y1, z1 = np.cos(dyaw / 2), 0.0, 0.0, np.sin(dyaw / 2)
    w2, x2, y2, z2 = q
    sim.data.qpos[adr[0] + 3:adr[0] + 7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    sim.data.qvel[:] = 0
    sim.forward()


def random_layout(env, rng, x_range=(-0.20, 0.12), y_range=(-0.18, 0.13), min_dist=0.12, max_tries=500):
    sim = env.sim
    names = [n for n in env.env.obj_body_id if not n.startswith("basket")]
    for _ in range(200):
        placed, ok = {}, True
        for n in names:
            for _ in range(max_tries):
                p = np.array([rng.uniform(*x_range), rng.uniform(*y_range)])
                if all(np.linalg.norm(p - q) >= min_dist for q in placed.values()):
                    placed[n] = p
                    break
            else:
                ok = False
                break
        if ok:
            break
    else:
        raise RuntimeError(f"could not place {len(names)} objects with min_dist={min_dist}")
    for n, p in placed.items():
        adr = sim.model.get_joint_qpos_addr(env.env.objects_dict[n].joints[0])
        sim.data.qpos[adr[0]:adr[0] + 2] = p
    sim.data.qvel[:] = 0
    sim.forward()
    return {n: (float(p[0]), float(p[1])) for n, p in placed.items()}


def move_into_basket(env, names, basket_name="basket_1", slots=((-0.03, -0.03), (0.03, 0.03), (0.03, -0.03), (-0.03, 0.03)), drop_h=0.12):
    from .grasp import basket_geometry
    sim = env.sim
    lo, hi, top = basket_geometry(env, basket_name)
    T = base_transform(env)
    c = (lo + hi) / 2
    for k, n in enumerate(names):
        off = np.array(slots[k % len(slots)])
        world_xy = T[:3, :3] @ np.array([c[0] + off[0], c[1] + off[1], 0.0]) + T[:3, 3]
        adr = sim.model.get_joint_qpos_addr(env.env.objects_dict[n].joints[0])
        sim.data.qpos[adr[0]:adr[0] + 2] = world_xy[:2]
        sim.data.qpos[adr[0] + 2] = top + drop_h
    sim.data.qvel[:] = 0
    sim.forward()

