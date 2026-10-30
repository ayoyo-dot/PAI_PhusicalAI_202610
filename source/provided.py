import numpy as np
from scipy.spatial.transform import Rotation

from libero_ros.control import base_transform, current_obs, ee_pose_base, rotvec_error

GRIP_YAW = np.pi / 2
ACTION_POS_SCALE = 0.05
ACTION_ROT_SCALE = 0.5
CONTROL_HZ = 20.0


def ee_position(env):
    return ee_pose_base(env, current_obs(env))[0]


def ee_quaternion(env):
    return ee_pose_base(env, current_obs(env))[1]


def base_rotation(env):
    return base_transform(env)[:3, :3]


def yawed_quat(q, yaw):
    return (Rotation.from_euler("z", yaw) * Rotation.from_quat(q)).as_quat()


def orientation_action(Rb, target_q, cur_q, gain=1.0):
    return np.clip(Rb @ rotvec_error(target_q, cur_q) * gain / ACTION_ROT_SCALE, -1, 1)


def gripper_opening(obs):
    g = obs["robot0_gripper_qpos"]
    return float(g[0] - g[1])


OBJECT_TABLE = {
    "milk": {"center_z": 0.070, "grasp_z": 0.097, "width": 0.053},
    "bbq_sauce": {"center_z": 0.055, "grasp_z": 0.073, "width": 0.029},
    "ketchup": {"center_z": 0.072, "grasp_z": 0.112, "width": 0.037},
}
PREGRASP_H = 0.12
FINGER_FULL_OPEN = 0.08
BDDL = "/workspace/scenes/sort_one_basket.bddl"


def make_env():
    from libero.libero.envs import OffScreenRenderEnv
    return OffScreenRenderEnv(bddl_file_name=BDDL, camera_names=["agentview", "robot0_eye_in_hand"], camera_heights=128,
                              camera_widths=128, use_camera_obs=False, horizon=10 ** 7, ignore_done=True)


def reset_scene(env, seed, shuffle=False):
    from libero_ros.layout import rotate_object_yaw, shuffle_layout
    env.seed(seed)
    np.random.seed(seed)
    env.reset()
    if shuffle:
        shuffle_layout(env, seed=seed, yaw=False)
    rotate_object_yaw(env, "cream_cheese_1", np.pi / 2)
    for _ in range(15):
        env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))


def object_position(env, name):
    from libero_ros.grasp import object_pose_base
    return object_pose_base(env, f"{name}_1")[0]


def make_recorder(env, size=128):
    from libero_ros.demos import Recorder
    return Recorder(env, size)
