from provided import GRIP_YAW, OBJECT_TABLE, base_rotation,current_obs, ee_position, gripper_opening ,object_position,ee_quaternion,orientation_action, yawed_quat
import numpy as np

class HandoffDetector:
    def __init__(self):
        self.reset()
        self.threshold = 10 * 0.01
        self.min_obj_width = 1 * 0.01
        self.max_obj_width = 7.2 * 0.01

    def reset(self):
        self.first_point = 0
        self.step = 0
        self.gripped = False
        return

    def update(self, grip_cmd, finger_opening, ee_z):
        self.step += 1

        if self.step == 1:
            self.low_point = ee_z
            return False

        if ee_z < self.low_point:
            self.low_point = ee_z
            return False


        if   grip_cmd > 0 and self.gripped == False:
            self.gripped = True
            return False

        if self.gripped != True:
            return False

        
        if (ee_z - self.low_point) > self.threshold:
            if finger_opening > self.min_obj_width and self.max_obj_width > finger_opening:
                return True    
            else:
                False    
        else:
            return False


def run_policy_until_handoff(env, observe_fn, act_fn, detector, max_steps=150):
    obs = current_obs(env)
    for i in range(max_steps):
   
        ag, wr, st = observe_fn(obs)
        a = act_fn(ag, wr, st)
        a = np.clip(a, -1, 1)
        obs, *_ = env.step(a)

        ee_z = ee_position(env)[2]
        finger_opening = gripper_opening(obs)
        grip_cmd = a[6]

        if detector.update(grip_cmd, finger_opening, ee_z) :
            return(dict(handoff_step= i, steps = i+1))

    return(dict(handoff_step= None, steps = max_steps))
  
