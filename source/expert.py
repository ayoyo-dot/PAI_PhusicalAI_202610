import numpy as np

from provided import GRIP_YAW, OBJECT_TABLE, base_rotation,current_obs, ee_position, gripper_opening ,object_position,ee_quaternion,orientation_action, yawed_quat
from approach import plan_waypoints,p_action

def grasp_and_lift(env, obj, on_step=None, lift_h=0.15, max_steps=300):

    tol=0.005
    success_yoko = False
    success_tate = False
    success = False
    repeat = 0
    obj_pos = object_position(env,obj)
    p = ee_position(env)
    q_start = ee_quaternion(env)
    Rb = base_rotation(env)
    q_target = q_start

    goal = [obj_pos[0],obj_pos[1],p[2]]
    way_points = plan_waypoints(p,goal)
    gripper = -1
    success_yoko, repeat,error = approach_v2(q_target,gripper,on_step,env,obj,way_points,Rb,obj_pos,max_steps,goal,tol,repeat)

    obj_pos = object_position(env,obj)
    obj_height = OBJECT_TABLE[obj]["grasp_z"]
    p = ee_position(env)    
    goal = [obj_pos[0],obj_pos[1],obj_height]
    way_points = plan_waypoints(p,goal)
    gripper = -1
    success_tate, repeat,error = approach_v2(q_target,gripper,on_step,env,obj,way_points,Rb,obj_pos,max_steps,goal,tol,repeat)

    repeat_grip = 0
    grip_OK = 0
    label = "success"
    while(repeat_grip < 25):
        repeat_grip += 1 
        p   = ee_position(env) 
        q_now = ee_quaternion(env) 
        rot = orientation_action(Rb,q_target,q_now) 
        action = np.concatenate((np.zeros(3),rot,[1]))
        obs , *_ = env.step(action)
        gripper_open_width=gripper_opening(obs)
        if on_step != None:
            on_step(env, obs, action)  
        if gripper_open_width <= OBJECT_TABLE[obj]["width"] + 0.004:
            grip_OK = 1
            break 

    if grip_OK == 0:
      return(dict(success=False, label="no_grasp", steps = repeat))


    obj_pos = object_position(env,obj)
    goal_height = OBJECT_TABLE[obj]["grasp_z"] + lift_h
    p = ee_position(env)    
    goal = [p[0],p[1],goal_height]
    gripper = 1
    way_points = plan_waypoints(p,goal)
    success_up, repeat,error = approach_v2(q_target,gripper,on_step,env,obj,way_points,Rb,obj_pos,max_steps,goal,tol,repeat)

    p = ee_position(env)    
    error = np.linalg.norm(p - goal)
    if error < tol:
        success = True

    p = ee_position(env)  
    obj_pos = object_position(env,obj)
    obj_height = OBJECT_TABLE[obj]["grasp_z"]

    lifted = obj_pos[2] - OBJECT_TABLE[obj]["center_z"]
    dist = np.linalg.norm(obj_pos - p)

    if lifted < 0.08 or dist >= 0.08:
      return(dict(success=False, label="slipped", steps = repeat))    
    else:    
      return(dict(success=success, label=label, steps = repeat))

    return(dict(success=success, label=label, steps = repeat))


def approach_v2(q_target,gripper,on_step,env,obj,way_points,Rb,obj_pos,max_steps,goal,tol,repeat):
    success = False

    for way_point in way_points:   
        while(repeat < max_steps):
            repeat += 1 
            q_now = ee_quaternion(env) 
            rot = orientation_action(Rb,q_target,q_now)  
            p = ee_position(env)
            err_value = np.linalg.norm(p - way_point)
            if err_value < tol * 0.5:     
                break
            err_base = way_point - p 
            p_action_point = p_action(err_base, Rb)
            action = np.concatenate((p_action_point,rot,[gripper]))
            obs , *_ = env.step(action)
            if on_step != None:
                on_step(env, obs, action)
    p = ee_position(env)    
    error = np.linalg.norm(p - goal)
    if error < tol:
       success = True

    return (success, repeat ,error)