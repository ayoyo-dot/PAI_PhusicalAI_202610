import numpy as np

from provided import ACTION_POS_SCALE, GRIP_YAW, ee_position, ee_quaternion, base_rotation, orientation_action, yawed_quat

DIFF = 1e-6

def plan_waypoints(start, goal):

    return_point_list = []
    start_height = start[2]
    goal_height = goal[2]
    next_point = start

    if abs(start_height -  goal_height) < DIFF :
        pass
    else:
        z_safe = max(start_height,goal_height)

        if(start_height == z_safe):
            next_point = [goal[0],goal[1],z_safe]
        else:
            next_point = [start[0],start[1],z_safe]
        if np.linalg.norm(np.array(next_point)-np.array(start)) > DIFF:   
            return_point_list.append(next_point)

    if np.linalg.norm(np.array(next_point)-np.array(goal)) > DIFF:
        return_point_list.append(goal)

    return return_point_list


def p_action(err_base, Rb, gain=1.0):

    err_base_5cm = (err_base*100 / 5) * gain

    lotated = Rb @ err_base_5cm 
    cliped = np.clip(lotated , -1, 1)

    return(cliped)


def approach(env, goal, tol=0.005, max_steps=400, gain=1.0):  
    success = False

    p = ee_position(env)
    way_points = plan_waypoints(p,goal)
    q_start = ee_quaternion(env)
    q_target = yawed_quat(q_start,GRIP_YAW)
    Rb = base_rotation(env)
    repeat = 0
    for way_point in way_points:
        while(repeat < max_steps):
            repeat += 1
            q_now = ee_quaternion(env) 
            rot = orientation_action(Rb,q_target,q_now)  
            p = ee_position(env)
            err_value = np.linalg.norm(p - way_point)
            if (err_value < tol * 0.5) and (max(np.abs(rot)) < 0.05/0.5 ):     
                break
            err_base = way_point - p 
            p_action_point = p_action(err_base, Rb,gain)
            env.step(np.concatenate((p_action_point,rot,[-1])))
    p = ee_position(env)    
    error = np.linalg.norm(p - goal)
    if error < tol:
        success = True

    return dict(success = success, steps=repeat ,error = error)
