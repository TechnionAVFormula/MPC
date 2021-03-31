import math
import numpy as np
from queue import Queue
import json
from .Integration import Integration

VEHICLE_DATA = json.loads(open("vehicle_data.json", "r").read())
OPT_PARAMS = json.loads(open("opt_params.json", "r").read())

L = VEHICLE_DATA["Wheel_base"]  # Total length
L_REAR = VEHICLE_DATA["Rear_length"]  # rear length
L_FRONT = L - L_REAR  # front length

R_delta = OPT_PARAMS["R_delta"]
R_Drive = OPT_PARAMS["R_Drive"]

q_beta = OPT_PARAMS["q_beta"]
q_s = OPT_PARAMS["q_s"]
q_ss = OPT_PARAMS["q_ss"]
q_e = OPT_PARAMS["q_e"]
q_l = OPT_PARAMS["q_l"]

horizon = OPT_PARAMS["horizon"]

R_track = OPT_PARAMS["R_track"]
V_max = VEHICLE_DATA["Vehicle_max_speed"]

cost_R = np.array([[R_Drive, 0], [0, R_delta]])

def J(state, path, t_param):
    x_ref = path[0]*t_param**3 + path[1]*t_param**2 + path[2]*t_param + path[0]
    y_ref = t_param
    path_lin = path_driv(path)
    e_c = #dist state position from path_deriv
    e_l = #dist state position from 1/path_deriv
    return 

def R(command):
    return np.transpose(np.array(command)).dot(cost_R.dot(np.array(command)))

def L(dyn_m, kin_m, command):
    beta_dyn = math.atan2(dyn_m.y_dot , dyn_m.x_dot)
    beta_kin = math.atan2(math.tan(command[1]) * L_REAR , (L_REAR + L_FRONT))
    return  q_beta * (beta_kin - beta_dyn)**2 

def C(slack):
    return q_s * abs(slack) + q_ss * (slack**2)

def do_step(integrator: Integration, command, path):
    integrator.RK4(command[1],command[0])

def step_cost(integrator: Integration, command, path, slack):
    return J(integrator.state, path, integrator.t_param) + R(command) + L(integrator.dyn_m, integrator.kin_m, command) + C(slack)

def check_constraints(integrator: Integration, command, slack, path):


"""
    params:
    -------------
    state - np.array, vehicle state after blend
    commands - 2 x horizon np.array, commands for cost calc
                commands[i][0] - D
                commands[i][1] - delta
    slack - 1 x horizon np.array, slack vars to be optimized
    
    return:
    --------------
    total_cost - cost of given command matrix
    horizon_state - last state to be in after horizon drive
    steps_cost - cost of every step along the way, 1 x horizon vector
"""
def total_cost_calc(state, commands, slack, path, steps_cost = None, prev_total_cost = 0, prev_t_param = 0, new_state = False):
    total_cost = 0
    if new_state:
        t_param = 0
    else:
        t_param = prev_t_param
    integrator = Integration(state=state, t_param=t_param)
    if len(commands) > 1 or new_state:
        steps_cost = Queue()
        for step in range(horizon):
            if not check_constraints(integrator, commands[step], slack[step], path):
                return math.inf, None, None, 0
            step_c = step_cost(integrator, commands[step], path, slack[step])
            steps_cost.put(step_c)
            total_cost += step_c
            do_step(integrator, commands[step], path)

    else:
        if not check_constraints(integrator, commands[0], slack[0], path):
            return math.inf, None, None, 0
        step_c = step_cost(integrator, commands[0], path, slack[0])
        total_cost = prev_total_cost - steps_cost.get(0) + step_c
        steps_cost.put(step_c)
        do_step(integrator, commands[0], path)
        
    return total_cost, integrator.state, steps_cost, integrator.t_param
    