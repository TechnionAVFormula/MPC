import math
import numpy as np
from queue import Queue

cost_R = np.array([[R_Drive, 0], [0, R_delta]])

def J(state, path):
    e_c = 
    e_l = 
    return 

def R(command):
    return np.transpose(np.array(command)).dot(cost_R.dot(np.array(command)))

def L(state, command):
    beta_dyn = math.atan2(dyn_m.y_dot / dyn_m.x_dot)
    beta_kin = math.atan2(math.tan(command[1]) * L_REAR / (L_REAR + L_FRONT))
    return  q_beta * (beta_kin - beta_dyn)**2 

def C(slack):
    return q_s * slack + q_ss * (slack**2)

def step(state, command, path):
    integrator.model.State = 
    return integrator.RK4(command[1],command[0])

def step_cost(state, command, path, slack):
    return J(state, path) + R(command) + L(state, command) + C(slack)

    """
        params:
        -------------
        state - np.array, vehicle state after blend
        commands - 2 x horizon np.array, commands for cost calc
        slack - 1 x horizon np.array, slack vars to be optimized

        return:
        --------------
        total_cost - cost of given command matrix
        horizon_state - last state to be in after horizon drive
        steps_cost - cost of every step along the way, 1 x horizon vector
    """
def total_cost(state, commands, slack, horizon_state = None, steps_cost = None, prev_total_cost = 0):
    total_cost = 0
    
    if len(commands) > 1:
        steps_cost = Queue()
        for step in range(horizon):
            if not check_constraints(state,commands[step],slack[step]):
                return math.inf, None, None
            step_c = step_cost(state,commands[step],path,slack[step])
            steps_cost.put(step_c)
            total_cost += step_c
            state = step(state,commands[step])
    else:
        step_c = step_cost(state,commands[0],path,slack[0])
        total_cost = prev_total_cost - steps_cost.get(0) + step_c
        steps_cost.put(step_c)
        state = step(horizon_state,commands[0])
        
    return total_cost, state, steps_cost
    