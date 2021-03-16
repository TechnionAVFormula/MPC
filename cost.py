import math
import numpy as np

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

def C():
    return 

def step(state, command, path):
    return 

def step_cost(state, command, path):
    return J(state, path) + R(command) + L(state, command) + C()

def total_cost():
    total_cost = 0
    for step in range(horizon):
        if not check_constraints(state,commands[step]):
            return math.inf
        total_cost += step_cost(state,commands[step],path)
        state = step(state,commands[step])
    
    return total_cost
    