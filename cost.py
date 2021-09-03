import math
import numpy as np
import json
from Integration import Integration

from pathlib import Path

VEHICLE_DATA = json.loads(open(Path("config") / "vehicle_data.json", "r").read())
OPT_PARAMS = json.loads(open(Path("config") / "opt_params.json", "r").read())

L = VEHICLE_DATA["wheel_base"]  # Total length
L_REAR = VEHICLE_DATA["Rear_length"]  # rear length
L_FRONT = L - L_REAR  # front length

D_R = VEHICLE_DATA["Magic_D_rear"]
D_F = VEHICLE_DATA["Magic_D_front"]

R_delta = OPT_PARAMS["R_delta"]
R_Drive = OPT_PARAMS["R_D"]

max_delta = VEHICLE_DATA["maxAlpha"]

p_long = VEHICLE_DATA["p_long"]
p_ellipse = VEHICLE_DATA["p_ellipse"]

q_beta = OPT_PARAMS["q_beta"]
q_s = OPT_PARAMS["q_s"]
q_ss = OPT_PARAMS["q_ss"]
q_c = OPT_PARAMS["q_c"]
q_l = OPT_PARAMS["q_l"]
gamma = OPT_PARAMS["gamma"]

static_horizon = OPT_PARAMS["horizon"]

R_track = OPT_PARAMS["R_track"]
V_max = VEHICLE_DATA["Vehicle_max_speed"]

cost_R = np.array([[R_Drive, 0], [0, R_delta]])


def path_deriv(path, y):
    return 3 * path[0] * (y ** 2) + 2 * path[1] * y + path[2]


def dist_line_from_point(slope, y_ref, x_ref, y_car, x_car):
    A = -slope
    B = 1
    C = slope * y_ref - x_ref
    return abs(A * y_car + B * x_car + C) / math.sqrt(A ** 2 + B ** 2)


def J(state, path, t_param):
    x_ref = path[0] * t_param ** 3 + path[1] * t_param ** 2 + path[2] * t_param + path[0]
    y_ref = t_param
    e_c_hat = dist_line_from_point(path_deriv(path, y_ref), y_ref, x_ref, state[1], state[0])
    e_l_hat = dist_line_from_point(1 / path_deriv(path, y_ref), y_ref, x_ref, state[1], state[0])
    return q_c * e_c_hat ** 2 + q_l * e_l_hat ** 2 - gamma * V_max


def R(command):
    return command @ cost_R @ command


def L(dyn_m, command):
    beta_dyn = math.atan2(dyn_m.v_y, dyn_m.v_x)
    beta_kin = math.atan2(math.tan(command[1]) * L_REAR, (L_REAR + L_FRONT))
    return q_beta * (beta_kin - beta_dyn) ** 2


def C(slack):
    return q_s * slack + q_ss * (slack ** 2)


def do_step(integrator: Integration, command):
    integrator.RK4(command[1], command[0])


def step_cost(integrator: Integration, command, path, slack):
    return J(integrator.state, path, integrator.t_param) + R(command) + L(integrator.dyn_m, command) + C(slack)


def constraints_cost_calc(integrator: Integration, command, slack, path):
    constraints_cost = 0
    x_ref = path[0] * integrator.t_param ** 3 + path[1] * integrator.t_param ** 2 + path[2] * integrator.t_param + path[0]
    y_ref = integrator.t_param
    # checking the following constraint:
    # (integrator.state.x - x_ref)**2 + (integrator.state.y - y_ref)**2 > R_track**2 + slack
    # this tests if the vehicle is still withing the track boundaries
    constraints_cost -= math.log(
        (R_track ** 2 + slack) - ((integrator.state[integrator.x] - x_ref) ** 2 + (integrator.state[integrator.y] - y_ref) ** 2))

    # checking the following constraint:
    # command[0] > 1 or command[0] < -1 or command[1] > max_delta or command[1] < -max_delta
    # this tests if the command given is within the physical limits of the vehicle
    constraints_cost -= math.log(abs(1 - command[0]))  # gas/break command
    constraints_cost -= math.log(abs(max_delta - command[1]))  # gas/break command

    # checking the following constraint:
    # (F_r,y)^2 + (p_long * F_r,x)^2 < (p_ellipse * D_r)^2
    # (F_f,y)^2 + (p_long * F_f,x)^2 < (p_ellipse * D_f)^2
    # this checks that we do not exceed the elliptic force budget of the tires
    constraints_cost -= math.log(((p_ellipse * D_R) ** 2 - integrator.dyn_m.rear_tire_force_y(
        integrator.dyn_m.rear_slip_angle(integrator.dyn_m.v_y, integrator.dyn_m.v_x)) ** 2 + (
                                              p_long * integrator.dyn_m.tire_force_x_(command[0],
                                                                                      integrator.dyn_m.v_x)) ** 2))  # rear wheels
    constraints_cost -= math.log(((p_ellipse * D_F) ** 2 - integrator.dyn_m.front_tire_force_y(
        integrator.dyn_m.front_slip_angle(integrator.dyn_m.v_y, integrator.dyn_m.v_x, command[1])) ** 2 + (
                                              p_long * integrator.dyn_m.tire_force_x_(command[0],
                                                                                      integrator.dyn_m.v_x)) ** 2))  # front wheels

    return constraints_cost


def total_cost_calc(inputs, state, path, horizon=static_horizon):
    """
        params:
        -------------
        state - np.array, vehicle state after blend
        inputs - horizon * 3 1-D np.ndarray, commands for cost calc
                 after the reshape it will look like this so that the function can calulate properly
                    inputs[0][i] - D
                    inputs[1][i] - delta
                    inputs[2][i] - slack
        path - np.array holding the parameters of the path polynomial

        return:
        --------------
        total_cost - cost of given command matrix
        horizon_state - last state to be in after horizon drive
    """
    # this is done so that we enter a single 1-D with shape(horizon*3,) into the minimize function
    inputs = np.reshape(inputs, (3, horizon))
    commands = inputs[:2, :]
    slack = inputs[2, :]
    
    total_cost = 0
    t_param = 0
    integrator = Integration(state=state)
    # this calc is done while no new state from state estimation was given
    for step in range(horizon):
        step_c = constraints_cost_calc(integrator, commands[:, step], slack[step], path)
        step_c += step_cost(integrator, commands[:, step], path, slack[step])
        total_cost += step_c
        do_step(integrator, commands[:, step])

    return total_cost
