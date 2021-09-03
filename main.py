import math
import numpy as np
import scipy.optimize as opt

import cost
from Integration import Integration

import json
from pathlib import Path

VEHICLE_DATA = json.loads(open(Path("config") / "vehicle_data.json", "r").read())
OPT_PARAMS = json.loads(open(Path("config") / "opt_params.json", "r").read())

horizon = OPT_PARAMS["horizon"]
max_delta = VEHICLE_DATA["maxAlpha"]

def main_runner():
    init_state = np.zeros(6)
    path = np.array([-0.00001, 0, 0.1, 0])
    integrator = Integration(init_state)
    step = 0
    while step < 3:
        print("state before opt: ", integrator.state, "\n")
        lb_D = -1 * np.ones(horizon)
        lb_delta = -max_delta * np.ones(horizon)
        lb_slack = -np.inf * np.ones(horizon)
        lb = np.concatenate((lb_D, lb_delta, lb_slack), axis=None)
        ub_D = np.ones(horizon)
        ub_delta = max_delta * np.ones(horizon)
        ub_slack = np.inf * np.ones(horizon)
        ub = np.concatenate((ub_D, ub_delta, ub_slack), axis=None)

        bnds = opt.Bounds(lb, ub)
        res = opt.minimize(cost.total_cost_calc, np.zeros(horizon*3), args=(integrator.state, path, horizon), bounds=bnds)
        if res.success:
            commands = np.reshape(res.x, (3, horizon))[:2, :]
            print("commands: ", commands, "\n")
            slack = np.reshape(res.x, (3, horizon))[2, :]
            print("slack: ", slack, "\n")
            
            print("min cost: ", cost.total_cost_calc(res.x, integrator.state, path), "\n")
            
            cost.do_step(integrator, commands[:, 0])
            print("state after do_step: ", integrator.state, "\n")
            
        else:
            print(res.message)
            break

        step += 1


if __name__ == "__main__":
    main_runner()
