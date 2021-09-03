import scipy.optimize as opt
import numpy as np
import cost
import json
from pathlib import Path

OPT_PARAMS = json.loads(open(Path("config") / "opt_params.json", "r").read())

opt_horizon = OPT_PARAMS["horizon"]

def optimize(state, path, horizon=opt_horizon):
    return opt.minimize(cost.total_cost_calc, np.zeros(horizon*3), args=(state, path, horizon))
