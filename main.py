from sim import Sim
from optimizers import DMD
from DescentFunctions import gd_step, gd_loss
import torch


def normal_dist_vec(vec, mean=0, std=0.5):
    return torch.normal(mean, std, size=(2, int(len(vec)/2)), generator=None, out=None)


def main():
    # test the DMD
    torch.autograd.set_detect_anomaly(True)
    params = []
    min_func = gd_loss
    step_func = gd_step
    control_dist = normal_dist_vec
    state_uncertainty = 0.5
    sub_opt_depth = 10
    path_str_center = [0, 0, 1, 0]
    path_str_side = [0, 0, 1, 0.5]
    path_par_center = [0, 1, 0, 0]
    dmd = DMD(params, min_func, step_func, control_dist, state_uncertainty, sub_opt_depth, path_str_center,
              learn_rate=1e-3)
    commands = dmd.step()

    from pathlib import Path
    import _pickle as pickle
    from datetime import datetime

    out_folder = Path('out')
    now = datetime.now()

    with open(out_folder / f"commands_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}", "wb") as f:
        pickle.dump(commands, f)

    with open(out_folder / f"commands_latest", "wb") as f:
        pickle.dump(commands, f)


if __name__ == "__main__":
    main()


