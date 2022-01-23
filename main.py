from sim import Sim
from optimizers import DMD
from DescentFunctions import gd_step, gd_loss
import torch
import numpy as np

import matplotlib.pyplot as plt


def normal_dist_vec(vec, mean=0, std=0.5):
    return torch.normal(mean, std, size=(2, int(len(vec)/2)), generator=None, out=None)


def main():
    from time import perf_counter
    # test the DMD
    torch.autograd.set_detect_anomaly(True)
    params = []
    min_func = gd_loss
    step_func = gd_step
    control_dist = normal_dist_vec
    state_uncertainty = 0.5
    sub_opt_depth = 10
    path_str_center = [0, 0, 0, 0]
    path_str_side = [0, 0, 1, 0]
    path_par_center = [0, 0.001, 0.0001, 0]
    start = perf_counter()
    dmd = DMD(params, min_func, step_func, control_dist, state_uncertainty, sub_opt_depth, path_par_center,
              learn_rate=0.35)
    commands = dmd.step()
    end = perf_counter()
    print(f'ran {end-start} s ')
    # plot the optimization graph
    # plt.plot(dmd.loss_log, range(len(dmd.loss_log)))
    from itertools import chain
    losses = [loss.item() for loss in chain.from_iterable(dmd.loss_log)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 401, 40)
    minor_ticks = np.arange(0, 401, 10)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    plt.plot(losses)
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.show()

    plt.plot(losses[10:20])
    plt.grid()
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.show()

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


