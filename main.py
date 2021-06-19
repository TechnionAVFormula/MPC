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

    # test - drive straight as fast as you can, check with sim
    right_edge = [0, 0, 0, 3]
    left_edge = [0, 0, 0, -3]
    # commands = []
    # for i in range(300):
    #     commands.append([0., 0.01])

    sim = Sim(right_edge=right_edge, left_edge=left_edge, commands=commands)
    sim.plot_final_drive()


if __name__ == "__main__":
    main()


