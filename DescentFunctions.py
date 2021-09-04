import json
from pathlib import Path

from cost import total_cost_calc

OPT_PARAMS = json.loads(open(Path("config") / "opt_params.json", "r").read())

sigma_sqr_gd = OPT_PARAMS["sigma_sqr_gd"]


############ Gradient Descent functions #############

# loss = self.min_func(theta, x)
# theta, slack = self.step_func(theta, slack, loss)


def gd_loss(theta, state, path, slack, t_param):
    total_cost, stat, steps_cost, t = total_cost_calc(state, theta, slack, path, theta.shape[1], t_param,
                                                      new_state=True)
    return total_cost


def gd_step(theta, slack, control_dist, learn_rate, state=None, path=None, t_param=0):
    g_t = 0
    theta_dot = theta.grad
    for i in range(10):
        u = control_dist(theta)
        g_t += (theta - u) * theta_dot * gd_loss(u, state, path, slack, t_param) / sigma_sqr_gd

    # with torch.no_grad():
    theta = theta - (learn_rate * g_t / 10)

    slack_dot = slack.grad
    slack = slack - learn_rate * slack_dot

    return theta, slack
