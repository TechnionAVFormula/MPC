import abc
import torch
from torch import Tensor
import json
import numpy as np
from Integration import Integration

from pathlib import Path

OPT_PARAMS = json.loads(open(Path("config") / "opt_params.json", "r").read())

horizon = OPT_PARAMS["horizon"]


class Optimizer(abc.ABC):
    """
    Base class for optimizers.
    """

    def __init__(self, params):
        """
        :param params: A sequence of model parameters to optimize. Can be a
        list of (param,grad) tuples as returned by the Blocks, or a list of
        pytorch tensors in which case the grad will be taken from them.
        """
        assert isinstance(params, list) or isinstance(params, tuple)
        self._params = params

    @property
    def params(self):
        """
        :return: A sequence of parameter tuples, each tuple containing
        (param_data, param_grad). The data should be updated in-place
        according to the grad.
        """
        returned_params = []
        for x in self._params:
            if isinstance(x, Tensor):
                p = x.data
                dp = x.grad.data if x.grad is not None else None
                returned_params.append((p, dp))
            elif isinstance(x, tuple) and len(x) == 2:
                returned_params.append(x)
            else:
                raise TypeError(f"Unexpected parameter type for parameter {x}")

        return returned_params

    def zero_grad(self):
        """
        Sets the gradient of the optimized parameters to zero (in place).
        """
        for p, dp in self.params:
            dp.zero_()

    @abc.abstractmethod
    def step(self):
        """
        Updates all the registered parameter values based on their gradients.
        """
        raise NotImplementedError()


class MomentumSGD(Optimizer):
    def __init__(self, params, learn_rate=1e-3, reg=0, momentum=0.9):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate
        :param reg: L2 Regularization strength
        :param momentum: Momentum factor
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.reg = reg
        self.momentum = momentum

        self.prev_v = [0.0] * len(self.params)
        self.param_idx = 0

    def step(self):
        for p, dp in self.params:
            if dp is None:
                continue

            if self.param_idx >= len(self.params):
                self.param_idx = 0

            dp += self.reg * p

            curr_v = self.momentum * self.prev_v[self.param_idx] - self.learn_rate * dp
            p += curr_v

            self.prev_v[self.param_idx] = curr_v
            self.param_idx += 1


class DMD(Optimizer):
    def __init__(self, params, min_func, step_func, control_dist, state_uncertainty, sub_opt_depth, path,
                 learn_rate=1e-3):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate vector (size = horizon)
        :param min_func: the function of the loss calc for a given action vector
                input: a given vector of action, a start state
                output: the total loss value of the vectors actions
        :param step_func: the function of the step optimization algorithm
                input: an action vector and its loss (for grad calc)
                output: a changed action vector
        :param control_dist: Control distribution to sample the commands from (theta will be its mean)
                input: a vector of theta
                output: a same size vector of chosen actions
        :param state_uncertainty: Variance of the initial state that will be used for sampling in later steps as well
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.sub_opt_depth = sub_opt_depth
        # TODO: add switch case instead the next 2 fields
        #   with different options chosen by the combination of breg div and loss func
        self.min_func = min_func
        self.step_func = step_func

        self.control_dist = control_dist
        self.sub_horizon = int(np.floor(horizon / 10))
        self.x = [0.0] * 6
        self.theta_idx = 0
        self.path = path
        self.state_uncertainty = state_uncertainty
        self.integrator = Integration(self.x)

        # for debug / measuring purposes
        self.loss_log = []

    def update_state(self, state):
        self.x = state

    def update_path(self, path):
        self.path = path

    def find_minima(self, theta_tilda, slack_tilda, x):
        theta = theta_tilda
        theta.retain_grad()
        slack = slack_tilda
        slack.retain_grad()
        loss_per_iteration = torch.zeros(self.sub_opt_depth)
        # loss = torch.zeros(1, requires_grad=True)

        for i in range(self.sub_opt_depth):
            # update the loss tensor
            loss_per_iteration[i] = self.min_func(theta, x, self.path, slack, self.integrator.t_param)

            # check if we reached a local minima already.
            # checks by comparing to the curr loss to the average of the last 3 losses.
            with torch.no_grad():
                if i > 3 and np.average(loss_per_iteration[i - 3:i]) <= loss_per_iteration[i]:
                    break
            # update the theta vector
            loss_per_iteration[i].backward(retain_graph=True)
            theta, slack = self.step_func(theta, slack, self.control_dist, self.learn_rate, x, self.path, self.integrator.t_param)
            # theta and slack are new references to non-leaf Variables. Need to enable .grad manually again
            theta.retain_grad()
            slack.retain_grad()

        # for logging purposes
        self.loss_log.append(loss_per_iteration)

        return theta, slack

    # do Algorithm 1 step
    def step(self):
        # the final action vector
        theta = torch.zeros([2, horizon + self.sub_horizon], requires_grad=True)
        slack = torch.zeros([1, horizon + self.sub_horizon], requires_grad=True)

        # u[0,:] - D, u[1,:] - delta
        u = torch.zeros([2, horizon + self.sub_horizon])
        # first state
        x_curr = self.x
        # update the integrator to the current state
        self.integrator.realign(x_curr)

        for t in range(horizon):
            theta_tilda = theta[:, t:t + self.sub_horizon]
            slack_tilda = slack[0, t:t + self.sub_horizon]
            # update the theta vector number of times -> full optimization for sub horizon
            # for loop with an exit if, of optimization steps to be closer to full optimization for sub horizon.
            # exit loop if you reached local minima before the end of the loop
            # FIXME: naming
            temp_theta, temp_slack = self.find_minima(theta_tilda, slack_tilda, x_curr)
            with torch.no_grad():
                theta[:, t:t + self.sub_horizon], slack[0, t:t + self.sub_horizon] = temp_theta, temp_slack

            # choose an action vector from the distribution and theta param
            u[:, t:t + self.sub_horizon] = self.control_dist(theta[:, t:t + self.sub_horizon])

            # choose an state estimation with regard to the chosen action and state normal distribution
            self.integrator.RK4(u[1, t], u[0, t])
            mean = self.integrator.state
            t_param = self.integrator.t_param
            x_curr = np.random.normal(mean, self.state_uncertainty)
            # update the integrator to the current state
            self.integrator.realign(x_curr, t_param)

            # shift theta
            #   - most of it is done automatically by the for loop, need to fill the last cell, now zero
            if t < horizon - 1:  # to not go out of range for theta, the shift in the last ran is not needed.
                with torch.no_grad():
                    theta[:, t + self.sub_horizon + 1] = theta[:, t + self.sub_horizon]
                    slack[0, t + self.sub_horizon + 1] = slack[0, t + self.sub_horizon]

        return u
