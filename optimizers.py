import abc
import torch
from torch import Tensor
import json

OPT_PARAMS = json.loads(open("opt_params.json", "r").read())

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
    def __init__(self, params, loss_func, breg_div, control_dist, state_uncertainty, learn_rate=1e-3):
        """
        :param params: The model parameters to optimize
        :param learn_rate: Learning rate vector (size=horizon)
        :param loss_func: Function to execute on the cost calculated
        :param breg_div: Bregman divergence for update of theta through MD step
        :param control_dist: Control distribution to sample the commands from (theta will be its mean)
        :param state_uncertainty: Variance of the initial state that will be used for sampling in later steps as well
        """
        super().__init__(params)
        self.learn_rate = learn_rate
        self.loss_func = loss_func
        self.breg_div = breg_div
        self.control_dist = control_dist

        self.sub_horizon = torch.floor(horizon/10)
        self.prev_x = [0.0] * 6
        self.theta = Tensor([0.0, 0.0] * self.sub_horizon, requires_grad=True)
        self.theta_idx = 0

    # do Algorithm 1 step
    def step(self):
        pass
