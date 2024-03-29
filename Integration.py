import numpy as np
import math
from .KinModel import KinModel
from .dynamic_model import DynamicState, Order
import json

VEHICLE_DATA = json.loads(open("vehicle_data.json", "r").read())


Vx_blend_max = VEHICLE_DATA["Vx_blend_max"]
Vx_blend_min = VEHICLE_DATA["Vx_blend_min"]


class Integration(Order):
    def __init__(self, state, t_param=0, dt=0.1):
        """
        parameters:
        ---------------
        defult integration time 
        dt = 0.1 [sec]
        method variables:
        k1, k2, k3, k4 - partial itegration steps in Runge Kutta 4
        equations:
        k1 = f(X_k, u)
        k2 =  f(X_k + dt * k1 / 2 , u)
        k3 =  f(X_k + dt * k2 / 2 , u)
        k4 =  f(X_k + dt * k3  , u)
        
        State_{i+1} = State_{i} + 1 / 6 * dt (k1 + 2 * k2 + 2 * k3 + k4) 
        """

        super().__init__()

        self.dt = dt
        self.kin_m = KinModel(state)
        self.dyn_m = DynamicState(state)
        self.state = state
        self.t_param = t_param

    def RK4(self, delta, D):
        """
        parameters
        -------------------
        delta - Steering angle
        D     - Driving command 
        """

        k1 = self.blend(self.state, delta, D)
        k2 = self.blend(self.state + self.dt * k1 / 2, delta, D)
        k3 = self.blend(self.state + self.dt * k2 / 2, delta, D)
        k4 = self.blend(self.state + self.dt * k3, delta, D)

        self.state = self.state + 1 / 6 * self.dt * (k1 + 2 * k2 + 2 * k3 + k4)

        self.t_param += self.dt * math.sqrt(self.state[self.v_x]**2 + self.state[self.v_y]**2)

        # k1 = self.model.state_derivative(self.model.State, delta, D)
        # k2 = self.model.state_derivative(self.model.State + self.dt * k1 / 2, delta, D)
        # k3 = self.model.state_derivative(self.model.State + self.dt * k2 / 2, delta, D)
        # k4 = self.model.state_derivative(self.model.State + self.dt * k3, delta, D)

        # self.model.State = self.model.State + 1 / 6 * self.dt * (k1 + 2 * k2 + 2 * k3 + k4)

    def lambd(self, Vx):
        return min(max((Vx - Vx_blend_min) / (Vx_blend_max - Vx_blend_min), 0), 1)

    def blend(self, state, delta, D, time_delta=0.1):
        if (state[3]**2+state[4]**2)**0.5 < Vx_blend_min:
            kin_state_deriv = self.kin_m.state_derivative(state, delta, D,time_delta)
            return kin_state_deriv
        if (state[3]**2+state[4]**2)**0.5 > Vx_blend_max:
            dyn_state_deriv = self.dyn_m.state_derivative(state, delta, D)#TODO: find negative x speed = (constants?)
            return dyn_state_deriv
        
        dyn_state_deriv = self.dyn_m.state_derivative(state, delta, D)
        kin_state_deriv = self.kin_m.state_derivative(state, delta, D,time_delta)
        return np.array(self.lambd(dyn_state_deriv[0]) * dyn_state_deriv) + np.array(
            (1 - self.lambd(kin_state_deriv[0])) * kin_state_deriv)


def main():
    pass


if __name__ == "main":
    main()
