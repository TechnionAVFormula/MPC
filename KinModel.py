from math import cos, sin
import numpy as np
import json

from pathlib import Path

VEHICLE_DATA = json.loads(open(Path("config") / "vehicle_data.json", "r").read())

L = VEHICLE_DATA["wheel_base"]  # Total length
L_REAR = VEHICLE_DATA["Rear_length"]  # rear length
L_FRONT = L - L_REAR  # front length
MASS = VEHICLE_DATA["Vehicle_weight"]

# motor model
C_M = VEHICLE_DATA["Motor_model"]

# shape parameter
C_R_0 = VEHICLE_DATA["Rolling_resistance"]
C_R_2 = VEHICLE_DATA["Drag"]


class Order:
    def __init__(self):
        """
        parameters:
        -----------
        pass
        """
        self.x = int(0)
        self.y = int(1)
        self.phi = int(2)
        self.v_x = int(3)
        self.v_y = int(4)
        self.r = int(5)


class KinModel(Order):
    def __init__(self, x=0.0, y=0.0, phi=0.0, v_x=0.0, v_y=0.0, r=0.0):
        """
        parameters:
        ------------
        State = [x, y, phi, v_x, v_y, r]
        x   - Inertial x direction
        y   - Inertial y direction
        phi - Inertial angular orientation
        v_x - Longitudinal velocity
        v_y - Lateral velocity
        r   - angular velocity
        """
        super().__init__()
        self.State = np.array([x, y, phi, v_x, v_y, r], dtype=object)
        # self.car_info = json.loads(open("vehicle_data.json", "r").read())
        self.prev_delta = 0

    def state_derivative(self, State, delta, D, dt=0.1):
        v_x = State[self.v_x]
        v_y = State[self.v_y]
        phi = State[self.phi]
        r = State[self.r]

        f_x = self._tire_force_x(D, v_x)
        delta_dot = self._delta_dot(delta, self.prev_delta, dt)

        x_dot = v_x * cos(phi) - v_y * sin(phi)
        y_dot = v_x * sin(phi) - v_y * cos(phi)
        phi_dot = r
        a_x = f_x / MASS
        a_y = (delta_dot * v_x + delta_dot * a_x) * L_REAR / L
        r_dot = (delta_dot * v_x + delta_dot * a_x) / L

        return np.array([x_dot, y_dot, phi_dot, a_x, a_y, r_dot])

    @staticmethod
    def _tire_force_x(D, v_x):
        tire_force_x_ = C_M * D - C_R_0 - C_R_2 * v_x ** 2
        return tire_force_x_.item()

    @staticmethod
    def _delta_dot(delta, prev_delta, dt):
        _delta_dot = ((delta - prev_delta) / dt)
        return _delta_dot.item()
