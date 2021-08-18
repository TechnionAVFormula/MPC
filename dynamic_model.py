"""
Bicycle Model: Dynamic version

"""

import math
import numpy as np
import json
from math import cos, sin, atan2

from pathlib import Path

VEHICLE_DATA = json.loads(open(Path("config") / "vehicle_data.json", "r").read())

L = VEHICLE_DATA["wheel_base"]  # Total length
L_REAR = VEHICLE_DATA["Rear_length"]  # rear length
L_FRONT = L - L_REAR  # front length
MASS = VEHICLE_DATA["Vehicle_weight"]
I_z = VEHICLE_DATA["Vehicle_inertia_moment"]

# Magic Formula
# Rear wheel
B_R = VEHICLE_DATA["Magic_B_rear"]
C_R = VEHICLE_DATA["Magic_C_rear"]
D_R = VEHICLE_DATA["Magic_D_rear"]

# Front wheel
B_F = VEHICLE_DATA["Magic_B_front"]
C_F = VEHICLE_DATA["Magic_C_front"]
D_F = VEHICLE_DATA["Magic_D_front"]

# motor model
C_M = VEHICLE_DATA["Motor_model"]

# shape parameter
C_R_0 = VEHICLE_DATA["Rolling_resistance"]
C_R_2 = VEHICLE_DATA["Drag"]

"""
parameters of electrical vehicle!!!!

"""
# engine parameter
P_TV = VEHICLE_DATA["Proportional_gain_torque"]


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


class DynamicState(Order):
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

    def state_derivative(self, State, delta, D):
        """
        parameters
        -------------------
        input:
        delta - Steering angle
        D     - Driving command 

        method variables:
        x_dot   - Inertial x direction velocity   
        y_dot   - Inertial y direction velocity   
        phi_dot - Inertial angular velocity
        a_x     - Longitudinal acceleration
        a_y     - Lateral acceleration
        r_dot   - Yaw angular acceleration


        rear_slip_angle_        - rear slip angle
        front_slip_angle_       - front slip angle
        rear_tire_force_y_      - lateral rear tire force 
        front_tire_force_y_     - lateral fron tire force
        tire_force_x_           - longitudinal tire force

        """
        phi = State[self.phi]
        v_x = State[self.v_x]
        v_y = State[self.v_y]
        r = State[self.r]

        rear_slip_angle_ = self.rear_slip_angle(v_y, v_x)
        front_slip_angle_ = self.front_slip_angle(v_y, v_x, delta)
        rear_tire_force_y_ = self.rear_tire_force_y(rear_slip_angle_)
        front_tire_force_y_ = self.front_tire_force_y(front_slip_angle_)
        tire_force_x_ = self.tire_force_x_(D, v_x)
        torque_moment_ = self.torque_moment(v_x, r, delta)

        x_dot = v_x * cos(phi) - v_y * sin(phi)
        y_dot = v_x * sin(phi) + v_y * cos(phi)
        phi_dot = r
        a_x = (
                1
                / MASS
                * (tire_force_x_ - front_tire_force_y_ * sin(delta) + MASS * v_y * r)
        )
        a_y = (
                1
                / MASS
                * (rear_tire_force_y_ + front_tire_force_y_ * cos(delta) - MASS * v_x * r)
        )

        r_dot = (
                1
                / I_z
                * (
                        front_tire_force_y_ * L_FRONT * cos(delta)
                        - rear_tire_force_y_ * L_REAR
                        + torque_moment_
                )
        )

        return np.array([x_dot, y_dot, phi_dot, a_x, a_y, r_dot])

    @staticmethod
    def rear_slip_angle(v_y, v_x):
        rear_slip_angle_ = atan2(v_y - L_REAR, v_x)
        return rear_slip_angle_

    @staticmethod
    def front_slip_angle(v_y, v_x, delta):
        front_slip_angle_ = atan2(v_y + L_FRONT, v_x) - delta
        return front_slip_angle_

    @staticmethod
    def rear_tire_force_y(rear_slip_angle):
        rear_tire_force_y_ = D_R * sin(C_R * atan2(B_R * rear_slip_angle, 1))
        return rear_tire_force_y_

    @staticmethod
    def front_tire_force_y(front_slip_angle):
        front_tire_force_y_ = D_F * sin(C_F * atan2(B_F * front_slip_angle, 1))
        return front_tire_force_y_

    @staticmethod
    def tire_force_x_(D, v_x):
        tire_force_x_ = C_M * D - C_R_0 - C_R_2 * v_x ** 2
        return tire_force_x_

    @staticmethod
    def torque_moment(v_x, r, delta):
        r_target = delta * v_x / L
        torque_moment_ = (r_target - r) * P_TV
        return torque_moment_


def main():
    pass


if __name__ == "main":
    main()
