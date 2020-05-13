###
import math as mat
import numpy as np
from numpy.linalg import norm

##Cost function
class Order:
    def __init__(self):
        self.X = int(0)
        self.Y = int(1)
        self.Vx = int(2)
        self.Vy = int(3)
        self.Theta = int(4)
        self.Throttle = int(0)
        self.Steering = int(1)


class MPC(Order):
    def __init__(
        self, Rmax, Initial_Position, Optimal_path, Path_center, Weights, Time_Delta,
    ):
        super().__init__()
        self.Vehicle_Total_Length = 1.535
        self.Vehicle_Rear_Length = 0.7675
        self.Horizon = 2
        self.slip_angle = np.array([])
        self.Control_Command = np.array([])
        self.Rotational_Speed = np.array([])
        self.Time_Delta = Time_Delta
        self.Target_Path = np.array([])
        self.Orthogonal_error = np.array([])
        self.Optimal_path = Optimal_path
        self.State = Initial_Position
        self.Predicted_State = np.empty([5, self.Horizon])
        self.Tanget_Angle = Optimal_path
        self.Weights = Weights
        self.Path_center = Path_center
        self.Rmax = Rmax
        self.Cost = 0

    @property
    def State(self):
        return self.__State

    @property
    def Tanget_Angle(self):
        return self.__Tanget_Angle

    @property
    def Target_Path(self):
        return self.__Target_Path

    @State.setter
    def State(self, State):
        self.__State = State

    @Target_Path.setter
    def Target_Path(self, Target_Path):
        self.__Target_Path = Target_Path

    @Tanget_Angle.setter
    def Tanget_Angle(self, Predicted_path):
        Tanget_Angle = np.zeros([self.Horizon, 1])
        Tanget_Angle[0] = mat.atan2(
            Predicted_path[0, 1] - self.State[self.Y],
            Predicted_path[0, 0] - self.State[self.X],
        )
        for i in range(1, self.Horizon):
            Tanget_Angle[i] = mat.pi - mat.atan2(
                Predicted_path[1, i] - Predicted_path[1, i - 1],
                Predicted_path[0, i] - Predicted_path[0, i - 1],
            )
        self.__Tanget_Angle = Tanget_Angle

    def Kinematic_Model(self, Control_Command):
        # Calculate the absolute velocity
        V_abs = norm(self.State[[self.Vx, self.Vy]])
        # Calculate The slip angle maybe take it from state!!!
        self.Slip_angle = mat.atan2(
            mat.tan(Control_Command[self.Steering]) * self.Vehicle_Rear_Length,
            self.Vehicle_Total_Length,
        )
        # Calculate the rotational speed maybe take it from state!!!
        self.Rotational_Speed = (
            V_abs
            * mat.cos(self.Slip_angle)
            * mat.tan(Control_Command[self.Steering])
            / self.Vehicle_Total_Length
        )
        # Calculate the process advancement
        advancement = np.array(
            [
                [
                    self.Time_Delta * self.State[self.Vx]
                    + (self.Time_Delta ** 2)
                    / 2
                    * mat.cos(self.State[self.Theta] + self.Slip_angle)
                    * Control_Command[self.Throttle]
                ],
                [
                    self.Time_Delta * self.State[self.Vy]
                    + (self.Time_Delta ** 2)
                    / 2
                    * mat.sin(self.State[self.Theta] + self.Slip_angle)
                    * Control_Command[self.Throttle]
                ],
                [
                    self.Time_Delta
                    * mat.cos(self.State[self.Steering] + self.Slip_angle)
                    * Control_Command[self.Throttle]
                ],
                [
                    self.Time_Delta
                    * mat.sin(self.State[self.Theta] + self.Slip_angle)
                    * Control_Command[self.Throttle]
                ],
                [self.Time_Delta * self.Rotational_Speed],
            ],
            dtype="float",
        )
        # Update the state
        self.State = self.State + advancement

    def Calculate_Target(self, Control_Command):
        Ec = 0.0
        El = 0.0
        Du = 0.0
        self.Control_Command = Control_Command.reshape(
            [int(len(Control_Command) / 2), 2]
        )
        for i in range(self.Horizon):
            # TODO: change angle calculation to polynomial dervitive from future point
            Ec += self.Weights[0] * (
                mat.sin(self.Tanget_Angle[i])
                * (self.Predicted_State[self.X, i] - self.Optimal_path[self.X, i])
                - mat.cos(self.Tanget_Angle[i])
                * (self.Predicted_State[self.Y, i] - self.Optimal_path[self.Y, i])
            )
            El += self.Weights[1] * (
                -mat.cos(self.Tanget_Angle[i])
                * (self.Predicted_State[self.X, i] - self.Optimal_path[self.X, i])
                - mat.sin(self.Tanget_Angle[i])
                * (self.Predicted_State[self.Y, i] - self.Optimal_path[self.Y, i])
            )
            Du += self.Weights[4] * (
                self.Control_Command[i + 1, 0] - self.Control_Command[i, 0]
            ) + self.Weights[5] * (
                self.Control_Command[i + 1, 1] - self.Control_Command[i, 1]
            )
        self.Cost = (
            Ec ** 2
            + El ** 2
            + Du ** 2
            + self.Weights[3]
            * np.dot(self.Control_Command[:, 0], self.Control_Command[:, 0])
            + self.Weights[4]
            * np.dot(self.Control_Command[:, 0], self.Control_Command[:, 1])
        )
        return self.Cost

    def Constraint(self, Control_Command):
        self.Control_Command = Control_Command.reshape(
            [int(len(Control_Command) / 2), 2]
        )
        self.State = np.zeros([5, 1])
        for i in range(self.Horizon):
            self.Kinematic_Model(self.Control_Command[i, :])
            self.Predicted_State[:, i : i + 1] = self.State
        for i in range(self.Horizon):
            if (
                ((self.Predicted_State[self.X, i] - self.Path_center[self.X, i]) ** 2)
                + (self.Predicted_State[self.Y, i] - self.Path_center[self.Y, i]) ** 2
            ) <= self.Rmax ** 2:
                return 0
        return 1

    def optimize(self, Initial_state):
        # TODO: call engine for optimize. Initial_state from state estimation return optimal control vector.
        pass


##RunningFunciton
# Rmax = 0.5
# Initial_Position = np.zeros([5, 1])
# # Optimal_path = ......!!
# # For first running we will check Path_center = Optimal_path
# Optimal_path = np.array([[5, 10, 15, 20], [5, 10, 15, 20]])
# Path_center = Optimal_path
# Weights = np.ones([6, 1])
# Time_Delta = 1
# ##Genetic initialization
# Number_of_Candidate = 10
# # control efforts
# Val_max = np.array([4, mat.pi / 4, 4, mat.pi / 4, 4, mat.pi / 4])
# val_min = np.array([-2, -mat.pi / 4, -2, -mat.pi / 4, -2, -mat.pi / 4])
# resulotion = 0.01
# # control arguments gas and steering
# contorlarguments = 6


# K = DNA(100, np.array([4, 4]), np.array([-2, -2]), 0.1, 2)
# K.Calculate_NumberofBits()
# K.initial_Parent_List()
# K.Initialize_Population()

##perpendicular distance from the poly
##newton optimization

##optimal path function
##constraints
## initial condition
##kinmatic model
##track bounds
##saturation constraints


##dynamic model
##slip constrints
