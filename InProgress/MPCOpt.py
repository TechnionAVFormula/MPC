"""
Model predictive controller 

suggested optimizer: casadi 

"""

import casadi
import sys

sys.path.insert(0, "./..")
from MPC.dynamic_model import DynamicState


class MpcOpt(DynamicState):
    def __init__(
        self,
        QC=1.0,
        QL=1.0,
        GAMMA=1.0,
        R=1.0,
        RU=[1.0, 1.0, 1.0],
        RDU=[1.0, 1.0, 1.0],
        HORIZON=10,
        dt=0.1,
    ):
        """
        inputs:
        ----------------
        J(x_k) = (qc * e_ck) ** 2 + (ql * e_lk) **2 - (gamma * v_tk)
            QC: perpendicular error weight
            QL: longitudinal error weight
            GAMMA: path velocity weight

        (X - X_cen) ** 2 + (Y - Y_cen) ** 2 <= R ** 2
            R: safety radius 

        L = q_beta * (beta_kin - beta_dyn) ** 2
            q_beta: squared error between dynamic and kinematic slip angle weight

        R(uk,duk) = uk^T @ R_u @ uk + duk^T @ R_du @ duk
        RU: control efforts weight
        RDU: control change ratio weight

        horizon: prediction steps 

        path: objective path 

        Members:
        ------------
        
        Matrics: sub class of matrics
        ------------------------------
        matrics.ru: control efforts weight 
        matrics.rdu: control change ratio weight 
        matrics.q:  error weights
        ------------------------------

        """
        self.time = dt
        self.matrics = self.Matrics()
        self.matrics.ru = casadi.diag(RU)
        self.matrics.rdU = casadi.diag(RDU)
        self.matrics.q = casadi.diag([QC, QL, GAMMA])
        self.horizon = HORIZON
        self.R = R

        self.opti = casadi.Opti()  # import opti stack

        self.x_current = self.opti.parameter(10)
        # current extended state: {X_0, Y_0, phi_0, vx_0, vy_0, r_0}
        # path velocity: {v_theta}
        # control efforts: {delta, D, DV_theta}

        self.path_ref = self.opti.parameter(2, self.horizon)  # The ref trajectory
        self.x_ref = self.path_ref[1, :]
        self.y_ref = self.path_ref[2, :]

        self.state_predict = self.opti.variable(10, self.horizon)
        self.delta_u = self.opti.variable(3, self.horizon - 1)

    def _optimization_constraints(self):

        self.opti.subject_to(
            self.state_predict[:, 0] == self.x_current[:]
        )  # initial condition

        for i in range(1, self.horizon):
            # state prediction
            self.opti.subject_to(
                self.state_predict[:6, i]
                == self.state_derivative(
                    self.state_predict[0:6, i - 1],
                    self.state_predict[:, 8],
                    self.state_predict[:, 9],
                )
            )

            # traj velocity prediction
            self.opti.subject_to(
                self.state_predict[6, i]
                == (
                    self.state_predict[6, i - 1]
                    + casadi.dot(
                        (self.path_ref[:, i] - self.path_ref[:, i - 1])
                        / casadi.norm_2(self.path_ref[:, i] - self.path_ref[:, i - 1]),
                        (self.state_predict[3:5, i])
                        / casadi.norm_2(self.state_predict[3:5, i]),
                    )
                    / self.time
                )
            )

            # control efforts state
            self.opti.subject_to(
                self.state_predict[7:, i]
                == (self.state_predict[7:, i - 1] + self.delta_u[:, i - 1] / self.time)
            )

            self.opti.subject_to(
                (self.state_predict(0, i) - self.path_ref[0, i]) ** 2
                + (self.state_predict(1, i) - self.path_ref[1, i]) ** 2
                <= self.R ** 2
            )

            # require the bounding of delta_u and ellipsoidal constraint

        # def _cost_function(self):
        #     cost = 0
        #     for i in range(self.horizon):
        #         cost += casadi.mtimes(,self.matrics.q)

    class Matrics:
        def __init__(self):
            self.ru = None
            self.rdu = None
            self.q = None
