"""
Model predictive controller 

suggested optimizer: casadi 

"""

import casadi
import sys

sys.path.insert(0, "./..")
from dynamic_model import DynamicState


class MpcOpt:
    def __init__(
        self,
        QC=1.0,
        QL=1.0,
        GAMMA=1.0,
        R=1.0,
        RU=[1.0, 1.0, 1.0],
        RDU=[1.0, 1.0, 1.0],
        HORIZON=10,
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

        Members:
        ------------
        
        Matrics: sub class of matrics
        ------------------------------
        matrics.ru: control efforts weight 
        matrics.rdu: control change ratio weight 
        matrics.q:  error weights
        ------------------------------

        """
        self.matrics = self.Matrics()
        self.matrics.ru = casadi.diag(RU)
        self.matrics.rdU = casadi.diag(RDU)
        self.matrics.q = casadi.diag([QC, QL, GAMMA])
        self.horizon = HORIZON

        self.opti = casadi.Opti()  # import opti stack

        self.x_current = self.opti.parameter(
            10
        )  # current extended state [state: {X, Y, phi, vx, vy, r}, path velocity: {v_theta},
        # control efforts: {delta, D, V_theta}]

        self.state_predict = self.opti.parameter(10, HORIZON)

        self.path_ref = self.opti.parameter(2, HORIZON)  # The ref trajectory
        self.x_ref = self.path_ref[1, :]
        self.y_ref = self.path_ref[2, :]

    def _optimization_constraints(self):

        self.opti.subject_to(self.state_predict[:, 0] == self.x_current[:])

    class Matrics:
        def __init__(self):
            self.ru = None
            self.rdu = None
            self.q = None
