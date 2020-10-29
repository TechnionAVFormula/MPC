class Integration:
    def __init__(self, model, dt=0.1):
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

        self.dt = dt
        self.model = model

    def RK4(self, delta, D):

        """
        parameters
        -------------------
        input:
        delta - Steering angle
        D     - Driving command 
        """

        k1 = self.model.state_derivative(self.model.State, delta, D)
        k2 = self.model.state_derivative(self.model.State + self.dt * k1 / 2, delta, D)
        k3 = self.model.state_derivative(self.model.State + self.dt * k2 / 2, delta, D)
        k4 = self.model.state_derivative(self.model.State + self.dt * k3, delta, D)

        self.model.State = self.model.State + 1 / 6 * self.dt * (k1 + 2 * k2 + 2 * k3 + k4)


def main():
    pass


if __name__ == "main":
    main()