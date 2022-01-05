import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import math

V_MAX = 300
VEHICLE_LEN_X = 1
VEHICLE_LEN_Y = 1


class Sim:
    def __init__(self, right_edge, left_edge, commands, road_length=100, dt=0.5):
        self.right_edge = right_edge
        self.left_edge = left_edge
        self.commands = commands
        self.steps_x = []
        self.steps_y = []
        self.road_length = road_length
        self.dt = dt
        self.v = []
        self.ang = []

    def __calc_edges(self):
        x_edge = np.linspace(0, 10*self.road_length, math.floor(self.road_length / self.dt))
        y_right_edge = self.right_edge[0] * x_edge ** 3 + self.right_edge[1] * x_edge ** 2 + self.right_edge[
            2] * x_edge + self.right_edge[3]
        y_left_edge = self.left_edge[0] * x_edge ** 3 + self.left_edge[1] * x_edge ** 2 + self.left_edge[2] * x_edge + \
                      self.left_edge[3]

        return x_edge, y_right_edge, y_left_edge

    @staticmethod
    def __convert_d_to_speed(d, ang, old_v_x, old_v_y):
        v = math.sqrt(old_v_x ** 2 + old_v_y ** 2)
        v_x = old_v_x + math.sin(ang) * d * max(V_MAX - v, 0)  # TODO: change the speed change to linear
        v_y = old_v_y + math.cos(ang) * d * max(V_MAX - v, 0)

        return v_x, v_y

    def __calc_driving_steps(self):
        # starting point
        self.steps_x.append(0)
        self.steps_y.append(0)
        self.v.append(0)
        self.ang.append(0)
        v_x = 0
        v_y = 0

        # calc next point by analysing the command.
        # for i in range(len(self.commands)):

        # for ang, D in self.commands:
        for D, ang in self.commands:
            # ang = math.radians(ang_deg)
            v_x, v_y = self.__convert_d_to_speed(D, ang, v_x, v_y)
            self.steps_x.append(self.steps_x[-1] + v_x * self.dt)
            self.steps_y.append(self.steps_y[-1] + v_y * self.dt)
            self.v.append(math.sqrt(v_x ** 2 + v_y ** 2))
            self.ang.append(ang)

        return

    def calc_score(self):
        # calc driving steps
        self.__calc_driving_steps()
        pass

    def plot_final_drive(self):
        # calc and plot the edges of the road
        y_edge, x_right_edge, x_left_edge = self.__calc_edges()
        plt.plot(x_right_edge, y_edge, marker='', color='gray', linewidth=4)
        plt.plot(x_left_edge, y_edge, marker='', color='gray', linewidth=4)

        # calc and plot the driving steps
        self.__calc_driving_steps()
        plt.plot(self.steps_x, self.steps_y, marker='.', markerfacecolor='blue', markersize=12, color='skyblue',
                 linewidth=2)
        plt.plot(self.steps_x[0], self.steps_y[0], marker='.', markerfacecolor='red', markersize=12, color='skyblue',
                 linewidth=2)
        plt.plot(self.steps_x[-1], self.steps_y[-1], marker='.', markerfacecolor='green', markersize=12, color='skyblue',
                 linewidth=2)
        # plot cosmetics and show
        plt.grid()
        plt.show()

        return

    def plot_drive_steps(self):
        # calc and plot the edges of the road
        y_edge, x_right_edge, x_left_edge = self.__calc_edges()
        plt.plot(x_right_edge, y_edge, marker='', color='olive', linewidth=4)
        plt.plot(x_left_edge, y_edge, marker='', color='olive', linewidth=4)

        self.__calc_driving_steps()

        for i in range(len(self.commands)):
            # plot the 0-i steps of the drive
            plt.plot(self.steps_x[:i], self.steps_y[:i],
                     marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=2)

            # add a vector of the vehicle's speed and driving angle (for now the same as wheel angle)
            plt.arrow(x=self.steps_x[i - 1], y=self.steps_y[i - 1],
                      dx=self.v[i - 1] * math.sin(self.ang[i - 1]),
                      dy=self.v[i - 1] * math.cos(self.ang[i - 1]))

            # add a rectangle as the vehicle
            x = self.steps_x[i - 1] - 0.5 * VEHICLE_LEN_X
            y = self.steps_y[i - 1] - 0.5 * VEHICLE_LEN_Y
            plt.Rectangle(xy=(x, y), width=VEHICLE_LEN_X, height=VEHICLE_LEN_Y, angle=self.ang[i - 1])

            # display plot
            plt.grid()
            plt.show()

        return


if __name__ == '__main__':
    from pathlib import Path
    import _pickle as pickle

    out_folder = Path('out')

    with open(out_folder / "commands_latest", 'rb') as f:
        commands = pickle.load(f)

    # test - drive straight as fast as you can, check with sim
    right_edge = [0, 0, 0, 3]
    left_edge = [0, 0, 0, -3]
    # commands = []
    # for i in range(10):
    #    commands.append([0.25, 0.0])

    sim = Sim(right_edge=right_edge, left_edge=left_edge, commands=commands)
    sim.plot_final_drive()
    # sim.plot_drive_steps()
