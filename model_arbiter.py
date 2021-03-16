import math 
import numpy as np

def lambda(Vx):
    Vx_blend_max = 5
    Vx_blend_min = 3
    return min(max((Vx-Vx_blend_min)/(Vx_blend_max-Vx_blend_min),0),1)

def blend(self, dyn_m, kin_m):
    return np.array(lambda(dyn_m.x_dot) * dyn_m) + np.array((1 - lambda(kin_m.x_dot)) * kin_m)

