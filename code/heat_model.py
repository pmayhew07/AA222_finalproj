''' - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    Simple Transient Heat Transfer Solver Code
    Parker Mayhew and Peter Krenek (Stanford AA222 Final Project)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - '''

## Import packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from helpers import *

# - - - - - - - - - - - - - - - #
#          EQN SETUP            #
# - - - - - - - - - - - - - - - #

## FEA method for abblative layer of heat shielf
def heatshield(Tinitial, Tmax, Tapplied, L, k, rho, c):
    # Compute heat penetration depth
    alpha = k/(rho*c) # Thermal diffusivity
    theta = (Tmax - Tapplied)/(Tinitial - Tapplied) # Percentage temperature rise
    delta_coeff = 2*erfinv(theta)
    delta = delta_coeff*np.sqrt(L*alpha) # Penetration depth

    # Compute penetration time
    t_max = 1/(delta_coeff**2) * L**2/alpha
    return t_max


if __name__ == '__main__':
    T_atm = 3033
    ablator = ablators["PICA"]
    conductor = conductors["SS"]
    conductor_Ti = 273; conductor_maxT = maxtemp(conductor)
    payload_Ti = 295; payload_maxT = 325;
    t1, t2 = (0.009122831324789719, 0.022480528762798983)
    print(totalcost(t1, t2, ablator, conductor))
    t1, t2 = (0.0092, 0.02234)
    print(totalcost(t1, t2, ablator, conductor))
    t1, t2 = (0.008847289350818495, 0.02299)
    print(totalcost(t1, t2, ablator, conductor))

    kb, rhob, cb = blend(t1, t2, ablator, conductor)
    L = t1+t2; k = kb; rho = rhob; c = cb
    t_max = heatshield(payload_Ti, payload_maxT, T_atm, L, k, rho, c)
    print("Max flight duration = " + str(round(t_max, 10)))