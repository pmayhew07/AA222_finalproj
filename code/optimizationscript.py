# imports and libraries
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from heat_model import *
from helpers import *


# Returns objective cost and related info for the passed in materials and thicknesses
# Cost is weighted mass/expense of heatshield, penalized if it does not meet minimum flight time
def constraint(material1, material2, pay_Ti, pay_maxT, cond_Ti, cond_maxT, T_atm, t_combo, flight_time, obj_fxn):
    rho_cost = 1e5 # Cost penalty weight
    rho_mass = 1e3 # Mass penalty weight
    feasible = False

    # evaluate the objective function at the observation points
    if obj_fxn == "cost":
        w1 = 1; w2 = 0
        cost_obs, mass_obs, obj_obs = weightedcost(t_combo[0], t_combo[1], material1, material2, w1, w2)
    elif obj_fxn == "mass":
        w1 = 0; w2 = 1
        cost_obs, mass_obs, obj_obs = weightedcost(t_combo[0], t_combo[1], material1, material2, w1, w2)

    # evaluate flight time for each observation
    t1 = t_combo[0]
    t2 = t_combo[1]
    kb, rhob, cb = blend(t1, t2, material1, material2)
    time_abl = heatshield(cond_maxT, cond_Ti, T_atm, material1.k, material1.rho, material1.c, t1)
    time_pay = heatshield(pay_maxT, pay_Ti, T_atm, kb, rhob, cb, t1+t2)
    time = min(time_abl, time_pay)

    # Penalize observations that do not meet flight duration
    if time < flight_time:
        violations = [t < flight_time for t in [time_abl, time_pay]]
        quad_diff = [(t - flight_time)**2 for t in [time_abl, time_pay]]
        rho_exp = rho_cost
        rho_count = rho_cost/10
        cost_penalty = rho_exp*np.dot(quad_diff, violations) + rho_count*violations.count(True)
        rho_exp = rho_mass
        rho_count = rho_mass/10
        mass_penalty = rho_exp*np.dot(quad_diff, violations) + rho_count*violations.count(True)
        obj_penalty = w1*cost_penalty + w2*mass_penalty
        obj_obs += obj_penalty
    else:
        feasible = True

    return cost_obs, mass_obs, obj_obs, time, feasible


# Gaussian Process fitting, prediction, and exploration for objective function
def runGP(ablator, conductor, pay_Ti, pay_maxT, cond_Ti, T_atm, pred_range, t_obs, min_time, obj_fxn):
    num_evals = 0
    max_evals = 200

    # set material property variables
    k1 = ablator.k; rho1 = ablator.rho; c1 = ablator.c
    k2 = conductor.k; rho2 = conductor.rho; c2 = conductor.c

    # get maximum allowed temp of conductive layer
    cond_maxT = maxtemp(conductor)

    # solve initial observations
    cost_obs = []; mass_obs = []; obj_obs = []; flight_times = []; feasible = []
    for t_combo in t_obs:
        cost, mass, obj, time, valid = constraint(ablator, conductor, pay_Ti, pay_maxT, cond_Ti, cond_maxT, T_atm, t_combo, min_time, obj_fxn) 
        cost_obs.append(cost); mass_obs.append(mass); obj_obs.append(obj);
        flight_times.append(time); feasible.append(valid)

    if feasible.count(True) == 0:
        print("No observations met constraints")
        exit(-1)

    # generate kernel
    end = len(pred_range)-1
    t1_pred_delta = pred_range[end][0]-pred_range[0][0]
    t2_pred_delta = pred_range[end][1]-pred_range[0][1]
    scales = np.array([.1*t1_pred_delta, .1*t2_pred_delta])
    kernel = RBF(length_scale=scales, length_scale_bounds="fixed")
    # generate GP
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(np.array(t_obs), np.array(obj_obs))

    # RUN GAUSSIAN PROCESS
    while num_evals < max_evals:
        # predict
        predictions = gpr.predict(np.array(pred_range), return_std=True)
        # extract values
        mean = predictions[0]
        std = predictions[1]

        # choose next evaluation point
        t_next = Next_Evaluation_Point(pred_range, obj_obs, mean, std, None, "EI")
        if t_next in t_obs or (obj_obs[1] - obj_obs[-2] < 1):
            break
        # print("Next evaluation combo = " + str(t_next))
        t_obs.append(t_next)
        cost, mass, obj, time, valid = constraint(ablator, conductor, pay_Ti, pay_maxT, cond_Ti, cond_maxT, T_atm, t_next, min_time, obj_fxn) 
        cost_obs.append(cost); mass_obs.append(mass); obj_obs.append(obj);
        flight_times.append(time); feasible.append(valid)

        # re-fit
        gpr.fit(np.array(t_obs), np.array(obj_obs))

        num_evals += 1

    # final predictions
    predictions = gpr.predict(np.array(pred_range), return_std=True)
    # extract values
    mean = predictions[0]
    std = predictions[1]
    # get best value
    best_ind = np.argmin(np.array(obj_obs))
    print("Best combo = " + str(t_obs[best_ind]) + ", with cost of " + str(round(cost_obs[best_ind], 2))
        + " and mas of " + str(round(mass_obs[best_ind], 2)))

    # Plot outputs
    plot_flight_durations(np.array(t_obs), np.array(flight_times), ablator, conductor)
    plot_GP(np.array(t_obs), np.array(obj_obs), np.array(pred_range), mean, std, ablator, conductor)

    feas_cost = [cost_obs[i] for i in range(len(cost_obs)) if feasible[i]]
    feas_mass = [mass_obs[i] for i in range(len(mass_obs)) if feasible[i]]
    feas_t = [t_obs[i] for i in range(len(t_obs)) if feasible[i]]
    return mean, std, feas_t, feas_cost, feas_mass


# sample run
if __name__ == '__main__':
    t_obs = [[.1, .1], [.2, .15], [.3, .2], [.4, .25], [.5, .3]]
    pred_range_abl = np.linspace(0, .5, 10)
    pred_range_cond = np.linspace(0, .5, 10)
    pred_range = [[t1, t2] for t1 in pred_range_abl for t2 in pred_range_cond]

    mean, std = runGP(conductors["Ti"], conductors["Al"], 300, 350, 300, 1500, pred_range, t_obs, 3, "cost")


