import numpy as np

from optimizationscript import *
import matplotlib.pyplot as plt

## Define problem parameters
num_obs = 5
num_pred = 100
num_plots = len(ablators) * len(conductors)

# constraints
payload_maxT = 325
payload_Ti = 295
conductor_Ti = 273
T_atm = 3033
flight_time = 30

# create combination arrays
combos = []

# Create multi-objective figure
plt.figure('Mass vs. Cost')
plt.clf()
ax = plt.axes()
cm = plt.get_cmap('gist_rainbow')
ax.set_prop_cycle(color=[cm(1.*i/num_plots) for i in range(num_plots)])

# Run optimization
for ablator in ablators.values():
    for conductor in conductors.values():

        # generate exploration range
        min_t1 = get_min_thickness(maxtemp(conductor), conductor_Ti, T_atm, ablator.k, ablator.rho, ablator.c, flight_time)
        max_t1 = get_min_thickness(payload_maxT, payload_Ti, T_atm, ablator.k, ablator.rho, ablator.c, flight_time)
        min_t2 = 0;
        max_t2 = get_min_thickness(payload_maxT, payload_Ti, T_atm, conductor.k, conductor.rho, conductor.c, flight_time)

        # initial observations
        ablator_obs = np.linspace(min_t1, max_t1, num_obs)
        conductor_obs = np.linspace(min_t2, max_t2, num_obs)
        observed_thicknesses = [[t1, t2] for t1 in ablator_obs for t2 in conductor_obs]

        # prediction range
        desired_pred_ablator = np.linspace(min_t1, max_t1, num_pred)  # x to predict on for GP
        desired_pred_cond = np.linspace(min_t2, max_t2, num_pred)
        desired_predictions = [[t1, t2] for t1 in desired_pred_ablator for t2 in desired_pred_cond]

        combo = [ablator, conductor]
        combos.append(combo)

        # Cost optimization
        print("Optimizing " + ablator.name + ", " + conductor.name + " wrt cost...")
        cost_mean, cost_std, t_obs_1, cost_obs_1, mass_obs_1 = runGP(ablator, conductor, payload_Ti, payload_maxT, conductor_Ti, T_atm, desired_predictions,
                                                    observed_thicknesses.copy(), flight_time, "cost")

        # Mass optmization
        print("Optimizing " + ablator.name + ", " + conductor.name + " wrt mass...")
        mass_mean, mass_std, t_obs_2, cost_obs_2, mass_obs_2 = runGP(ablator, conductor, payload_Ti, payload_maxT, conductor_Ti, T_atm, desired_predictions,
                                                    observed_thicknesses.copy(), flight_time, "mass")

        # Combine data, ignoring repeat points
        itr = len(t_obs_1)
        print("Converged with " + str(itr) + " feasible evaluations for cost analysis")
        [cost_obs_1.append(cost_obs_2[i]) for i in range(len(cost_obs_2)) if t_obs_2[i] not in t_obs_1]
        [mass_obs_1.append(mass_obs_2[i]) for i in range(len(mass_obs_2)) if t_obs_2[i] not in t_obs_1]
        [t_obs_1.append(t_obs_2[i]) for i in range(len(t_obs_2)) if t_obs_2[i] not in t_obs_1]
        itr = len(t_obs_1)  - itr
        print("Converged with " + str(itr) + " feasible evaluations for mass analysis")
        print()
        # Add scatter to figure
        plt.figure('Mass vs. Cost')
        ax.scatter(mass_obs_1, cost_obs_1, label=ablator.name+", "+conductor.name)


# Format plot
plt.figure('Mass vs. Cost')
ax.set_xlabel("Mass")
ax.set_ylabel("Cost")
ax.legend(loc="best")
plt.savefig('../plots/mass_v_cost.png')
plt.show()
