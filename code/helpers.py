import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfinv
from scipy.stats import norm
from material import Material

ablators = {"LI-2200":Material("LI-2200", .126, 352.5, 628, None, 2500), "PICA":Material("PICA", .3, 144, 1600, None, 4000),
                "AVCOAT":Material("AVCOAT", .164, 530, 3281, None, 3000)}
conductors = {"Ti":Material("Ti", 24.5, 4540, 520, 1668, 34.42), "Al":Material("Al", 193, 2780, 890, 660, 7.25), 
                "SS":Material("SS", 14.4, 7860, 500, 1530, 4.47)}

# cost function to minimize
def totalcost(t1, t2, material1, material2):
    # determine material properties
    return t1*material1.rho*material1.cost + t2*material2.rho*material2.cost

# mass function to minimize
def totalmass(t1, t2, material1, material2):
    return t1*material1.rho + t2*material2.rho

def weightedcost(t1, t2, material1, material2, w_cost, w_mass):
    cost = totalcost(t1, t2, material1, material2)
    mass = totalmass(t1, t2, material1, material2)
    weighted = w_cost*cost + w_mass*mass
    return cost, mass, weighted


# function to give out max temp allowed of material
def maxtemp(material):
    return 0.75*material.maxTemp

# function to blend together material properties
def blend(t1, t2, material1, material2):
    k1 = material1.k; rho1 = material1.rho; c1 = material1.c
    k2 = material2.k; rho2 = material2.rho; c2 = material2.c

    w1 = t1/(t1+t2)
    w2 = t2/(t1+t2)

    kb = k1*w1+k2*w2
    rhob = rho1*w1+rho2*w2
    cb = c1*w1+c2*w2

    return kb, rhob, cb

# function to get minimum thickness for a given heat transfer time
def get_min_thickness(Tmax, Tinitial, Tapplied, k, rho, c, time):
    # Compute heat penetration depth
    alpha = k/(rho*c) # Thermal diffusivity
    theta = (Tmax - Tapplied)/(Tinitial - Tapplied) # Percentage temperature rise
    delta_coeff = 2*erfinv(theta)

    # Compute penetration time
    t_min = np.sqrt(alpha*time*delta_coeff**2)
    return t_min

# heat model
def heatshield(Tmax, Tinitial, Tapplied, k, rho, c, L):
    # Compute heat penetration depth
    alpha = k/(rho*c) # Thermal diffusivity
    theta = (Tmax - Tapplied)/(Tinitial - Tapplied) # Percentage temperature rise
    delta_coeff = 2*erfinv(theta)
    delta = delta_coeff*np.sqrt(L*alpha) # Penetration depth

    # Compute penetration time
    t_max = 1/(delta_coeff**2) * L**2/alpha
    return t_max

# exploration strategy - probability of improvement
def prob_of_improv(y_min, mean, std):
    return norm.cdf(y_min, loc=mean, scale=std)

# exploration strategy - expected improvement
def expected_improv(y_min, mean, std):
    p_imp = prob_of_improv(y_min, mean, std)
    p_ymin = norm.pdf(y_min, loc=mean, scale=std)
    EI = (y_min - mean)*p_imp + (std**2)*p_ymin
    return EI

# exploration strategy - expected improvement
def Next_Point_ProbImprov(objective_obs, pred_range, mean, std):
    y_min = np.argmin(objective_obs)
    prob_improv = prob_of_improv(y_min, mean, std)
    index = np.argmax(prob_improv)
    return pred_range[index]

# Returns next point using expected improvement exploration
def Next_Point_ExpecImprov(objective_obs, pred_range, mean, std):
    index = np.argmin(objective_obs)
    y_min = objective_obs[index]
    EI = expected_improv(y_min, mean, std)
    index2 = np.argmax(EI)
    return pred_range[index2]

# Returns next point using prediction-based exploration
def Next_Point_PredBased(pred_range, mean):
    index = np.argmin(mean)
    return pred_range[index]

# Returns next point using error-based exploration
def Next_Point_ErrBased(pred_range, std):
    index = np.argmax(std)
    return pred_range[index]

# Returns next point using lower confidence bound exploration
def Next_Point_LB(pred_range, mean, std, alpha):
    LB = mean - alpha*std
    index = np.argmin(LB)
    return pred_range[index]

# Returns next evaluation point based on passed in exploration method
def Next_Evaluation_Point(pred_range, objective_obs, mean, std, alpha, method):
    if method == "PB":
        return Next_Point_PredBased(pred_range, mean)
    elif method == "EB":
        return Next_Point_ErrBased(pred_range, std)
    elif method == "LB":
        return Next_Point_LB(pred_range, mean, std, alpha)
    elif method == "EI":
        return Next_Point_ExpecImprov(objective_obs, pred_range, mean, std)
    elif method == "ProbI":
        return Next_Point_ProbImprov(objective_obs, pred_range, mean, std)

# Plot flight duration for each thickness pair
def plot_flight_durations(X, Y, ablator, conductor):
    plt.figure()
    plt.clf()
    ax = plt.axes(projection ='3d')
    ax.scatter(X[:, 0], X[:, 1], Y) # Observation points
    ax.set_title("Flight Durations vs. Thicknesses for\n" + ablator.name + " and " + conductor.name)
    ax.set_xlabel('Ablative Thickness')
    ax.set_ylabel('Conductive Thickness')
    plt.savefig("../plots/flight_durations_" + ablator.name[0] + conductor.name[0] + ".png")
    plt.close()
    
# Plot GP cost predictions for varying thickness combinations
def plot_GP(X, Y, Xhat, Yhat, std, ablator, conductor):
    plt.figure()
    plt.clf()
    ax = plt.axes(projection ='3d')
    ax.scatter(X[:, 0], X[:, 1], Y, c='k', label='Observations', zorder=2) # Observation points
    ax.scatter(Xhat[:, 0], Xhat[:, 1], Yhat, label='Predictions', s=1) # Prediction surface
    ax.set_title("Gaussian Process Cost Predictions for\n"  + ablator.name + " and " + conductor.name)
    ax.set_xlabel('Ablative Thickness')
    ax.set_ylabel('Conductive Thickness')
    ax.legend()
    plt.savefig("../plots/GP_plot_" + ablator.name[0] + conductor.name[0] + ".png")
    plt.close()

