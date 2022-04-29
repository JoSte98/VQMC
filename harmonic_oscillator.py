import numpy as np

def ho_trial(parameters, alpha):
    x = parameters[0]
    return np.exp(-alpha * x**2)

def ho_local(parameters, alpha):
    x = parameters[0]
    return alpha + x**2 * (0.5 - 2*alpha**2)

ho_init_alpha = 0.5

ho_dimension = 1