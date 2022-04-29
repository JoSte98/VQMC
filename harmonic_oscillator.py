"""
@author: Magdalena and Johannes
"""

import numpy as np

def ho_trial(parameters, alpha):
    x = parameters[0]
    return np.exp(-alpha * x**2)

def ho_local(parameters, alpha):
    x = parameters[0]
    return alpha + x**2 * (0.5 - 2*alpha**2)

def ho_trial_ln_derivation(parameters,alpha):
    """
    Calculates the value of the derivation of log(\psi_trial) acording to alpha
    """
    x = parameters[0]
    return -x**2

ho_init_alpha = 0.5

ho_dimension = 1