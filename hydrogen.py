"""
@author: Magdalena and Johannes
"""
import numpy as np


def hydrogen_trial(parameters, alpha):
    r = np.linalg.norm(parameters)
    return np.exp(-alpha * r)

def hydrogen_local(parameters, alpha):
    position = parameters
    r = np.linalg.norm(position)
    return -1/r - alpha/2 *(alpha - 2/r)
            
def hydrogen_trial_ln_derivation(parameters,alpha):
    """
    Calculates the value of the derivation of log(\psi_trial) acording to alpha
    """
    r = np.linalg.norm(parameters)
    return -r

hydrogen_init_alpha = 1.0

hydrogen_dimension = 3