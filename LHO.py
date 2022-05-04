"""
@author: Magdalena and Johannes
"""

import numpy as np

class LHO:
    def __init__(self):
        self.init_alpha = 0.1
        self.dimension = 1

    def trial(self, parameters, alpha):
        x = parameters[0]
        return np.exp(-alpha * x**2)

    def local(self, parameters, alpha):
        x = parameters[0]
        return alpha + x**2 * (0.5 - 2*alpha**2)

    def trial_ln_derivation(self, parameters,alpha):
        """
        Calculates the value of the derivation of log(\psi_trial) acording to alpha
        """
        x = parameters[0]
        return -x**2