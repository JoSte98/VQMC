"""
@author: Magdalena and Johannes
"""

import numpy as np

class LHO:
    def __init__(self):
        self.init_alpha = 0.3
        self.dimension = 1

    def trial(self, parameters, alpha):
        x = parameters[0]
        return np.exp(-alpha * x**2)

    def local(self, parameters, alpha):
        x = parameters[0]
        return alpha + x**2 * (0.5 - 2*alpha**2)

    def trial_ln_derivative(self, parameters, alpha):
        """
        Calculates the value of the derivation of log(\psi_trial) acording to alpha
        """
        x = parameters[0]
        return -x**2
    
    def trial_ln_2nd_derivative(self,parameters,alpha):
        
        return 0.0
    
    def local_derivative(self,parameters,alpha):
        
        x = parameters[0]
        return -4 * x**2 * alpha