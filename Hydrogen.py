"""
@author: Magdalena and Johannes
"""
import numpy as np

class Hydrogen:
    def __init__(self):
        self.init_alpha = 1.3
        self.dimension = 3

    def trial(self, parameters, alpha):
        r = np.linalg.norm(parameters)
        return np.exp(-alpha * r)

    def local(self, parameters, alpha):
        position = parameters
        r = np.linalg.norm(position)
        return -1/r - alpha/2 *(alpha - 2/r)

    def trial_ln_derivative(self, parameters, alpha):
        """
        Calculates the value of the derivative of log(\psi_trial) according to alpha.
        """
        r = np.linalg.norm(parameters)
        return -r
    
    def trial_ln_2nd_derivative(self,parameters,alpha):
        
        return 0.0
    
    def local_derivative(self,parameters,alpha):
        r = np.linalg.norm(parameters)
        return - alpha + 1/r



