""" Model class for Hydrogen variational ansatz.
@author: Magdalena and Johannes
"""

import numpy as np

class Hydrogen:
    """
    Model class for Hydrogen variational ansatz.
    """
    def __init__(self):
        """
        Constructor of Model class for Hydrogen variational ansatz.
        """
        self.init_alpha = 1.3
        self.dimension = 3

    def trial(self, parameters, alpha):
        """
        Trial function for 1 electron wave function in Hydrogen atom.

        :param parameters: (np.array [3x1]) 3d position of electron.
        :param alpha: (float) Variational parameter.

        :return: Value of trial function at electron position.
        """
        r = np.linalg.norm(parameters)
        return np.exp(-alpha * r)

    def local(self, parameters, alpha):
        """
        Local energy of Hydrogen ansatz.

        :param parameters: (np.array [3x1]) 3d position of electron.
        :param alpha: (float) Variational parameter.

        :return: Local energy at electron position.
        """
        position = parameters
        r = np.linalg.norm(position)
        return -1/r - alpha/2 *(alpha - 2/r)

    def trial_ln_derivative(self, parameters, alpha):
        """
        Calculates the value of the derivative of log(trial) according to alpha.

        :param parameters: (np.array [3x1]) 3d position of electron.
        :param alpha: (float) Variational parameter.

        :return: Value of d/(d alpha) log(trial) at electron position (here: independent of alpha).
        """
        r = np.linalg.norm(parameters)
        return -r
    
    def trial_ln_2nd_derivative(self, parameters, alpha):
        """
        Calculates the value of the 2nd derivative of log(trial) according to alpha.

        :param parameters: (np.array [3x1]) 3d position of electron.
        :param alpha: (float) Variational parameter.

        :return: Value of d^2/(d alpha^2) log(trial) at electron position (here: independent of alpha and r).
        """
        return 0.0
    
    def local_derivative(self, parameters, alpha):
        r = np.linalg.norm(parameters)
        return - alpha + 1/r



