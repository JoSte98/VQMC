""" Model class for Linear Harmonic Oscillator variational ansatz.
@author: Magdalena and Johannes
"""

import numpy as np

class LHO:
    """
    Model class for Linear Harmonic Oscillator variational ansatz.
    """
    def __init__(self):
        """
        Constructor of Model class for Linear Harmonic Oscillator variational ansatz.
        """
        self.init_alpha = 0.3
        self.dimension = 1

    def trial(self, parameters, alpha):
        """
        Trial function for 1 electron wave function in LHO.

        :param parameters: (np.array [1,]) 1d positions of electron.
        :param alpha: (float) Variational parameter.

        :return: Value of trial function at electron position.
        """
        x = parameters[0]
        return np.exp(-alpha * x**2)

    def local(self, parameters, alpha):
        """
        Local energy E_L of LHO ansatz.

        :param parameters: (np.array [1,]) 1d position of electron.
        :param alpha: (float) Variational parameter.

        :return: Local energy at electron position.
        """
        x = parameters[0]
        return alpha + x**2 * (0.5 - 2*alpha**2)

    def trial_ln_derivative(self, parameters, alpha):
        """
        Calculates the value of the derivative of log(trial) according to alpha.

        :param parameters: (np.array [1,]) 1d position of electron.
        :param alpha: (float) Variational parameter.

        :return: Value of d/(d alpha) log(trial) at electron position (here: independent of alpha).
        """
        x = parameters[0]
        return -x**2
    
    def trial_ln_2nd_derivative(self, parameters, alpha):
        """
        Calculates the value of the 2nd derivative of log(trial) according to alpha.

        :param parameters: (np.array [1,]) 1d position of electron.
        :param alpha: (float) Variational parameter.

        :return: Value of d^2/(d alpha^2) log(trial) at electron position (here: independent of alpha and r).
        """

        return 0.0
    
    def local_derivative(self, parameters, alpha):
        """
        Calculates the value of the derivative of local energy E_L according to alpha.

        :param parameters: (np.array [1,]) 1d position of electron.
        :param alpha: (float) Variational parameter.

        :return: Value of d/(d alpha) E_L at electron position.
        """
        
        x = parameters[0]
        return -4 * x**2 * alpha
    
    def force(self, parameters, alpha):
        """
        Calculates the the Langevin force.

        :param parameters: (np.array [1,]) 1d position of electron.
        :param alpha: (float) Variational parameter.

        :return: Value of the Langevin force at electron position.
        """
        x = parameters[0]
        return -4 * x * alpha
        