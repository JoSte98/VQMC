""" Model class for Helium variational ansatz.
@author: Magdalena and Johannes
"""

import numpy as np

class Helium:
    """
    Model class for Hydrogen variational ansatz.
    """
    def __init__(self):
        """
        Constructor of Model class for Hydrogen variational ansatz.
        """
        self.init_alpha = 0.25
        self.dimension = 6

    def trial(self, parameters, alpha):
        """
        Trial function for 2 electrons wave function in Helium atom.

        :param parameters: (np.array [6x1]) 3d positions of 2 electrons.
        :param alpha: (float) Variational parameter.

        :return: Value of trial function at electron positions.
        """
        position_1 = parameters[:3]
        position_2 = parameters[3:]
        r12 = np.linalg.norm(position_1-position_2)
        return np.exp(-2*np.linalg.norm(position_1) - 2*np.linalg.norm(position_2) + r12/(2*(1 + alpha*r12)))

    def local(self, parameters, alpha):
        """
        Local energy of Helium ansatz.

        :param parameters: (np.array [6x1]) 3d position of 2 electrons.
        :param alpha: (float) Variational parameter.

        :return: Local energy E_L at electron positions.
        """
        position_1 = parameters[:3]
        position_2 = parameters[3:]
        r_hat_1 = position_1/np.linalg.norm(position_1)
        r_hat_2 = position_2/np.linalg.norm(position_2)
        r12 = np.linalg.norm(position_1 - position_2)
        return -4 + np.dot((r_hat_1 - r_hat_2),(position_1 - position_2)) * (1/(r12*(1 + alpha*r12)**2)) \
                - 1/(r12*(1 + alpha*r12)**3) - 1/(4*(1 + alpha*r12)**4) + 1/r12

    def trial_ln_derivative(self, parameters,alpha):
        """
        Calculates the value of the derivative of log(trial) according to alpha.

        :param parameters: (np.array [6x1]) 3d position of 2 electrons.
        :param alpha: (float) Variational parameter.

        :return: Value of d/(d alpha) log(trial) at electron positions.
        """
        position_1 = parameters[:3]
        position_2 = parameters[3:]
        r12 = np.linalg.norm(position_1-position_2)
        return - r12**2 / (2*(1+alpha*r12)**2)
    
    def trial_ln_2nd_derivative(self, parameters, alpha):
        """
        Calculates the value of the 2nd derivative of log(trial) according to alpha.

        :param parameters: (np.array [6x1]) 3d position of 2 electrons.
        :param alpha: (float) Variational parameter.

        :return: Value of d^2/(d alpha^2) log(trial) at electron positions.
        """
        position_1 = parameters[:3]
        position_2 = parameters[3:]
        r12 = np.linalg.norm(position_1-position_2)
        return r12**3 / ((1+alpha*r12)**3)
    
    def local_derivative(self,parameters,alpha):
        """
        Calculates the value of the derivative of local energy E_L according to alpha.

        :param parameters: (np.array [6x1]) 3d position of 2 electrons.
        :param alpha: (float) Variational parameter.

        :return: Value of d/(d alpha) E_L at electron positions.
        """

        position_1 = parameters[:3]
        position_2 = parameters[3:]
        r_hat_1 = position_1/np.linalg.norm(position_1)
        r_hat_2 = position_2/np.linalg.norm(position_2)
        r12 = np.linalg.norm(position_1 - position_2)
        return -2 * np.dot((r_hat_1 - r_hat_2),(position_1 - position_2)) * (1/((1 + alpha*r12)**3)) \
                + 3/((1 + alpha*r12)**4) - r12/((1 + alpha*r12)**5)
                
    def force(self, parameters, alpha):
        """
        Calculates the the Langevin force.

        :param parameters: (np.array [6x1]) 3d position of 2 electrons.
        :param alpha: (float) Variational parameter.

        :return: Langevin force at electron positions.
        """
        position_1 = parameters[:3]
        position_2 = parameters[3:]
        r_1 = np.linalg.norm(position_1)
        r_2 = np.linalg.norm(position_2)
        position_12 = position_1 - position_2
        r12 = np.linalg.norm(position_1 - position_2)
        F = np.zeros(6)
        F[:3] = (-2*position_1/r_1 + alpha*position_12/(1+alpha*r12)/r12)
        F[3:] = (-2*position_2/r_2 - alpha*position_12/(1+alpha*r12)/r12)
        return 2*F
