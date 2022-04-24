"""
@author: Magdalena and Johannes
"""

import numpy as np
from helium import helium_trial, helium_local, helium_init_alpha, helium_dimension

class VQMC:
    """
    Variational Quantum Markoc Chain class.
    """

    def __init__(self, num_walkers=10, trial_function=None, local_energy=None, init_alpha=None, dimension=None):
        if trial_function is None:
            self.psi_T = helium_trial
            self.energy_L = helium_local
            self.init_alpha = helium_init_alpha
            self.dimension = helium_dimension
        elif ((trial_function is not None) and (local_energy is not None) and (init_alpha is not None) and
              (dimension is not None)):
            self.psi_T = trial_function
            self.energy_L = local_energy
            self.init_alpha = init_alpha
            self.dimension = dimension
        else:
            print("Check your model inputs!")
            exit(1)

        self.num_walkers = num_walkers
        self.chains = [[] for i in range(self.num_walkers)]
        self.initialize_walkers()


    def initialize_walkers(self):
        pass





