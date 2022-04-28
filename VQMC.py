"""
@author: Magdalena and Johannes
"""

import numpy as np
from helium import helium_trial, helium_local, helium_init_alpha, helium_dimension

import matplotlib.pyplot as plt

class VQMC:
    """
    Variational Quantum Markov Chain class.
    """

    def __init__(self, num_walkers=10, max_step_length=0.1, num_steps_equilibrate=10000, trial_function=None,
                 local_energy=None, init_alpha=None, dimension=None):
        if trial_function is None:
            self.psi_T = helium_trial
            self.energy_L = helium_local
            self.alpha = helium_init_alpha
            self.dimension = helium_dimension
        elif ((trial_function is not None) and (local_energy is not None) and (init_alpha is not None) and
              (dimension is not None)):
            self.psi_T = trial_function
            self.energy_L = local_energy
            self.alpha = init_alpha
            self.dimension = dimension
        else:
            print("Check your model inputs!")
            exit(1)

        self.num_walkers = num_walkers
        self.chains = [[] for i in range(self.num_walkers)]
        self.initialize_walkers()

        self.max_step_length = max_step_length
        self.energy = []

        self.equilibrate(num_steps_equilibrate)




    def initialize_walkers(self):
        np.random.seed(42)
        init = np.random.normal(loc=0, scale=10, size=(self.num_walkers,self.dimension))
        self.old_psi_squared = []
        for walker in range(self.num_walkers):
            self.chains[walker].append(init[walker, :])
            self.old_psi_squared.append(self.psi_T(init[walker, :], self.alpha)**2)

        return 0

    def single_walker_step(self, old_state, old_psi_squared):
        displacement = (2*np.random.rand(self.dimension) - 1)*self.max_step_length
        new_state = old_state + displacement
        new_psi_squared = self.psi_T(new_state, self.alpha)**2

        p = new_psi_squared/old_psi_squared
        if p >= 1.0:
            return new_state, new_psi_squared
        else:
            q = np.random.random()
            if q >= p:
                return new_state, new_psi_squared
            else:
                return old_state, old_psi_squared

    def MC_step(self):
        for walker in range(self.num_walkers):
            #if walker == 0:
            #    print(self.old_psi_squared[walker])
            #    print(self.chains[walker])

            new_state, self.old_psi_squared[walker] = \
                self.single_walker_step(self.chains[walker][-1], self.old_psi_squared[walker])
            self.chains[walker].append(new_state)

    def equilibrate(self, num_steps):
        tot_energy=0
        self.walker_energy = [[] for i in range(self.num_walkers)]
        for i in range(num_steps):
            self.MC_step()
            energy = 0
            for walker in range(self.num_walkers):
                E = self.energy_L(self.chains[walker][-1], self.alpha)
                self.walker_energy[walker].append(E)
                energy += E
            self.energy.append(energy/self.num_walkers)

            if i > 4000:
               tot_energy += energy

        self.chains = [[self.chains[walker][-1]] for walker in range(self.num_walkers)]

        print(self.chains)
        print(self.old_psi_squared)

        print("Total energy: ", tot_energy/((num_steps-4000)*self.num_walkers))

    def plot_energy(self):
        #for walker in range(self.num_walkers):
        #    plt.plot(range(len(self.energy)), self.walker_energy[walker])

        plt.plot(range(len(self.energy)), self.energy)
        plt.show()










