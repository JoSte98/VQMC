"""
@author: Magdalena and Johannes
"""

import numpy as np
from Helium import Helium
from LHO import LHO
from Hydrogen import Hydrogen

import matplotlib.pyplot as plt

class VQMC:
    """
    Variational Quantum Markov Chain class.
    """

    def __init__(self, num_walkers=40, max_step_length=0.6, num_steps_equilibrate=4000, MC_num_steps=10000,
                 model="Helium", init_alpha=None):
        self.model_name = model
        if model == "Helium":
            model = Helium()
        elif model == "Hydrogen":
            model = Hydrogen()
        elif model == "LHO":
            model = LHO()
        else:
            print("Check your model inputs!")
            exit(1)

        self.psi_T = model.trial
        self.energy_L = model.local
        self.energy_L_derivative = model.local_derivative
        self.alpha = model.init_alpha
        self.dimension = model.dimension
        self.derivative_log_trial = model.trial_ln_derivative
        self.derivative_2nd_log_trial = model.trial_ln_2nd_derivative

        if init_alpha is not None:
            self.alpha = init_alpha

        self.num_walkers = num_walkers
        self.chains = [[] for i in range(self.num_walkers)]
        self.initialize_walkers()

        self.max_step_length = max_step_length
        self.energy = []
        
        self.num_tried = 0
        self.num_accepted = 0

        self.num_steps_equilibrate = num_steps_equilibrate
        self.equilibrate(self.num_steps_equilibrate)

        self.MC_num_steps = MC_num_steps

    def reinitialize(self, alpha):
        self.alpha = alpha

        self.chains = [[] for i in range(self.num_walkers)]
        self.initialize_walkers()

        self.energy = []

        self.num_tried = 0
        self.num_accepted = 0

        self.equilibrate(self.num_steps_equilibrate)

    def initialize_walkers(self):
        np.random.seed(42)
        init = np.random.normal(loc=0, scale=10, size=(self.num_walkers,self.dimension))
        self.old_psi_squared = []
        for walker in range(self.num_walkers):
            self.chains[walker].append(init[walker, :])
            self.old_psi_squared.append(self.psi_T(init[walker, :], self.alpha)**2)

        return 0

    def single_walker_step(self, old_state, old_psi_squared):
        
        self.num_tried += 1
        displacement = (2*np.random.rand(self.dimension) - 1)*self.max_step_length
        new_state = old_state + displacement
        new_psi_squared = self.psi_T(new_state, self.alpha)**2

        p = new_psi_squared/old_psi_squared
        if p >= 1.0:
            self.num_accepted +=1
            return new_state, new_psi_squared
        else:
            q = np.random.random()
            if q < p:
                self.num_accepted +=1
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

    def equilibrate(self, num_steps): #NEEDS TO BE REDONE!!!
    
        for i in range(num_steps):
            self.MC_step()
            
        #print(self.chains)
        #print(self.old_psi_squared)

        #print("Total energy: ", tot_energy/((num_steps-4000)*self.num_walkers))
        
    def energy_mean(self):
        """
        Initializes the system and calculates the mean value of energy for a current alpha of the system.

        """
        tot_energy=0
        self.walker_energy = [[] for i in range(self.num_walkers)]
        for i in range(self.MC_num_steps):
            self.MC_step()
            energy = 0
            for walker in range(self.num_walkers):
                E = self.energy_L(self.chains[walker][-1], self.alpha)
                self.walker_energy[walker].append(E)
                energy += E
            self.energy.append(energy/self.num_walkers)
            tot_energy += energy

        print("accepted/tried ratio: ", self.num_accepted / self.num_tried)

        mean_energy_walkers = [np.mean(self.walker_energy[walker]) for walker in range(self.num_walkers)]
        self.variance = np.var(mean_energy_walkers)

        self.chains = [[self.chains[walker][-1]] for walker in range(self.num_walkers)]

        self.expected_energy = tot_energy/((self.MC_num_steps)*self.num_walkers)
        return 0

    def plot_average_local_energies(self):
        """
        Plots the evolution of average local energy of all walkers.

        """
        #for walker in range(self.num_walkers):
        #    plt.plot(range(len(self.energy)), self.walker_energy[walker])

        plt.plot(range(len(self.energy)), self.energy)
        plt.show()
        
    def alpha_energy_dependence(self, stop, steps, start=None, save=True, plot=True):
        """
        Returns lists of alpha and corresponding calculated mean energies of the system. Lenght of the list is given by steps, start and stop values given by start and stop.

        """
        if start == None:
            alpha_start = self.alpha
        else:
            alpha_start = start
            self.reinitialize(alpha_start)
        alpha_stop = stop
        alphas = np.linspace(alpha_start, alpha_stop, steps)
        mean_energies = np.zeros(steps)
        variances = np.zeros(steps)
        for i in range(alphas.size):
            if i != 0:
                self.reinitialize(alphas[i])
                self.energy_mean()
                mean_energies[i] = self.expected_energy
                variances[i] = self.variance
            else:
                self.energy_mean()
                mean_energies[i] = self.expected_energy
                variances[i] = self.variance

        if save:
            self.save_mean_energies(alphas, mean_energies, variances)
        if plot:
            self.plot_alpha_energy_dependence(alphas, mean_energies, variances)
                
        return alphas, mean_energies, variances
    
    def save_mean_energies(self, alphas, mean_energies, variances, name_of_file=None):
        """
        

        """
        if name_of_file==None:
            name_of_file = "alpha-energy_"+str(alphas[0]).replace(".", "")+"_"+str(alphas[-1]).replace(".", "")+".txt"
            
        with open(name_of_file, "a") as file:
            for i in range(alphas.size):
                file.write("%f %f %f\n" % (alphas[i], mean_energies[i], variances[i]))

        return 0    
    
    def load_mean_energies(self,name_of_file):
        """
        
        
        """
        alphas = []
        mean_energies = []
        variances = []
        with open(name_of_file, "r") as file:
            lines = file.read().split('\n')
            del lines[-1]

            for line in lines:
                alpha, mean, var = line.split(" ")
                alphas.append(float(alpha))
                mean_energies.append(float(mean))
                variances.append(float(var))

        return alphas, mean_energies, variances
    
    def plot_alpha_energy_dependence(self, alphas, mean_energies, variances):
        """
        Plots dependence of mean value of energy on parameters

        """
        fig, ax = plt.subplots()

        ax.errorbar(alphas, mean_energies, yerr=np.sqrt(variances), fmt='ro', label="Measurement")

        ax.set_xlabel(r"$\alpha$", fontsize=18)
        ax.set_ylabel(r"Energy", fontsize=18)

        ax.legend(loc="best", fontsize=16)
        ax.grid(visible=True)
        plt.tight_layout()

        plt.show()
        
        return 0









